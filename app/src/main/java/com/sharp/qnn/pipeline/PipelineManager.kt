package com.sharp.qnn.pipeline

import android.content.Context
import android.net.Uri
import com.sharp.qnn.R
import com.sharp.qnn.data.ModelEntry
import com.sharp.qnn.data.ModelFormat
import com.sharp.qnn.data.ModelStatus
import com.sharp.qnn.data.ModelType
import com.sharp.qnn.data.SettingsRepository
import com.sharp.qnn.data.ModelStore
import com.sharp.qnn.util.FileUtil
import com.sharp.qnn.util.FileUtil.ensureDir
import com.sharp.qnn.util.LocaleUtil
import com.sharp.qnn.util.MsgKey
import com.sharp.qnn.util.SkelExtractor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException

/**
 * Pipeline 编排器。
 * Pipeline orchestrator.
 *
 * Chains the full inference flow:
 * prepImage → runPre → PE → IE → Merge → REST(A/B/C) → Post(PLY)
 *
 * 通过 [ProgressCallback] 接收 native 层进度，并以 [StateFlow] 暴露给 UI。
 * Receives native progress through [ProgressCallback] and exposes it to the
 * UI as a [StateFlow].
 * 每步检查模型是否已编译并加载，DLC 需先编译。
 * Each step checks that models are compiled and loaded; DLCs must be compiled first.
 */
class PipelineManager(
    private val context: Context,
    private val modelStore: ModelStore,
    private val settings: SettingsRepository
) : ProgressCallback {

    private val _state = MutableStateFlow(PipelineState())
    val state: StateFlow<PipelineState> = _state.asStateFlow()

    private val runMutex = Mutex()

    /** Active inference backend (null until the first init) */
    private val _backend = MutableStateFlow<BackendInfo?>(null)
    val backend: StateFlow<BackendInfo?> = _backend.asStateFlow()

    /** True when the CPU (ONNX Runtime) fallback is active */
    val isCpuBackend: Boolean get() = _backend.value?.isCpu == true

    // Per-run model load timings (ms), for the benchmark summary
    private val modelLoadMs = linkedMapOf<ModelType, Long>()

    // Runtime state
    @Volatile
    private var initialized = false
    @Volatile
    private var backendStale = false
    private val loadedModels = mutableSetOf<ModelType>()
    private val loadedModelsLock = Any()
    private var pipelineStartTime = 0L
    private var stageStartTime = 0L
    private var compileStageStartTime = 0L

    // Locale-wrapped context for this run (consistent for the whole run,
    // even if the language setting changes mid-run)
    private var langCtx: Context = context

    // prepImage output (used by the later Post stage)
    private var prepMeta: FloatArray? = null

    /** Localized model name for the current run's language */
    private fun modelName(type: ModelType): String = LocaleUtil.modelName(langCtx, type)

    /**
     * 运行完整 Pipeline。
     * Run the full pipeline.
     * @param imageUri 用户选择的图片 URI
     * @param imageUri the image URI selected by the user
     * @param overrideFpx optional user-overridden focal length in pixels (null = auto-detect from EXIF)
     * @return 成功与否
     * @return success/failure
     */
    suspend fun runPipeline(imageUri: Uri, overrideFpx: Float? = null): Result<Unit> = runMutex.withLock {
        withContext(Dispatchers.IO) {
            try {
                // Pin the language for this run (detail messages use it)
                val plySettings = settings.settingsFlow.first()
                langCtx = LocaleUtil.wrap(context, plySettings.language)

                // Reset state (loadedModels is kept; already-loaded models are reused)
                pipelineStartTime = System.currentTimeMillis()
                val hasUncompiled = ModelType.coreTypes.any { type ->
                    modelStore.getModel(type)?.let {
                        it.format == ModelFormat.DLC && it.status != ModelStatus.COMPILED
                    } == true
                }
                val stages = {
                    val base = if (hasUncompiled) {
                        val initIdx = DEFAULT_STAGES.indexOfFirst { it.id == PREPARE_STAGE_ID }
                        DEFAULT_STAGES.toMutableList().apply {
                            add(initIdx + 1, StageState(id = COMPILE_STAGE_ID, name = "Model Compilation", nameRes = R.string.stage_compile))
                        }
                    } else {
                        DEFAULT_STAGES.toMutableList()
                    }
                    // Decide PLY optimization inclusion based on settings, and set stage name dynamically
                    if (plySettings.plyOptimize) {
                        val parts = mutableListOf<String>()
                        if (plySettings.plyPruneThreshold > 0.0) parts.add("Prune")
                        if (plySettings.plySorNeighbors > 0) parts.add("SOR")
                        if (plySettings.plyMergeRatio in 0.001..0.999) parts.add("kNN")
                        base[base.indexOfFirst { it.id == PLY_OPTIMIZE_STAGE_ID }] =
                            StageState(id = PLY_OPTIMIZE_STAGE_ID, name = "PLY Optimize", nameRes = R.string.stage_ply_optimize, total = parts.size.coerceAtLeast(1))
                    } else {
                        base.removeAll { it.id == PLY_OPTIMIZE_STAGE_ID }
                    }
                    base
                }()
                _state.value = PipelineState(stages = stages, isRunning = true)

                // Start the init stage immediately so the user sees progress
                onStageStart(PREPARE_STAGE_ID, "Initialization")
                onProgress(PREPARE_STAGE_ID, 0, ModelType.coreTypes.size, 0, MsgKey.DETAIL_INIT_QNN)

                // Register the callback
                runCatching { QnnJni.setProgressCallback(this@PipelineManager) }

                // 1. Initialize the backend (HTP preferred, CPU fallback) and compile any DLC models
                QnnJni.resetPeakRss()
                modelLoadMs.clear()
                ensureReadyForCompile()
                if (isCpuBackend) {
                    _state.update { st -> st.copy(stages = st.stages.filter { it.id != COMPILE_STAGE_ID }) }
                }

                // Init stage done
                onStageComplete(PREPARE_STAGE_ID, "Initialization", System.currentTimeMillis() - pipelineStartTime)

                // 2. Prepare the work directory
                val workDir = File(context.cacheDir, "sharp_work").also {
                    if (it.exists()) FileUtil.deleteRecursively(it)
                    ensureDir(it)
                }

                // 3. Copy the input image
                val fileName = FileUtil.getFileNameFromUri(context, imageUri)
                android.util.Log.i(TAG, "=== Pipeline started, image=$fileName ===")
                val ext = fileName.substringAfterLast('.', "jpg").lowercase()
                val inputFile = File(workDir, "input.$ext")
                if (!FileUtil.copyUriToFile(context, imageUri, inputFile)) {
                    fail(MsgKey.ERR_COPY_IMAGE_FAILED)
                    return@withContext Result.failure(IOException("copy image failed"))
                }

                // 4. Stage 0: decode the image (prepImage)
                val imageRaw = File(workDir, "image.raw")
                val meta = runStage(0) {
                    QnnJni.prepImage(inputFile.absolutePath, imageRaw.absolutePath)
                } ?: run {
                    fail(MsgKey.ERR_PREP_NULL)
                    return@withContext Result.failure(IOException("prepImage failed"))
                }
                prepMeta = meta

                // 5. Stage 1: pre-process and split patches (runPre)
                if (!runStage(1) { QnnJni.runPre(imageRaw.absolutePath, workDir.absolutePath) }) {
                    fail(MsgKey.ERR_PRE_FAILED)
                    return@withContext Result.failure(IOException("runPre failed"))
                }

                // 6. Stage 2: PE inference (load model on-demand)
                ensureModelLoaded(ModelType.PE)
                if (!runStage(2) { QnnJni.runPatchEncoder(workDir.absolutePath) }) {
                    fail(MsgKey.ERR_PE_INFER_FAILED)
                    return@withContext Result.failure(IOException("runPatchEncoder failed"))
                }
                // All PE outputs are on disk and the model is no longer needed:
                // release its context memory immediately (QnnContext_free)
                android.util.Log.i(TAG, "PE done, releasing PE model")
                runCatching { QnnJni.freeModel(ModelType.PE.code) }
                synchronized(loadedModelsLock) { loadedModels.remove(ModelType.PE) }

                // 7. Stage 3: IE inference (load model on-demand)
                ensureModelLoaded(ModelType.IE)
                if (!runStage(3) { QnnJni.runImageEncoder(workDir.absolutePath) }) {
                    fail(MsgKey.ERR_IE_INFER_FAILED)
                    return@withContext Result.failure(IOException("runImageEncoder failed"))
                }
                android.util.Log.i(TAG, "IE done, releasing IE model")
                runCatching { QnnJni.freeModel(ModelType.IE.code) }
                synchronized(loadedModelsLock) { loadedModels.remove(ModelType.IE) }

                // 8. Stage 4: Merge
                // 8. Stage 4: Merge
                // Note: directory names must match the JNI layer (sharp_jni.cpp): out_pe / out_ie
                val peOutDir = File(workDir, "out_pe")
                val ieOutDir = File(workDir, "out_ie")
                if (!runStage(4) { QnnJni.runMerge(workDir.absolutePath, peOutDir.absolutePath, ieOutDir.absolutePath) }) {
                    fail(MsgKey.ERR_MERGE_FAILED)
                    return@withContext Result.failure(IOException("runMerge failed"))
                }

                // 9. Stage 5: REST Seg A (feature fusion, load model on-demand)
                ensureModelLoaded(ModelType.REST_A)
                if (!runStage(5) { QnnJni.runRestSegA(workDir.absolutePath) }) {
                    fail(MsgKey.ERR_REST_A_FAILED)
                    return@withContext Result.failure(IOException("runRestSegA failed"))
                }
                android.util.Log.i(TAG, "REST A done, releasing REST_A model")
                runCatching { QnnJni.freeModel(ModelType.REST_A.code) }
                synchronized(loadedModelsLock) { loadedModels.remove(ModelType.REST_A) }

                // 10. Stage 6: REST Seg B (disparity estimation, load model on-demand)
                ensureModelLoaded(ModelType.REST_B)
                if (!runStage(6) { QnnJni.runRestSegB(workDir.absolutePath) }) {
                    fail(MsgKey.ERR_REST_B_FAILED)
                    return@withContext Result.failure(IOException("runRestSegB failed"))
                }
                android.util.Log.i(TAG, "REST B done, releasing REST_B model")
                runCatching { QnnJni.freeModel(ModelType.REST_B.code) }
                synchronized(loadedModelsLock) { loadedModels.remove(ModelType.REST_B) }

                // 11. Stage 7: REST Seg C (Gaussian increment, load model on-demand)
                ensureModelLoaded(ModelType.REST_C)
                val fpx = overrideFpx ?: (prepMeta?.getOrNull(0) ?: 0f)
                val origW = (prepMeta?.getOrNull(2) ?: 0f).toInt()
                val origH = (prepMeta?.getOrNull(3) ?: 0f).toInt()
                if (!runStage(7) { QnnJni.runRestSegC(workDir.absolutePath, fpx, origW) }) {
                    fail(MsgKey.ERR_REST_C_FAILED)
                    return@withContext Result.failure(IOException("runRestSegC failed"))
                }
                // REST C output is on disk (delta.raw); the model is no longer needed
                android.util.Log.i(TAG, "REST C done, releasing REST_C model")
                runCatching { QnnJni.freeModel(ModelType.REST_C.code) }
                synchronized(loadedModelsLock) { loadedModels.remove(ModelType.REST_C) }

                // 12. Stage 8: Post (PLY generation)
                val plyPath = File(workDir, "output.ply").absolutePath
                if (!runStage(8) { QnnJni.runPost(workDir.absolutePath, fpx, origW, origH, plyPath) }) {
                    fail(MsgKey.ERR_POST_FAILED)
                    return@withContext Result.failure(IOException("runPost failed"))
                }

                // Debug: dump raw model outputs (before in-place PLY optimization)
                // (its copy time is excluded from the run's total)
                var dumpMs = 0L
                if (plySettings.dumpTensors) {
                    val t0 = System.currentTimeMillis()
                    runCatching { dumpTensors(workDir, fpx, origW, origH) }
                        .onFailure { android.util.Log.w(TAG, "Tensor dump failed", it) }
                    dumpMs = System.currentTimeMillis() - t0
                }

                // SOR / kNN Merge)
                if (plySettings.plyOptimize) {
                    android.util.Log.i(TAG, "PLY optimization enabled, running GaussSimplify...")
                    if (!runStage(9) {
                        QnnJni.nativeOptimizePly(
                            plyPath = plyPath,
                            mergeK = plySettings.plyMergeK,
                            mergeRatio = plySettings.plyMergeRatio,
                            mergeCap = plySettings.plyMergeCap,
                            pruneThreshold = plySettings.plyPruneThreshold,
                            sorNeighbors = plySettings.plySorNeighbors,
                            sorStdRatio = plySettings.plySorStdRatio
                        )
                    }) {
                        android.util.Log.w(TAG, "PLY optimization failed, but PLY is still usable")
                        // Optimization failure is non-fatal; the PLY is still usable
                    }
                } else {
                    android.util.Log.i(TAG, "PLY optimization disabled, skipping")
                }

                // Done
                val totalElapsed = System.currentTimeMillis() - pipelineStartTime - dumpMs
                android.util.Log.i(TAG, "=== Pipeline done, total ${totalElapsed}ms, PLY=$plyPath ===")
                val peakMb = QnnJni.getPeakRssKb() / 1024
                android.util.Log.i(BENCH_TAG, benchSummary(totalElapsed, peakMb))
                _state.value = _state.value.copy(
                    isRunning = false,
                    totalElapsedMs = totalElapsed,
                    errorMessage = null,
                    peakRssMb = peakMb
                )
                Result.success(Unit)
            } catch (e: Exception) {
                fail(e.message ?: e.toString())
                Result.failure(e)
            }
        }
    }

    /** Cancel and reset the current pipeline state */
    fun reset() {
        _state.value = PipelineState()
        // Release the HTP memory of loaded models
        synchronized(loadedModelsLock) {
            for (modelType in loadedModels) {
                runCatching { QnnJni.freeModel(modelType.code) }
            }
            loadedModels.clear()
        }
        runCatching { QnnJni.clearRestCache() }
    }

    /** Path of the PLY file from the latest run (for export) */
    fun getLastPlyFile(): File? {
        val workDir = File(context.cacheDir, "sharp_work")
        val ply = File(workDir, "output.ply")
        return if (ply.exists()) ply else null
    }

    // ======  ======
    // ====== Internals: initialization and model readiness ======

    /**
     * Ensure an inference backend is initialized (idempotent). Also required before
     * standalone compilation from the models tab.
     *
     * Capability probe (no SoC-name gate): HTP is tried first unless the user forced
     * CPU; any failure (no FastRPC lib, no HTP arch reported, missing skel, native
     * init error) is logged under [BACKEND_TAG] and the ONNX Runtime CPU backend is
     * used instead. With the override set to HTP, an HTP failure is an error.
     */
    suspend fun ensureQnnInitialized() {
        if (initialized && !backendStale) return
        if (backendStale) {
            // Settings changed the backend/threads: tear down so it is re-probed below
            synchronized(loadedModelsLock) { loadedModels.clear() }
            runCatching { QnnJni.nativeDestroy() }
            initialized = false
            backendStale = false
            _backend.value = null
        }
        // Notify the init stage (only effective in pipeline context; harmless otherwise)
        onProgress(PREPARE_STAGE_ID, 0, ModelType.coreTypes.size, 0, MsgKey.DETAIL_INIT_QNN)
        val s = settings.settingsFlow.first()
        // Pin the language for this call (compile progress details use it)
        langCtx = LocaleUtil.wrap(context, s.language)
        val override = s.backendOverride

        val htpFailure: String? = if (override == SettingsRepository.BackendOverride.CPU) {
            "forced CPU in settings"
        } else {
            tryInitHtp(s)
        }
        if (htpFailure == null) {
            _backend.value = BackendInfo(BackendKind.HTP, name = "HTP", threads = 0, fallbackReason = null)
            android.util.Log.i(BACKEND_TAG, "backend=HTP")
            initialized = true
            return
        }
        if (override == SettingsRepository.BackendOverride.HTP) {
            android.util.Log.e(BACKEND_TAG, "HTP forced but unavailable: $htpFailure")
            throw IOException(MsgKey.ERR_QNN_INIT)
        }
        android.util.Log.w(BACKEND_TAG, "HTP unavailable ($htpFailure) -> CPU fallback (ONNX Runtime)")
        val ok = runCatching { QnnJni.nativeInitOrt(s.cpuThreads) }.getOrDefault(false)
        if (!ok) throw IOException(MsgKey.ERR_QNN_NATIVE_INIT)
        val info = BackendInfo(BackendKind.CPU, QnnJni.getBackendName(), QnnJni.getOrtThreads(), htpFailure)
        _backend.value = info
        android.util.Log.i(BACKEND_TAG, "backend=${info.name} threads=${info.threads} reason=\"$htpFailure\"")
        initialized = true
    }

    /**
     * Compile a DLC model into an HTP context binary (models tab). Initializes the
     * backend first; fails with [MsgKey.ERR_NEEDS_NPU] on the CPU backend.
     */
    suspend fun compileDlc(type: ModelType): Result<Unit> {
        val init = runCatching { ensureQnnInitialized() }
        if (init.isFailure) {
            return Result.failure(init.exceptionOrNull() ?: RuntimeException(MsgKey.ERR_QNN_INIT_DEFAULT))
        }
        if (isCpuBackend) return Result.failure(RuntimeException(MsgKey.ERR_NEEDS_NPU))
        return modelStore.compileModel(type)
    }

    /** Try to bring up QNN HTP. Returns null on success, or the failure reason. */
    private fun tryInitHtp(s: SettingsRepository.Settings): String? = try {
        if (!hasFastRpcLib()) {
            "libcdsprpc.so not present (no Qualcomm DSP)"
        } else {
            val libDir = context.applicationInfo.nativeLibraryDir
            // The skel must match the device architecture exactly
            val arch = QnnJni.probeHtpArch(libDir)
            if (arch.isBlank()) {
                "probeHtpArch: no HTP device reported"
            } else {
                // The HTP perf config must be sent before nativeInit
                QnnJni.setPerfConfig(
                    type = s.perfType,
                    lockedCorner = s.perfLockedCorner,
                    minCorner = s.perfRangeMin,
                    targetCorner = s.perfRangeTarget,
                    maxCorner = s.perfRangeMax,
                    dcvsMode = s.perfDcvsMode
                )
                val skelDir = SkelExtractor.extractSkel(context, arch)
                when {
                    skelDir == null -> "no skel library for HTP $arch"
                    !QnnJni.nativeInit(libDir, skelDir, arch) -> "QNN HTP nativeInit failed (arch=$arch)"
                    else -> {
                        android.util.Log.i(BACKEND_TAG, "HTP arch=$arch")
                        null
                    }
                }
            }
        }
    } catch (t: Throwable) {
        "HTP probe threw: ${t.javaClass.simpleName}: ${t.message}"
    }

    /** Qualcomm FastRPC user library (needed by the HTP backend) present on this device */
    private fun hasFastRpcLib(): Boolean =
        listOf("/vendor/lib64/libcdsprpc.so", "/system/vendor/lib64/libcdsprpc.so", "/system/lib64/libcdsprpc.so")
            .any { File(it).exists() }

    /**
     * Mark the backend for re-probing at the next init (after the backend override or
     * CPU thread count changed). The teardown happens there, on the caller's IO thread.
     */
    fun invalidateBackend() {
        backendStale = true
    }

    private suspend fun ensureReadyForCompile() {
        ensureQnnInitialized()

        if (isCpuBackend) {
            // CPU backend: no DLC compilation; the ONNX set must be present
            val missing = modelStore.missingOnnxModels()
            if (missing.isNotEmpty()) {
                throw IOException(
                    "CPU backend: missing ONNX models ${missing.joinToString { it.code + ".onnx" }} in " +
                        modelStore.onnxDir().absolutePath
                )
            }
            return
        }

        // Models to compile (DLC and not yet compiled)
        val toCompile = ModelType.coreTypes.filter { type ->
            val e = modelStore.getModel(type)
            e != null && e.format == ModelFormat.DLC && e.status != ModelStatus.COMPILED
        }
        val total = toCompile.size

        // Compile any DLC models that need it (but DON'T load contexts yet)
        if (total > 0) {
            compileStageStartTime = System.currentTimeMillis()
            onStageStart(COMPILE_STAGE_ID, "Model Compilation")
        }
        for ((index, type) in toCompile.withIndex()) {
            onProgress(
                COMPILE_STAGE_ID,
                index, total,
                0,
                MsgKey.k(MsgKey.DETAIL_COMPILING, modelName(type))
            )
            val compileResult = modelStore.compileModel(type)
            if (compileResult.isFailure) {
                throw IOException(
                    MsgKey.k(MsgKey.ERR_MODEL_COMPILE_FAILED, modelName(type), compileResult.exceptionOrNull()?.message ?: "")
                )
            }
            onProgress(
                COMPILE_STAGE_ID,
                index + 1, total,
                System.currentTimeMillis() - compileStageStartTime,
                MsgKey.k(MsgKey.DETAIL_COMPILED, modelName(type))
            )
        }
        if (total > 0) {
            onStageComplete(COMPILE_STAGE_ID, "Model Compilation", System.currentTimeMillis() - compileStageStartTime)
        }
    }

    // Load a single model context on-demand (just before its inference stage)
    private suspend fun ensureModelLoaded(type: ModelType) {
        synchronized(loadedModelsLock) { if (loadedModels.contains(type)) return }

        if (isCpuBackend) {
            val f = modelStore.onnxModelFile(type)
                ?: throw IllegalStateException("ONNX model for ${type.code} missing in ${modelStore.onnxDir()}")
            QnnJni.validateModelFile(f.absolutePath, "onnx")?.let {
                throw IOException("Invalid ONNX ${f.name}: $it")
            }
            val t0 = System.currentTimeMillis()
            android.util.Log.i(TAG, "Loading ONNX on-demand: ${type.code} <- ${f.name} (${f.length() / 1_000_000} MB)")
            if (!QnnJni.loadOnnxModel(type.code, f.absolutePath)) {
                throw IOException(MsgKey.k(MsgKey.ERR_MODEL_LOAD_FAILED, modelName(type)))
            }
            modelLoadMs[type] = System.currentTimeMillis() - t0
            synchronized(loadedModelsLock) { loadedModels.add(type) }
            return
        }
        val loadStart = System.currentTimeMillis()

        val entry: ModelEntry = modelStore.getModel(type)
            ?: throw IllegalStateException(MsgKey.k(MsgKey.ERR_MODEL_NOT_IMPORTED, modelName(type)))

        // Compile first if the DLC is not compiled yet (should already be done, but guard here)
        if (entry.status != ModelStatus.COMPILED) {
            val compileResult = modelStore.compileModel(type)
            if (compileResult.isFailure) {
                throw IOException(
                    MsgKey.k(MsgKey.ERR_MODEL_COMPILE_FAILED, modelName(type), compileResult.exceptionOrNull()?.message ?: "")
                )
            }
        }

        val ready = modelStore.getModel(type) ?: throw IllegalStateException(MsgKey.k(MsgKey.ERR_MODEL_MISSING, modelName(type)))
        val binPath = ready.runtimeBinPath
            ?: throw IllegalStateException(MsgKey.k(MsgKey.ERR_MODEL_NO_BIN, modelName(type)))

        android.util.Log.i(TAG, "Loading model on-demand: ${type.displayName} (${type.code}), binSize=${ready.fileSize}")
        val ok = QnnJni.loadContextBinary(type.code, binPath)
        if (!ok) throw IOException(MsgKey.k(MsgKey.ERR_MODEL_LOAD_FAILED, modelName(type)))
        modelLoadMs[type] = System.currentTimeMillis() - loadStart
        synchronized(loadedModelsLock) { loadedModels.add(type) }
    }

    // ====== Instrumentation: benchmark summary + tensor dumps ======

    /** One-line, grep-able benchmark summary (logcat tag [BENCH_TAG]) */
    private fun benchSummary(totalMs: Long, peakMb: Long): String {
        val b = _backend.value
        val stages = _state.value.stages.filter { it.id >= 0 && it.isComplete }
            .joinToString(" ") { "s${it.id}=${it.elapsedMs}" }
        val loads = modelLoadMs.entries.joinToString(" ") { "load_${it.key.code}=${it.value}" }
        return "backend=${b?.name ?: "?"} threads=${b?.threads ?: 0} precision=${if (b?.isCpu == true) "fp32" else "w8a16"} " +
            "total_ms=$totalMs peak_rss_mb=$peakMb $stages $loads device=${android.os.Build.MODEL} soc=${android.os.Build.SOC_MODEL}"
    }

    /**
     * Copy raw model inputs/outputs of this run to <externalFiles>/dumps/<timestamp>/
     * (pull with adb, compare with tools/compare_outputs.py). Skips the ~0.9 GB of
     * REST inter-segment tensors.
     */
    private fun dumpTensors(workDir: File, fpx: Float, origW: Int, origH: Int) {
        val base = context.getExternalFilesDir(null) ?: context.filesDir
        val ts = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US).format(java.util.Date())
        val out = File(base, "dumps/$ts").also { ensureDir(it) }
        val rel = mutableListOf(
            "image.raw", "x2.raw", "disparity_factor.raw",
            "x_latent0.raw", "x_latent1.raw", "x0_feat.raw", "x1_feat.raw", "x2_feat.raw", "x_lowres_feat.raw",
            "disparity.raw", "delta.raw", "output.ply"
        )
        for (p in DUMP_PATCHES) {
            rel += "patch_p%04d.raw".format(p)
            File(workDir, "out_pe/Result_$p").listFiles()?.forEach { rel += "out_pe/Result_$p/${it.name}" }
        }
        File(workDir, "out_ie/Result_0").listFiles()?.forEach { rel += "out_ie/Result_0/${it.name}" }
        var bytes = 0L
        for (r in rel) {
            val src = File(workDir, r)
            if (!src.isFile) continue
            val dst = File(out, r)
            dst.parentFile?.let { ensureDir(it) }
            src.copyTo(dst, overwrite = true)
            bytes += src.length()
        }
        val b = _backend.value
        val meta = org.json.JSONObject().apply {
            put("backend", b?.name ?: "")
            put("threads", b?.threads ?: 0)
            put("fallback_reason", b?.fallbackReason ?: org.json.JSONObject.NULL)
            put("fpx", fpx.toDouble()); put("orig_w", origW); put("orig_h", origH)
            put("device", android.os.Build.MODEL); put("soc", android.os.Build.SOC_MODEL)
            put("pe_patches", org.json.JSONArray(DUMP_PATCHES))
            put("stage_ms", org.json.JSONObject().apply {
                _state.value.stages.forEach { put(it.id.toString(), it.elapsedMs) }
            })
            put("model_load_ms", org.json.JSONObject().apply { modelLoadMs.forEach { (k, v) -> put(k.code, v) } })
            put("peak_rss_mb", QnnJni.getPeakRssKb() / 1024)
        }
        File(out, "meta.json").writeText(meta.toString(2))
        android.util.Log.i(BENCH_TAG, "tensor dump: ${out.absolutePath} (${bytes / 1_000_000} MB)")
    }

    // Backward compatibility: keep for standalone compilation from Models tab
    private suspend fun ensureReady() {
        ensureQnnInitialized()

        val toCompile = ModelType.coreTypes.filter { type ->
            val e = modelStore.getModel(type)
            e != null && e.format == ModelFormat.DLC && e.status != ModelStatus.COMPILED
        }
        val total = toCompile.size

        if (total > 0) {
            compileStageStartTime = System.currentTimeMillis()
            onStageStart(COMPILE_STAGE_ID, "Model Compilation")
        }
        var loadedCount = 0
        for (type in ModelType.coreTypes) {
            val compileIndex = toCompile.indexOf(type)
            val alreadyLoaded = synchronized(loadedModelsLock) { loadedModels.contains(type) }
            ensureModelReady(type, compileIndex, total)
            if (!alreadyLoaded) {
                loadedCount++
                onProgress(
                    PREPARE_STAGE_ID,
                    loadedCount, ModelType.coreTypes.size,
                    System.currentTimeMillis() - pipelineStartTime,
                    MsgKey.k(MsgKey.DETAIL_LOADING_MODEL, modelName(type))
                )
            }
        }
        if (total > 0) {
            onStageComplete(COMPILE_STAGE_ID, "Model Compilation", System.currentTimeMillis() - compileStageStartTime)
        }
    }

    private suspend fun ensureModelReady(type: ModelType, compileIndex: Int, compileTotal: Int) {
        synchronized(loadedModelsLock) { if (loadedModels.contains(type)) return }

        val entry: ModelEntry = modelStore.getModel(type)
            ?: throw IllegalStateException(MsgKey.k(MsgKey.ERR_MODEL_NOT_IMPORTED, modelName(type)))

        // Compile first if the DLC is not compiled yet
        if (entry.status != ModelStatus.COMPILED) {
            // Compilation start: progress shows x/total done (current model not counted yet)
            onProgress(
                COMPILE_STAGE_ID,
                compileIndex, compileTotal,
                0,
                MsgKey.k(MsgKey.DETAIL_COMPILING, modelName(type))
            )
            val compileResult = modelStore.compileModel(type)
            if (compileResult.isFailure) {
                throw IOException(
                    MsgKey.k(MsgKey.ERR_MODEL_COMPILE_FAILED, modelName(type), compileResult.exceptionOrNull()?.message ?: "")
                )
            }
            // Compilation done: update the count and elapsed time
            onProgress(
                COMPILE_STAGE_ID,
                compileIndex + 1, compileTotal,
                System.currentTimeMillis() - compileStageStartTime,
                MsgKey.k(MsgKey.DETAIL_COMPILED, modelName(type))
            )
        }

        // Re-read the entry (it may have been updated by compilation)
        val ready = modelStore.getModel(type) ?: throw IllegalStateException(MsgKey.k(MsgKey.ERR_MODEL_MISSING, modelName(type)))
        val binPath = ready.runtimeBinPath
            ?: throw IllegalStateException(MsgKey.k(MsgKey.ERR_MODEL_NO_BIN, modelName(type)))

        val ok = QnnJni.loadContextBinary(type.code, binPath)
        if (!ok) throw IOException(MsgKey.k(MsgKey.ERR_MODEL_LOAD_FAILED, modelName(type)))
        android.util.Log.i(TAG, "Model loaded: ${type.displayName} (${type.code})")
        synchronized(loadedModelsLock) { loadedModels.add(type) }
    }

    // ======  ======
    // ====== Internals: stage execution ======

    /** Run a single stage, emitting start / complete callbacks automatically */
    private fun <T> runStage(stageId: Int, block: () -> T): T {
        val stage = _state.value.stages.first { it.id == stageId }
        stageStartTime = System.currentTimeMillis()
        android.util.Log.i(TAG, "Stage ${stage.id} start: ${stage.name}")
        onStageStart(stageId, stage.name)
        try {
            val result = block()
            val elapsed = System.currentTimeMillis() - stageStartTime
            android.util.Log.i(TAG, "Stage ${stage.id} done: ${stage.name} (${elapsed}ms)")
            // Only mark the stage complete while the pipeline is still running
            if (_state.value.isRunning) {
                onStageComplete(stageId, stage.name, elapsed)
            }
            return result
        } catch (e: Exception) {
            // Mark the stage as failed on exception
            if (_state.value.isRunning) {
                fail(MsgKey.k(MsgKey.ERR_STAGE_EXCEPTION, stageId, e.message ?: e.toString()))
            }
            throw e
        }
    }

    private fun fail(message: String) {
        android.util.Log.e(TAG, "Pipeline failed: $message")
        // Release HTP memory of loaded models after failure to avoid leaks
        synchronized(loadedModelsLock) {
            for (modelType in loadedModels) {
                runCatching { QnnJni.freeModel(modelType.code) }
            }
            loadedModels.clear()
        }
        runCatching { QnnJni.clearRestCache() }
        // Reset every running stage so the state stays consistent after
        // cancellation or an exception
        val current = _state.value
        _state.value = current.copy(
            isRunning = false,
            errorMessage = message,
            stages = current.stages.map { s ->
                if (s.isRunning) s.copy(isRunning = false, detail = if (s.id == COMPILE_STAGE_ID) MsgKey.DETAIL_CANCELLED else s.detail)
                else s
            }
        )
    }

    // ====== ProgressCallback Implementation ======
    // ====== ProgressCallback implementation ======

    override fun onStageStart(stageId: Int, stageName: String) {
        updateStage(stageId) { it.copy(isRunning = true, isComplete = false, detail = "") }
    }

    override fun onProgress(stageId: Int, current: Int, total: Int, elapsedMs: Long, detail: String) {
        updateStage(stageId) {
            val effTotal = if (total > 0) total else it.total
            it.copy(
                current = current.coerceAtMost(effTotal),
                total = effTotal,
                elapsedMs = if (elapsedMs > 0) elapsedMs else it.elapsedMs,
                detail = detail
            )
        }
    }

    override fun onStageComplete(stageId: Int, stageName: String, elapsedMs: Long) {
        updateStage(stageId) {
            val total = if (it.total > 0) it.total else it.current
            it.copy(
                isRunning = false,
                isComplete = true,
                current = if (total > 0) total else it.current,
                total = total,
                elapsedMs = if (elapsedMs > 0) elapsedMs else it.elapsedMs,
                detail = ""
            )
        }
    }

    override fun onLog(level: Int, message: String) {
        // Log lines may be forwarded to Logcat or a UI log panel here (reserved)
        android.util.Log.println(logPriority(level), TAG, message)
    }

    override fun onError(stageId: Int, message: String) {
        fail(MsgKey.k(MsgKey.ERR_STAGE_ERROR, stageId, message))
    }

    // ======  ======
    // ====== Internals: state updates ======

    private fun updateStage(stageId: Int, transform: (StageState) -> StageState) {
        _state.update { current ->
            val idx = current.stages.indexOfFirst { it.id == stageId }
            if (idx < 0) return@update current
            val newStages = current.stages.toMutableList()
            newStages[idx] = transform(newStages[idx])
            current.copy(stages = newStages)
        }
    }

    private fun logPriority(level: Int): Int = when (level) {
        0 -> android.util.Log.DEBUG
        1 -> android.util.Log.INFO
        2 -> android.util.Log.WARN
        3 -> android.util.Log.ERROR
        else -> android.util.Log.INFO
    }

    companion object {
        private const val TAG = "PipelineManager"

        /** Logcat tag for backend selection decisions */
        const val BACKEND_TAG = "SHARP_BACKEND"

        /** Logcat tag for per-run benchmark summaries */
        const val BENCH_TAG = "SHARP_BENCH"

        /** PE patches whose outputs are dumped (first x0 patch, centre x1 patch, x2) */
        val DUMP_PATCHES = listOf(0, 29, 34)

        // Init stage (QNN runtime + model loading, runs first)
        const val PREPARE_STAGE_ID = -2

        // Model compilation stage (inserted at the head of stages; does not collide with 0-8)
        const val COMPILE_STAGE_ID = -1

        // Removed)
        const val PLY_OPTIMIZE_STAGE_ID = 9
    }
}