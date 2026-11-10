# Porting notes: ML-Sharp-QNN → non-Snapdragon Android (target: Galaxy A57 / Exynos 1680)

Running log of findings, decisions, benchmarks and open questions.
Phase status: Phase 0 complete (merged). **Phase 1 in progress** on `phase1-cpu-fallback`: code complete, desktop-verified, waiting on the build toolchain (QNN SDK, NDK) for the on-device test.

---

## 0. Corrections to the starting assumptions

| Assumption | Reality (verified) |
|---|---|
| Project license is MIT | **GPL-3.0** (`LICENSE`, README badge). GaussSimplify (statically linked into `libsharp_qnn.so`) is also GPL-3.0, so the binary must stay GPL. GaussForge is Apache-2.0, stb_image is public domain, Apple SHARP is AML-R, the QNN SDK is proprietary. |
| "Smallest change = libQnnCpu.so + FP32 DLCs reused from the pipeline" | No FP32 DLCs or ONNX files are published. The HF repo `kjcpc/ML-Sharp-QNN` only has `dlc/w8a16/*.dlc` (pe 305.8 MB, ie 305.8 MB, rest_a 33.1 MB, rest_b 6.2 MB, rest_c 7.7 MB = 658.6 MB). **Any** non-HTP path needs locally regenerated models (see §6). |
| QNN CPU can maybe run the W8A16 DLCs | QNN docs: the CPU backend runs "quantized 8-bit and float 32-bit networks". There is no 16-bit activation support, so W8A16 is out. Qualcomm also calls it a *reference implementation*, scoped to "currently enabled Qualcomm chipsets". |
| AnyCalib is part of the pipeline | It isn't. It only runs when the user taps "AI suggest focal" (`HomeScreen.kt:228`). It's optional. |

---

## 1. Where the QNN backend is chosen and loaded

* **Library loading:** native only, via `dlopen` (never linked). `qnn_runtime.cpp:157` `createSharedState()` opens `<nativeLibraryDir>/libQnnSystem.so` and then `libQnnHtp.so`. It resolves `QnnSystemInterface_getProviders` / `QnnInterface_getProviders` and picks the first provider with matching API major and >= minor. The backend name `libQnnHtp.so` is hard-coded, so there's no backend abstraction.
* **Shared state:** one log + backend + device (`g_qnnShared`), ref-counted across 6 `HtpRuntime` objects (pe, ie, rest_a/b/c, anycalib). Each runtime owns a context and graph (`sharp_jni.cpp:45-55`).
* **Device config:** HTP-only custom configs (SOC=auto, ARCH from probe, unsigned PD), then DCVS v3 power config from user settings (`qnn_runtime.cpp:298-477`).
* **Model execution format:** the app downloads **DLC** files. On first run it compiles each one **on-device** into an HTP context binary with `QnnSystemDlc_createFromFile → composeGraphs → graphSetConfig(HTP O3, DLBC, SLC, weight packing) → graphFinalize → contextGetBinary` (`qnn_dlc_compiler.cpp`). It writes `<model>/bin/<type>_<hash>.bin`, which `loadFromBinary()` loads later. Compiling needs `libQnnHtpPrepare.so`.
* **Packaging (`app/build.gradle.kts`):**
  * `abiFilters = arm64-v8a`, `useLegacyPackaging = true`, `extractNativeLibs=true`.
  * Gradle task `copyQnnLibs` copies these from `$qnnSdkDir/lib/aarch64-android` into `app/src/main/jniLibs/arm64-v8a/` (git-ignored): `libQnnHtp`, `libQnnHtpPrepare`, `libQnnSystem`, `libQnnHtpNetRunExtensions`, `libQnnModelDlc`, `libQnnHtpV{68,69,73,75,79,81}Stub`, `libQnnHtpV*CalculatorStub`.
  * `copyQnnSkel` copies the Hexagon `libQnnHtpV*Skel.so` into `assets/qnn_skel/v*/` (git-ignored). `SkelExtractor` extracts the skel matching the probed arch at runtime, and `ADSP_LIBRARY_PATH` is pointed at it.
  * Building **requires** the QNN SDK: `build.gradle.kts` hard-fails without `qnn.sdk.dir`, and CMake includes `${QNN_SDK_DIR}/include/QNN`.

## 2. Every place a non-Snapdragon device is rejected or would fail

| # | Location | What happens on the A57 | Severity |
|---|---|---|---|
| G1 | `AndroidManifest.xml:36` `<uses-native-library android:name="libcdsprpc.so" android:required="true"/>` | PackageManager **refuses to install** the APK on a device without Qualcomm's FastRPC lib (expected `INSTALL_FAILED_MISSING_SHARED_LIBRARY`; confirm on device). | **Blocker: never installs** |
| G2 | `PipelineManager.ensureQnnInitialized()` (`PipelineManager.kt:308-311`) | `probeHtpArch()` returns `""` and the app throws `ERR_QNN_INIT`. Every path goes through this: pipeline run, "compile", "compile all", AnyCalib. | Blocker |
| G3 | `probeHtpArch` (`sharp_jni.cpp:251`) | dlopens `libQnnHtp.so` and calls `deviceGetPlatformInfo`. Expected to return an error without FastRPC. **Not verified that it can't crash** on Exynos. Needs a device log. | Risk |
| G4 | `SkelExtractor.extractSkel` / `nativeInit` (`sharp_jni.cpp:440`) | No arch, so no skel and HTP `deviceCreate` fails. `nativeInit` returns false if **any** runtime fails, which takes down the whole native layer. | Blocker (all-or-nothing init) |
| G5 | `ModelStore.compileModel` / `PipelineManager.ensureReadyForCompile` | Downloaded models are DLCs that must be compiled to an HTP `.bin`, and that only works with HTP. | Blocker for the model flow |
| G6 | `validateModelFile` (`sharp_jni.cpp:1379`) | Rejects files > 1 GB. FP32 pe/ie are about 1.22 GB each, and the AnyCalib FP32 DLC is 1.28 GB. Only `bin`/`dlc` formats are known. | Blocker for FP32 models |
| G7 | `runRestInference` (`sharp_jni.cpp:1040-1050`) calls `execute(..., keepOutputQuantized=true)` with `outputs[j].data = nullptr` | For **FP32** output tensors, `HtpRuntime::execute` binds `clientBuf.data = outputs[i].data` (null) and `getQuantizedOutputData()` returns null. On FP32 REST models this fails or crashes. **This is where the W8A16 assumption leaks.** | Bug for any FP32 path |
| G8 | README/UI text "Snapdragon 8 Gen 2+" | Cosmetic. | Low |

No `Build.SOC_MODEL` / `Build.HARDWARE` checks exist anywhere in Kotlin. The gates are G1, G2 and G4.

## 3. How tensors cross the QNN boundary (dtype, layout, quant)

* **Tensor metadata** is read from the context binary (`QnnSystemContext_getBinaryInfo`): name, dims, dtype, and SCALE_OFFSET / BW_SCALE_OFFSET params (`qnn_runtime.cpp:684-751`). Only `graphs[0]` is used.
* **Inputs:** float32 is quantized on the fly to UFIXED_16/8 (`round(x/scale) - offset`), or converted to FP16 if the tensor is FP16. FP32 is passed through **without a copy**. REST inputs can be pre-quantized while streaming from file (`Tensor.quantized=true`).
* **Outputs:** dequantized back to float32 (`(q + offset) * scale`). REST uses `keepOutputQuantized` and dequantizes chunk-by-chunk while streaming to disk, to halve peak RSS (see G7).
* **Layout:** NCHW float32 end-to-end, matching the ONNX layout (the converter preserved it). No transposes anywhere in the pipeline. `nchwToNhwc` exists but is unused.
* **C pre/post (`core/*.c`) is fully backend-agnostic.** It only reads and writes float32 NCHW `.raw` files. The W8A16 assumption is confined to `qnn_runtime.cpp` and `sharp_jni.cpp::runRestInference`/`restCachePut`.
* **Inter-stage transport is files** under `cacheDir/sharp_work/` (plus a small in-RAM cache for REST edge tensors < 2–16 MB). For example, PE writes 35 × 3 × 2.36 MB ≈ 248 MB of raw files. This is a Phase 3 target.

## 4. Inference pipeline (order, shapes, model vs. C)

| # | Stage | Kind | Inputs → Outputs (all float32 NCHW) |
|---|---|---|---|
| 0 | `prepImage` (`sharp_prep_image`) | C | JPEG/PNG → EXIF orientation + focal → bilinear (align_corners) resize → `image.raw` **[1,3,1536,1536]** in [0,1]; returns `fpx, dfactor, origW, origH` |
| 1 | `runPre` (`sharp_pre`) | C | normalize to [-1,1]; pyramid x0 1536², x1 768², x2 384²; split into **35 patches [1,3,384,384]** (25 from x0: 5×5, overlap 0.25; 9 from x1: 3×3, overlap 0.5; 1 = x2). `x2.raw` = patch 34 |
| 2 | **PE** `patch_encoder` (DINOv2-L-class ViT, ~306 M params) × **35 runs** | model | `patch_1` [1,3,384,384] → `patch_features`, `latent0`, `latent1` each **[1,1024,24,24]** |
| 3 | **IE** `image_encoder` (same size) × 1 run | model | `image_1` [1,3,384,384] → `image_features` **[1,1024,24,24]** |
| 4 | `runMerge` (`sharp_merge`) | C | `x0_feat` [1,1024,96,96] (25 patches), `x1_feat` [1,1024,48,48] (9), `x2_feat` [1,1024,24,24], `x_latent0`/`x_latent1` [1,1024,96,96], `x_lowres_feat` [1,1024,24,24] |
| 5 | **REST_A** (ONNX nodes 0–51: upsample + fuse + early decoder) | model | 6 merged features → edge tensors (largest ≈ 37.7 MB; total REST_A output ≈ 1 GB as float, per code comment) |
| 6 | **REST_B** (nodes 52–83: decoder tail + monodepth head) | model | REST_A edges → `disparity` **[1,2,1536,1536]** (18.9 MB) + edges |
| 7 | **REST_C** (nodes 84–end: init_model, feature_model, prediction head) | model | `image` [1,3,1536,1536], `disparity_factor` [1] (= fpx/origW) + edges → `delta` **[1,14,2,768,768]** |
| 8 | `runPost` (`sharp_post`) | C | depth_from_disparity → initializer (2 layers, stride 2) → composer → unproject → `save_ply`. **2 × 768 × 768 = 1,179,648 Gaussians** before optimization |
| 9 | `nativeOptimizePly` (optional) | C++ / OpenMP | GaussSimplify: opacity prune + SOR + kNN merge (in place) |
| — | AnyCalib (on demand only) | model | Kotlin centre-crop → [1,3,322,322] [0,1] → `rays` [1,3,322,322] → C++ median focal fit |

Models load on demand right before their stage and are freed right after (`PipelineManager.kt:165-228`). Only one model is resident at a time.

Exact REST edge-tensor names and shapes aren't in the repo; they depend on the exported ONNX. **Open question Q4.**

## 5. `build_rest_pipeline.py`

* **Inputs it needs:** the `apple/ml-sharp` source tree (the script must sit in that repo root; it imports `sharp.*` from `src/`), the checkpoint `sharp_2572gikvuh.pt`, torch, onnx, onnxsim, onnxruntime, and calibration images in `data/`.
* **ONNX export:** `-t onnx` writes `output/onnx/{patch_encoder,image_encoder,rest,rest_seg_a,rest_seg_b,rest_seg_c}.onnx` (opset 17 for pe/ie, 13 for rest), each simplified with onnxsim. **This step does not need the QNN SDK.**
  * Export hacks made for QNN: the prediction-head 5D concat is rewritten as a 4D concat + reshape, and `aten::unflatten` is registered as a Reshape symbolic.
* **Why `rest` is split:** the README only says "for DSP memory"; nothing else documents it. Cuts are **hard-coded topological node indices** `CUTS=[52,84]` on the simplified graph, so they're fragile if the export changes. The full-resolution (1536²) decoder and Gaussian head produce GB-scale activations, so this is almost certainly an HTP context/VTCM/DDR-mapping limit. On CPU/GPU the unsplit `rest.onnx` can run directly (Phase 3: measure it).
* **Calibration:** 20 images by default. PE gets ≤70 samples spread over the 35 patch slots, IE gets 1 per image. REST calibration is produced by running the split ONNX segments in onnxruntime (~600 MB per image). It's only needed for quantization.
* **FP32 DLCs** already land in `output/dlc/fp32/` and are never deleted. The only obstacle to an FP32-only run is that `_convert_one` insists on calibration data and always runs the quantizer. That's a small change if we keep QNN CPU.

## 6. Model-weights licensing (flag)

Running the export locally from Apple's checkpoint for personal use on your own phone is fine. **Hosting** the resulting ONNX/FP32 files (on HF or anywhere else) would be redistributing Apple's weights in a new form under AML-R. I won't do that; it's your call. The existing W8A16 DLCs on HF were published by the upstream author, not by us.

## 7. Build environment (this Mac)

| Item | Status |
|---|---|
| JDK 17 | ✅ `/opt/homebrew/opt/openjdk@17` (not on PATH; set `JAVA_HOME`) |
| Android SDK platform 35, build-tools 35.0.0 | ✅ `/opt/homebrew/share/android-commandlinetools` |
| NDK 29.0.14206865 | ❌ missing (`sdkmanager "ndk;29.0.14206865"`) |
| CMake 3.22.1 | ❌ missing in the SDK (`sdkmanager "cmake;3.22.1"`) |
| Gradle 9.6.1 (wrapper) | ❌ not cached; downloads on first run. **Risk:** AGP 8.7.3 is not documented as compatible with Gradle 9.x. Untested. |
| `local.properties` | ❌ missing (needs `sdk.dir`, `qnn.sdk.dir`) |
| **QNN SDK 2.48.0** | ❌ missing. Required even to compile, because of the headers. Download is gated behind Qualcomm's license, so you need to fetch it. |
| Python for the export | system 3.9.6, no torch/onnx. A venv plus the `apple/ml-sharp` checkout and checkpoint are needed. |
| adb | ✅; device `R5CRC3TBKQN` is attached. I haven't touched it. |

**I did not attempt a build.** It can't get past Gradle configuration without the QNN SDK, and the NDK/CMake installs (~1.5 GB) would change your system.

## 8. Phase 1 proposal: ORT CPU instead of QNN CPU

The goal is still the smallest change that gets a correct `.ply` on the A57. I recommend **skipping QNN CPU and going straight to ONNX Runtime (CPU EP, XNNPACK optional)**:

1. Both options need regenerated models anyway (no FP32 artifacts exist). The ONNX files are the *first* output of the same script and don't need the QNN SDK. FP32 DLCs need ONNX first **plus** the QNN converter.
2. QNN CPU is documented as a reference implementation scoped to Qualcomm chipsets. Whether it even runs on Exynos is unverified, and it's likely slow on ~7 TFLOP of ViT-L work (36 encoder runs).
3. ORT's CPU path is the Phase 2/3 fallback anyway. QNN CPU would be throwaway work.
4. `libonnxruntime.so` is MIT-licensed and ships as a Maven AAR (`com.microsoft.onnxruntime:onnxruntime-android`) with C/C++ headers via prefab. There's no proprietary-redistribution question, unlike `libQnnCpu.so` (whose redistribution terms would need checking in the SDK license).

The cost is a small `OrtRuntime` class (~200 lines). It exposes the same `getInputInfos / getOutputInfos / execute(Tensor…)` surface as `HtpRuntime`, so `sharp_jni.cpp` barely changes.

**Concrete Phase 1 steps (branch `phase1-cpu-fallback`):**
1. Manifest: `libcdsprpc.so` `required="false"` (fixes G1; Snapdragon still gets the library mapped).
2. Capability probe: in `PipelineManager`, try HTP (probe arch → extract skel → `nativeInit`). On any failure, log the reason under one tag (`SHARP_BACKEND`) and select CPU. Make native init per-backend, not all-or-nothing (G2, G4).
3. Native: add `ort/ort_runtime.{h,cpp}` and a minimal `IRuntime` base that both `HtpRuntime` and `OrtRuntime` implement. Link ORT via prefab. Fix G7 (handle FP32 outputs when `keepOutputQuantized`). Use FP32 pass-through; ORT session options: memory pattern on, `intra_op_num_threads` = big cores, external-data/mmap where possible.
4. Models: add `ModelFormat.ONNX` and an `onnx/` directory with no compile step, and raise or format-scope the 1 GB validation limit (G6). Keep the 3 REST segments in Phase 1 so tensor names and JNI routing are reused. The unsplit `rest.onnx` is a Phase 3 experiment. You side-load the ONNX via `adb push`; there's no download URL (see §6).
5. AnyCalib on CPU: hidden or disabled with a "needs NPU" message in Phase 1, unless you also want an AnyCalib ONNX export (Q3).
6. Instrumentation: backend chip in the UI ("NPU (HTP)" / "CPU (ORT)"), per-stage ms, peak RSS from `VmHWM`. Add a debug switch that dumps raw model outputs to `files/dumps/`, plus `tools/compare_outputs.py` (desktop onnxruntime on a fixed image vs. the pulled dumps: cosine, max/mean abs error, Gaussian count).
7. Expected (to be measured): very slow (minutes) and memory-heavy in REST (several GB peak at 1536²). That's acceptable for Phase 1 on an 8–12 GB device.

## 8b. Test devices

Decision (user): develop and experiment on the **A52s first**, then move to the A57. The same build must work on both and run better on the A57.

| | Galaxy A52s 5G (dev device, connected) | Galaxy A57 (final target) |
|---|---|---|
| Model / build | SM-A528B, Android 14 (API 34), One UI build `A528BXXSBGYI3` | Android 16 / One UI 8.5 |
| SoC | **Snapdragon 778G** (SM7325 "Yupik", board `lahaina`) | Exynos 1680 |
| CPU | 4× Cortex-A78 (0xd41) @ 2.4 GHz + 4× Cortex-A55 (0xd05) @ 1.8 GHz. Features: `asimddp` (dotprod), `fphp/asimdhp` (FP16 arith). **No i8mm, no bf16.** | TBD (probe on device) |
| GPU | Adreno 642L. `/vendor/lib64/libOpenCL.so` ✅, `vulkan.adreno.so` ✅ | Xclipse 550 (Vulkan ✅, OpenCL TBD) |
| NPU | Hexagon 770, **HTP v68**. `libcdsprpc.so` present. The app already ships V68 stub/skel. README claims SD 8 Gen 2+ only, so V68 is untested by upstream. | none usable |
| RAM | **5.5 GB total, ~1.4 GB `MemAvailable`** at probe time | 8 / 12 GB |
| Free storage | 13 GB | TBD |

What this means:
* **The A52s is a Snapdragon device.** It doesn't reproduce the Exynos failures G1 (manifest) or G3 (HTP probe on a non-Qualcomm SoC). Those can only be confirmed on the A57. To exercise the CPU fallback on the A52s, Phase 1 needs a **"force backend" developer override** in Settings now rather than in Phase 2.
* **The A52s can give us an HTP v68 baseline:** an on-device reference output and timing, if the current W8A16 path works on v68. HTP v68 support for 16-bit activations has to be verified.
* **Memory is the binding constraint on the A52s.** An FP32 encoder alone is ~1.22 GB of weights, and REST works at 1536² with GB-scale activations. Expect a low-memory kill with plain FP32 on 5.5 GB. Plan per device class:
  * ≥ 8 GB (A57): FP32 (or FP16-stored) weights.
  * < 8 GB (A52s): int8 weight-only / dynamic-quant PE and IE (~310 MB each, uses dotprod), plus REST possibly split finer. Every model set goes through `compare_outputs.py`.
* Defaults are chosen by RAM and CPU features at runtime, not by device name.

## 8c. Phase 1 log

**Decisions (user, 2026-11-10):** ORT CPU instead of QNN CPU. ONNX export runs on this Mac in conda env `sharp-export`. Device experiments run on the A52s first.

**Model export (done, local only, not hosted):**
* Env: conda `sharp-export` (Python 3.12, torch 2.8.0, onnx 1.23.2, onnxruntime 1.31.1, onnxsim). `apple/ml-sharp` is cloned at `~/Codebase/ml-sharp` (commit `aed6527`), with checkpoint `sharp_2572gikvuh.pt` (2.81 GB) from `ml-site.cdn-apple.com`. Use is under `LICENSE_MODEL` (research use).
* Command: `python build_rest_pipeline.py -t onnx -a all --sharp_root ~/Codebase/ml-sharp -o output`. The new `--sharp_root` option means the script no longer has to live inside the ml-sharp checkout.
* Output (`output/onnx/`, git-ignored): patch_encoder 1215 MB, image_encoder 1215 MB, rest 186 MB, rest_seg_a 126 MB, rest_seg_b 29 MB, rest_seg_c 32 MB.
* **Split finding:** with torch 2.8 the hard-coded `CUTS=[52,84]` put `disparity` in seg C, so the app's REST_B contract broke. Cuts are now automatic. Seg B ends right after the node producing `disparity` (node 91 of 249), and A/B is split where the fewest bytes cross. Result: **[24, 92]**, with 878 MB crossing A→B and 897 MB crossing B→C (the old second cut carried 2086 MB). `--cuts a,b` overrides.
* **Activation sizes (REST, FP32):** the largest single tensor is 1208 MB (`/md_head/md_head.1/ConvTranspose`, 1536²). Many are 604 MB (768²×256). Expect a REST peak of several GB on the CPU path. This decides whether the A52s (5.5 GB) can run FP32.
* `tools/verify_onnx.py`: **PASS.** Split chain vs. monolithic `rest.onnx` is bit-identical. PE/IE ONNX vs. PyTorch: cos 1.0, max |d| ≤ 1.6e-5.

**Code (commit `e8a2e6a`):** see the commit message. In short:
* Manifest `required=false` (G1).
* `sharp::ModelRuntime` interface with `HtpRuntime` (unchanged behaviour) and `OrtRuntime`.
* Capability probe with logged fallback (G2, G4).
* Fixes: G7 null FP32 output buffer; uninitialised `Tensor::quantized`; disparity taken from either REST segment.
* ONNX format plus 2 GB validation for ONNX (G6). DLC compile and AnyCalib say "needs NPU" on the CPU backend (G5).
* UI backend chip plus peak RSS; Settings > Inference backend (Auto/HTP/CPU, threads, tensor dump).
* Log tags `SHARP_BACKEND`, `SHARP_BENCH`, `SHARP_ORT`.

**Build status (after cleanup commit `3db06a3`):**
* NDK 29.0.14206865 and CMake 3.22.1 are installed in the Homebrew SDK; `local.properties` has `sdk.dir` (git-ignored).
* Gradle 9.6.1 with AGP 8.7.3 configures fine.
* `:app:compileDebugKotlin` passes. It's run with a placeholder `-Pqnn.sdk.dir`, since Kotlin compilation doesn't touch CMake.
* `:app:extractOrt` unpacks the ORT headers and `libonnxruntime.so`.
* `ort_runtime.cpp` passes a syntax check against the ORT 1.31.1 headers.
* **QNN SDK:** the installed version is **2.52.0.261104** (QNN API v2.41) at `/Users/stefano/SDKs/qairt/2.52.0.261104`, not the 2.48 upstream used. All libraries and v68–v81 skels that `copyQnnLibs`/`copyQnnSkel` expect are present; the SDK also ships hexagon-v85, which the app doesn't package yet. The upstream W8A16 DLCs were converted with 2.48 and are now compiled on-device by the 2.52 runtime. This is a change to the HTP stack to watch during the HTP baseline. If on-device DLC compilation fails, try 2.48 first.
* **`./gradlew :app:assembleDebug` passes, including the native code**, on the first full build. The APK is 152 MB: `libQnnHtpPrepare.so` 89.6 MB, `libonnxruntime.so` 34.2 MB, six skels at ~13.5 MB each in assets, `libsharp_qnn.so` 0.5 MB.
* Cleanup also fixed: FP32 REST outputs are written without an extra copy, the ORT arena is enabled only for PE, the backend kind is an enum, and changing backend settings re-probes at the next run instead of tearing down on the UI thread.

**Correctness tooling:**
* `tools/host_post/` builds the app's own C core (prep/pre/merge/post) for the desktop, so the desktop reference uses the exact same C pre/post-processing.
* `tools/compare_outputs.py --dump <pulled dump>` runs per-model checks (device inputs → desktop FP32 model) and an end-to-end check (device `image.raw` → full desktop pipeline → `.ply`). It reports cosine, max/mean |d| and Gaussian count.
* `--reference_image <jpg> --out <dir>` makes a desktop-only reference.
* Self-check on the reference: 51/51 identical.
* Reference image: `~/Codebase/ml-sharp/data/teaser.jpg` (Apple's, not committed). fpx = 1246.67, 1635×748, **1,179,648 Gaussians**.

## 8d. A52s run 1 (2026-11-10): first on-device result

Setup: debug APK (`0696772` + Home gating fix), QNN SDK 2.52 libs, upstream W8A16 DLCs and the local FP32 ONNX set side-loaded with `adb push`, input `teaser.jpg`, backend override Auto, tensor dump on.

1. **The capability probe and fallback work on real hardware.** On the SD778G (v68), `probeHtpArch` found no HTP device. FastRPC initialised on the cDSP (domain 3), but `deviceGetPlatformInfo` reported nothing. The app logged `HTP unavailable (probeHtpArch: no HTP device reported) -> CPU fallback (ONNX Runtime)` and continued on ORT with 4 threads. There was no crash. This is the same path the A57 will take.
2. **New gate found and fixed:** Home's Run button required the five *HTP* slots to be compiled. With only the ONNX set present (the A57 case) the app could never start a run. `canRun` now also accepts a complete ONNX set.
3. **PE on CPU:** 35 patches in 352.7 s. Early patches took ~10 s each and later ones ~5 s (probably memory pressure and swapping from other apps settling). RSS ~1.34 GB.
4. **REST_B does not fit in FP32 on a 5.5 GB phone.** lmkd killed the foreground app ~20 s into seg B ("min2x watermark is breached even after kill", 2.04 GB RSS + 590 MB swap), after first killing most background apps. On desktop, seg B peaks at **~4.6 GB** (arena on or off). The cause is the monodepth head at full resolution:
   `md_head.0` Conv → [1,128,768,768] (302 MB), `md_head.1` ConvTranspose → **[1,128,1536,1536] (1208 MB)**, `md_head.2` Conv → [1,32,1536,1536] (302 MB), then 1×1 Conv → disparity [1,2,1536,1536]. On top of that come seg B's 878 MB of inputs and the decoder's 768²×256 activations (604 MB each).
   * The head is purely local (3×3 conv, 2×2 stride-2 transposed conv, 3×3 conv, 1×1 conv), so it can be **run in spatial tiles with a small halo and give exact results**. The same holds for the full-resolution parts of seg C.
   * For the A57 (8/12 GB), FP32 at ~4.6 GB peak probably fits; 8 GB is borderline.
5. **HTP on the A52s:** to investigate. Is `deviceGetPlatformInfo` failing because of QNN 2.52 on v68, unsigned-PD support on SM7325, or missing env/skel at probe time? Upstream never claimed v68 support.

## 9. Open questions

* **Q1** Approve ORT-CPU instead of QNN-CPU for Phase 1?
* **Q2** QNN SDK 2.48: please download and give me the path. Also OK to install NDK 29 + CMake 3.22.1 via `sdkmanager`?
* **Q3** Models: should I set up a local Python env here and run the ONNX export (`-t onnx`)? That needs the `apple/ml-sharp` checkout, `sharp_2572gikvuh.pt` from Apple, and ~10 GB disk. Or will you run it? AnyCalib ONNX too?
* **Q4** REST edge-tensor names/shapes: they'll be recorded here after the export.
* **Q5** Device facts: SoC string, CPU core layout, OpenCL presence on Xclipse 550, behaviour of `probeHtpArch` on Exynos. See the adb probe commands in the Phase 0 summary.

## 10. Benchmark table

### Galaxy A57 (target)
_No measurements yet._

### Other devices
| Device | Backend | Precision | Threads | pre | PE ×35 | IE | merge | REST a+b+c | post | Total | Peak RSS | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Apple M4 Mac (desktop reference, onnxruntime 1.31.1 CPU, Python) | ORT CPU EP | FP32 | default | 0.04 s | 19.5 s | 1.8 s | 0.14 s | 15.1 s | 0.74 s | ~38 s | — | reference |
| Galaxy A52s (SD778G) | HTP v68 | W8A16 | — | n/a: HTP not detected (see A52s run 1) | | | | | | | | |
| Galaxy A52s (SD778G), run 1 2026-11-10 | ORT 1.31.1 CPU EP (auto fallback) | FP32 | 4 | 0.68 s (+0.14 s decode) | **352.7 s** (load 7 s; 5–11 s/patch) | 4.6 s (load 5.8 s) | 2.4 s | A: 12.5 s (run 3.5 s, rest is writing 0.9 GB of edges); **B: killed by lmkd** | — | — | 2.04 GB RSS + 590 MB swap at kill | — |
