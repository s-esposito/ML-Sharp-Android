package com.sharp.qnn.pipeline

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.security.MessageDigest

/**
 * Checkpoint of an inference run, kept next to its intermediate files in the work dir
 * (`checkpoint.json`), so a run that crashed or was killed (e.g. by lmkd) can resume
 * from the last completed stage instead of starting over.
 *
 * A checkpoint is only resumed when its [key] matches the new run (same input image
 * bytes, focal override, backend and model files) and the previous run did not finish.
 * Every write goes to a temp file first and is renamed, so a crash mid-write never
 * leaves a corrupt checkpoint.
 */
class RunCheckpoint(private val workDir: File) {

    private val file = File(workDir, "checkpoint.json")
    private var json = JSONObject()

    /** Previous run in this work dir had the same [key] and did not finish */
    fun canResume(key: String): Boolean {
        val prev = runCatching { JSONObject(file.readText()) }.getOrNull() ?: return false
        if (prev.optBoolean("finished", false)) return false
        val prevKey = prev.optString("key")
        if (prevKey != key) {
            val a = prevKey.split('|'); val b = key.split('|')
            val diff = (a.indices.filter { it >= b.size || a[it] != b[it] }.map { "${a[it]} -> ${b.getOrNull(it)}" })
            android.util.Log.i("RunCheckpoint", "not resuming, key changed: $diff")
            return false
        }
        json = prev
        return true
    }

    /** Start a fresh checkpoint for [key] (the caller wiped the work dir) */
    fun start(key: String) {
        json = JSONObject().put("version", VERSION).put("key", key)
            .put("completed", JSONArray()).put("finished", false)
        save()
    }

    fun isDone(stageId: Int): Boolean = completed().contains(stageId)

    fun markDone(stageId: Int) {
        if (!isDone(stageId)) json.getJSONArray("completed").put(stageId)
        save()
    }

    fun completed(): Set<Int> {
        val a = json.optJSONArray("completed") ?: return emptySet()
        return (0 until a.length()).map { a.getInt(it) }.toSet()
    }

    /** prepImage output [fpx, dfactor, origW, origH], needed when stage 0 is skipped */
    fun putPrepMeta(meta: FloatArray) {
        json.put("prep_meta", JSONArray(meta.map { it.toDouble() }))
        save()
    }

    fun prepMeta(): FloatArray? {
        val a = json.optJSONArray("prep_meta") ?: return null
        return FloatArray(a.length()) { a.getDouble(it).toFloat() }
    }

    fun markFinished() {
        json.put("finished", true)
        save()
    }

    private fun save() {
        val tmp = File(workDir, "checkpoint.json.tmp")
        tmp.writeText(json.toString())
        if (!tmp.renameTo(file)) {
            file.delete()
            tmp.renameTo(file)
        }
    }

    companion object {
        /** Bump when the work-dir layout or stage numbering changes */
        private const val VERSION = 1

        /**
         * Run identity: input image content, focal override, backend and model files
         * (path, size, mtime). Intermediates are only reused when all of them match.
         */
        fun key(input: InputStream, overrideFpx: Float?, backend: String, modelFiles: List<File>): String {
            val inMd = MessageDigest.getInstance("SHA-1")
            input.use { s ->
                val buf = ByteArray(1 shl 16)
                while (true) {
                    val n = s.read(buf)
                    if (n < 0) break
                    inMd.update(buf, 0, n)
                }
            }
            val modelMd = MessageDigest.getInstance("SHA-1")
            for (f in modelFiles.sortedBy { it.absolutePath }) {
                modelMd.update("${f.absolutePath}:${f.length()}:${f.lastModified()}|".toByteArray())
            }
            // Readable parts so a mismatch can be logged component by component
            return "v$VERSION|input=${hex(inMd.digest())}|fpx=$overrideFpx|backend=$backend|models=${hex(modelMd.digest())}"
        }

        private fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }.take(16)
    }
}
