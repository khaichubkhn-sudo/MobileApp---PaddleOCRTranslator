package com.example.paddleocrtranslator.util

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.Handler
import android.os.Looper
import android.widget.TextView
import java.text.SimpleDateFormat
import java.util.ArrayDeque
import java.util.Locale

/**
 * Small, bounded-size debug logger that renders into a TextView at the bottom of the
 * screen. Deliberately keeps only the last [MAX_LINES] lines so the log itself never
 * becomes a source of unbounded memory growth across many OCR runs.
 */
class DebugLogger(private val targetView: TextView) {

    companion object {
        private const val MAX_LINES = 200
        private val TIME_FORMAT = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)
    }

    private val lines = ArrayDeque<String>(MAX_LINES)
    private val mainHandler = Handler(Looper.getMainLooper())

    @Synchronized
    fun log(message: String) {
        val stamped = "${TIME_FORMAT.format(System.currentTimeMillis())}  $message"
        if (lines.size >= MAX_LINES) {
            lines.removeFirst()
        }
        lines.addLast(stamped)
        render()
    }

    fun logMemory(context: Context, tag: String) {
        val rt = Runtime.getRuntime()
        val usedMb = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024)
        val maxMb = rt.maxMemory() / (1024 * 1024)

        val nativeMb = Debug.getNativeHeapAllocatedSize() / (1024 * 1024)

        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am?.getMemoryInfo(memInfo)

        log("[$tag] javaHeap=${usedMb}MB/${maxMb}MB nativeHeap=${nativeMb}MB")
    }

    fun logError(message: String, t: Throwable?) {
        log("ERROR: $message${if (t != null) " -> ${t.javaClass.simpleName}: ${t.message}" else ""}")
    }

    @Synchronized
    private fun render() {
        val text = lines.joinToString("\n")
        mainHandler.post {
            targetView.text = text
        }
    }
}
