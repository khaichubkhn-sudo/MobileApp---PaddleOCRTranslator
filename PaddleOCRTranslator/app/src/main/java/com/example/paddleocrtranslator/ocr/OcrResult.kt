package com.example.paddleocrtranslator.ocr

import android.graphics.Rect

/** One detected text line: its bounding box in the *original* image, plus the recognized text. */
data class OcrLine(
    val box: Rect,
    val text: String,
    val confidence: Float
)

/** Full result of running the detector + recognizer over one image. */
data class OcrResult(
    val lines: List<OcrLine>,
    val detectMs: Long,
    val recognizeMs: Long
) {
    /** Lines joined top-to-bottom in reading order, one per line of output text. */
    fun joinedText(): String = lines.joinToString("\n") { it.text }
}

class OcrException(message: String, cause: Throwable? = null) : Exception(message, cause)
