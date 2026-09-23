package com.example.paddleocrtranslator.ocr

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Loads the PP-OCR character dictionary once and greedily decodes CTC output
 * (argmax per timestep, collapse repeats, drop blanks) into text.
 *
 * Dictionary file expected at assets/labels/ppocr_keys_v1.txt (or whichever dict matches
 * the recognition model you bundle) — one character per line, index 0 reserved for CTC blank.
 * See README.md for where to get it.
 */
class CtcLabelDecoder(context: Context, assetPath: String = "labels/ppocr_keys_v1.txt") {

    private val labels: Array<String>

    init {
        val loaded = ArrayList<String>()
        loaded.add("blank") // index 0 = CTC blank placeholder
        try {
            context.assets.open(assetPath).use { stream ->
                BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { reader ->
                    var line: String?
                    while (true) {
                        line = reader.readLine() ?: break
                        loaded.add(line!!)
                    }
                }
            }
            loaded.add(" ") // space char, PP-OCR dicts are usually trained with use_space_char=True
        } catch (t: Throwable) {
            throw OcrException(
                "Missing or unreadable dictionary at assets/$assetPath. " +
                    "Download it from the PaddleOCR repo (see README.md) and place it there.",
                t
            )
        }
        labels = loaded.toTypedArray()
    }

    /**
     * @param logits shape [timesteps, numClasses], row-major, already the raw model output
     *   (softmax not required — argmax is invariant to it)
     */
    fun decode(logits: FloatArray, timesteps: Int, numClasses: Int): Pair<String, Float> {
        val sb = StringBuilder()
        var lastIdx = -1
        var confSum = 0f
        var confCount = 0

        for (t in 0 until timesteps) {
            var bestIdx = 0
            var bestVal = Float.NEGATIVE_INFINITY
            val base = t * numClasses
            for (c in 0 until numClasses) {
                val v = logits[base + c]
                if (v > bestVal) {
                    bestVal = v
                    bestIdx = c
                }
            }
            if (bestIdx != 0 && bestIdx != lastIdx) { // 0 = blank; collapse repeats
                if (bestIdx < labels.size) {
                    sb.append(labels[bestIdx])
                    confSum += bestVal
                    confCount++
                }
            }
            lastIdx = bestIdx
        }

        val avgConf = if (confCount > 0) confSum / confCount else 0f
        return sb.toString() to avgConf
    }
}
