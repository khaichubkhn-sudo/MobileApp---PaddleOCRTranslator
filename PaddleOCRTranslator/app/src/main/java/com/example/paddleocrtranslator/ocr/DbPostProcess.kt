package com.example.paddleocrtranslator.ocr

import android.graphics.Rect

/**
 * Turns the DB detection model's probability map into text-line bounding boxes.
 *
 * NOTE ON SCOPE: the official PaddleOCR demo extracts *rotated polygons* from the map
 * (threshold -> contours -> min-area-rect -> "unclip" expansion via Clipper), which needs
 * either OpenCV or a bundled C++ clipper lib. To keep this project pure-Kotlin and buildable
 * without extra native dependencies, this class instead does iterative connected-component
 * labeling and returns axis-aligned bounding boxes. That's a real simplification: it works
 * well for horizontal-ish text (the common case for photographed documents/screens/signage),
 * but won't tightly fit heavily rotated or curved text the way the official polygon pipeline
 * does. If you need that, port `deploy/lite/db_post_process.cc` + `clipper.cpp` from the
 * PaddleOCR repo instead of this class.
 */
class DbPostProcess {

    companion object {
        const val PROB_THRESHOLD = 0.3f
        const val MIN_BOX_AREA_PX = 20 // in probability-map pixel units
        const val UNCLIP_RATIO = 1.5f  // expand boxes outward a bit, DB output tends to be tight
    }

    // Reused across calls, sized lazily to the current map and kept for next call if big enough.
    private var visited: BooleanArray = BooleanArray(0)
    private var stackX: IntArray = IntArray(0)
    private var stackY: IntArray = IntArray(0)

    /**
     * @param prob probability map, row-major, size w*h, values in [0,1]
     * @param w map width, h map height (these are the *detection input* dims, not the original image)
     */
    fun extractBoxes(prob: FloatArray, w: Int, h: Int): List<Rect> {
        val size = w * h
        if (visited.size < size) {
            visited = BooleanArray(size)
            stackX = IntArray(size)
            stackY = IntArray(size)
        } else {
            java.util.Arrays.fill(visited, 0, size, false)
        }

        val boxes = ArrayList<Rect>()

        for (startY in 0 until h) {
            for (startX in 0 until w) {
                val startIdx = startY * w + startX
                if (visited[startIdx] || prob[startIdx] < PROB_THRESHOLD) continue

                var minX = startX; var maxX = startX
                var minY = startY; var maxY = startY
                var area = 0
                var sp = 0
                stackX[sp] = startX; stackY[sp] = startY; sp++
                visited[startIdx] = true

                while (sp > 0) {
                    sp--
                    val cx = stackX[sp]; val cy = stackY[sp]
                    area++
                    if (cx < minX) minX = cx
                    if (cx > maxX) maxX = cx
                    if (cy < minY) minY = cy
                    if (cy > maxY) maxY = cy

                    // 4-connected neighbors
                    for ((nx, ny) in arrayOf(cx - 1 to cy, cx + 1 to cy, cx to cy - 1, cx to cy + 1)) {
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
                        val nIdx = ny * w + nx
                        if (visited[nIdx] || prob[nIdx] < PROB_THRESHOLD) continue
                        visited[nIdx] = true
                        stackX[sp] = nx; stackY[sp] = ny; sp++
                    }
                }

                if (area >= MIN_BOX_AREA_PX) {
                    boxes.add(unclip(Rect(minX, minY, maxX + 1, maxY + 1), w, h))
                }
            }
        }

        return sortReadingOrder(boxes)
    }

    private fun unclip(box: Rect, mapW: Int, mapH: Int): Rect {
        val cx = (box.centerX())
        val cy = (box.centerY())
        val halfW = (box.width() * UNCLIP_RATIO / 2f).toInt()
        val halfH = (box.height() * UNCLIP_RATIO / 2f).toInt()
        return Rect(
            (cx - halfW).coerceIn(0, mapW - 1),
            (cy - halfH).coerceIn(0, mapH - 1),
            (cx + halfW).coerceIn(1, mapW),
            (cy + halfH).coerceIn(1, mapH)
        )
    }

    /** Simple top-to-bottom, then left-to-right reading order (good enough for single-column text/signage). */
    private fun sortReadingOrder(boxes: List<Rect>): List<Rect> {
        return boxes.sortedWith(compareBy({ it.top / 20 }, { it.left }))
    }
}
