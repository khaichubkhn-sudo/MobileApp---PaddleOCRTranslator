package com.example.paddleocrtranslator.ocr

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayOutputStream
import java.io.InputStream

/**
 * Everything involved in turning a user-picked image Uri into normalized float tensors
 * that the Paddle Lite models expect.
 *
 * Memory strategy:
 *  - The raw file is read into a small ByteArray only long enough to check its size and decode
 *    the bitmap; it's discarded immediately after.
 *  - Normalization writes into a small set of *reusable* FloatArray buffers held as fields on
 *    this class (allocated once, cleared and reused on every call) rather than allocating a new
 *    float[] per image/box. This is the practical Kotlin/JVM equivalent of "static" buffers,
 *    since the JVM has no raw stack/static-RAM concept for arrays the way native C does — reuse
 *    is what actually keeps the heap from growing run over run.
 *  - Bitmaps are recycled by the caller (PaddleOcrEngine) as soon as each stage is done with them.
 */
class ImagePreprocessor {

    companion object {
        const val MAX_IMAGE_BYTES = 1 * 1024 * 1024 // 1MB hard cap, per requirement

        // Detection model: PaddleOCR DB models accept a dynamic HxW as long as both are
        // multiples of 32. We cap the longer side to keep RAM + compute bounded on-device.
        const val DET_MAX_SIDE = 640
        const val DET_STRIDE = 32

        // Recognition model: fixed height 32 (standard PP-OCR rec input), dynamic width capped.
        const val REC_HEIGHT = 32
        const val REC_MAX_WIDTH = 320

        // ImageNet-style normalization used by PP-OCR det/rec models.
        val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    // Reused across calls - sized for the worst case, cleared before each use.
    private val detBuffer = FloatArray(3 * DET_MAX_SIDE * DET_MAX_SIDE)
    private val recBuffer = FloatArray(3 * REC_HEIGHT * REC_MAX_WIDTH)

    class LoadedImage(val bitmap: Bitmap, val originalBytes: Int)

    /** Reads [uri], enforces the size cap, and returns an EXIF-corrected, decoded bitmap. */
    fun loadAndValidate(resolver: ContentResolver, uri: Uri): LoadedImage {
        val bytes = readAllBytesCapped(resolver, uri)
            ?: throw OcrException("Could not read the selected image")

        if (bytes.size > MAX_IMAGE_BYTES) {
            throw OcrException(
                "Image is ${bytes.size / 1024}KB, which exceeds the ${MAX_IMAGE_BYTES / 1024}KB limit"
            )
        }

        val rotationDegrees = readExifRotation(bytes)

        val opts = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            ?: throw OcrException("Unsupported or corrupt image format")

        val rotated = if (rotationDegrees != 0) {
            val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            val out = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, m, true)
            if (out !== decoded) decoded.recycle()
            out
        } else {
            decoded
        }

        return LoadedImage(rotated, bytes.size)
    }

    /**
     * Reads the stream fully but bails out early (without allocating the whole remaining
     * stream) once it's clear the file is already over the cap, so an oversized file doesn't
     * force a large transient allocation just to be rejected.
     */
    private fun readAllBytesCapped(resolver: ContentResolver, uri: Uri): ByteArray? {
        var input: InputStream? = null
        return try {
            input = resolver.openInputStream(uri) ?: return null
            val out = ByteArrayOutputStream(64 * 1024)
            val chunk = ByteArray(32 * 1024)
            var total = 0
            while (true) {
                val read = input.read(chunk)
                if (read < 0) break
                total += read
                if (total > MAX_IMAGE_BYTES) {
                    // Keep counting isn't useful past this point - report as "too big" using
                    // what we know so far, without buffering the rest of a huge file.
                    out.write(chunk, 0, read)
                    return out.toByteArray()
                }
                out.write(chunk, 0, read)
            }
            out.toByteArray()
        } finally {
            input?.close()
        }
    }

    private fun readExifRotation(bytes: ByteArray): Int {
        return try {
            val exif = ExifInterface(bytes.inputStream())
            when (exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90
                ExifInterface.ORIENTATION_ROTATE_180 -> 180
                ExifInterface.ORIENTATION_ROTATE_270 -> 270
                else -> 0
            }
        } catch (t: Throwable) {
            0 // EXIF is best-effort; never fail the whole pipeline over it.
        }
    }

    /** Holds the resized dimensions actually used, needed to map detection boxes back to the original image. */
    class DetInput(val chw: FloatArray, val inputW: Int, val inputH: Int, val scaleX: Float, val scaleY: Float)

    /** Resize to fit within DET_MAX_SIDE, pad up to a multiple of DET_STRIDE, normalize into the reused buffer. */
    fun prepareForDetection(src: Bitmap): DetInput {
        val longSide = maxOf(src.width, src.height).toFloat()
        val scale = if (longSide > DET_MAX_SIDE) DET_MAX_SIDE / longSide else 1f

        val resizedW = roundToStride((src.width * scale).toInt().coerceAtLeast(1))
        val resizedH = roundToStride((src.height * scale).toInt().coerceAtLeast(1))

        val resized = Bitmap.createScaledBitmap(src, resizedW, resizedH, true)
        val chw = normalizeToChw(resized, detBuffer, resizedW, resizedH)
        if (resized !== src) resized.recycle()

        return DetInput(
            chw = chw,
            inputW = resizedW,
            inputH = resizedH,
            scaleX = src.width.toFloat() / resizedW,
            scaleY = src.height.toFloat() / resizedH
        )
    }

    /** Crops [box] out of [src], resizes to fixed rec height keeping aspect ratio, normalizes. */
    fun prepareForRecognition(src: Bitmap, box: android.graphics.Rect): Triple<FloatArray, Int, Int> {
        val safeBox = android.graphics.Rect(
            box.left.coerceIn(0, src.width - 1),
            box.top.coerceIn(0, src.height - 1),
            box.right.coerceIn(1, src.width),
            box.bottom.coerceIn(1, src.height)
        )
        val cropW = (safeBox.right - safeBox.left).coerceAtLeast(1)
        val cropH = (safeBox.bottom - safeBox.top).coerceAtLeast(1)

        val cropped = Bitmap.createBitmap(src, safeBox.left, safeBox.top, cropW, cropH)

        val targetW = (cropW.toFloat() / cropH * REC_HEIGHT).toInt().coerceIn(8, REC_MAX_WIDTH)
        val resized = Bitmap.createScaledBitmap(cropped, targetW, REC_HEIGHT, true)
        cropped.recycle()

        val chw = normalizeToChw(resized, recBuffer, targetW, REC_HEIGHT, padToWidth = REC_MAX_WIDTH)
        if (resized !== src) resized.recycle()

        return Triple(chw, targetW, REC_HEIGHT)
    }

    /**
     * Writes HWC pixel data from [bmp] into [dest] as normalized CHW float data.
     * [dest] is reused across calls (its capacity must already be big enough); only the
     * first w*h*3 floats (or padToWidth*h*3 if padding) are meaningful for this call.
     */
    private fun normalizeToChw(
        bmp: Bitmap,
        dest: FloatArray,
        w: Int,
        h: Int,
        padToWidth: Int = w
    ): FloatArray {
        val effectiveW = padToWidth
        val planeSize = effectiveW * h
        // Zero only the region we'll actually use this call (padding included), not the whole buffer.
        java.util.Arrays.fill(dest, 0, minOf(3 * planeSize, dest.size), 0f)

        val row = IntArray(w)
        for (y in 0 until h) {
            bmp.getPixels(row, 0, w, 0, y, w, 1)
            for (x in 0 until w) {
                val px = row[x]
                val r = ((px shr 16) and 0xFF) / 255f
                val g = ((px shr 8) and 0xFF) / 255f
                val b = (px and 0xFF) / 255f

                val idx = y * effectiveW + x
                dest[0 * planeSize + idx] = (r - MEAN[0]) / STD[0]
                dest[1 * planeSize + idx] = (g - MEAN[1]) / STD[1]
                dest[2 * planeSize + idx] = (b - MEAN[2]) / STD[2]
            }
        }
        return dest
    }

    private fun roundToStride(v: Int): Int {
        val r = v - (v % DET_STRIDE)
        return if (r < DET_STRIDE) DET_STRIDE else r
    }
}
