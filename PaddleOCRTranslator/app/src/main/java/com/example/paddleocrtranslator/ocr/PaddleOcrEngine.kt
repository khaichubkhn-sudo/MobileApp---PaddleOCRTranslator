package com.example.paddleocrtranslator.ocr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import com.baidu.paddle.lite.MobileConfig
import com.baidu.paddle.lite.PaddlePredictor
import com.baidu.paddle.lite.PowerMode
import com.example.paddleocrtranslator.util.DebugLogger
import java.io.File
import java.io.FileOutputStream

/**
 * Owns the two Paddle Lite predictors (text detection + text recognition) and runs the
 * full OCR pipeline over a bitmap.
 *
 * Threading: [threads] is clamped to at most 2, per requirement, and is passed straight to
 * MobileConfig.setThreads() — that controls how many CPU threads *Paddle Lite itself* uses
 * internally for a single inference call. The engine's public [run] method is expected to be
 * called from a single background worker thread owned by the caller (see MainActivity); this
 * class does not spin up any threads of its own, so total concurrent compute stays bounded.
 *
 * Memory: predictors are created once (in [ensureLoaded]) and reused for every image, instead
 * of being rebuilt per run — rebuilding a PaddlePredictor re-parses the model graph and is by
 * far the biggest source of run-over-run memory growth if done repeatedly. Bitmaps created
 * mid-pipeline are recycled as soon as they're no longer needed.
 */
class PaddleOcrEngine(
    private val context: Context,
    private val logger: DebugLogger,
    threads: Int = 2
) {
    private val threadCount = threads.coerceIn(1, 2)

    private var detPredictor: PaddlePredictor? = null
    private var recPredictor: PaddlePredictor? = null

    private val preprocessor = ImagePreprocessor()
    private val dbPostProcess = DbPostProcess()
    private val labelDecoder by lazy { CtcLabelDecoder(context) }

    companion object {
        // Filenames expected in assets/models/ (Paddle Lite "naive_buffer" .nb models,
        // produced with `paddle_lite_opt --optimize_out_type=naive_buffer`). See README.md.
        private const val DET_MODEL_ASSET = "models/det_db.nb"
        private const val REC_MODEL_ASSET = "models/rec_crnn.nb"
    }

    @Synchronized
    private fun ensureLoaded() {
        if (detPredictor != null && recPredictor != null) return

        try {
            val detFile = copyAssetToCache(DET_MODEL_ASSET, "det_db.nb")
            val recFile = copyAssetToCache(REC_MODEL_ASSET, "rec_crnn.nb")

            detPredictor = createPredictor(detFile)
            recPredictor = createPredictor(recFile)
            logger.log("Loaded detection + recognition models (threads=$threadCount)")
        } catch (t: Throwable) {
            // Don't leave a half-initialized engine around.
            detPredictor = null
            recPredictor = null
            throw OcrException("Failed to load OCR models", t)
        }
    }

    private fun createPredictor(modelFile: File): PaddlePredictor {
        val config = MobileConfig()
        config.setModelFromFile(modelFile.absolutePath)
        config.setThreads(threadCount)
        config.setPowerMode(PowerMode.LITE_POWER_LOW) // favor stability/battery over raw speed on phones
        return PaddlePredictor.createPaddlePredictor(config)
    }

    /**
     * Copies a model from assets (compressed inside the APK) to the app's cache dir once;
     * Paddle Lite needs a real filesystem path, it can't load directly from an AssetFileDescriptor.
     * Subsequent calls reuse the cached copy instead of re-extracting it.
     */
    private fun copyAssetToCache(assetPath: String, cacheName: String): File {
        val outFile = File(context.cacheDir, cacheName)
        if (outFile.exists() && outFile.length() > 0) return outFile

        try {
            context.assets.open(assetPath).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buf = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        output.write(buf, 0, n)
                    }
                }
            }
        } catch (t: Throwable) {
            throw OcrException(
                "Model file '$assetPath' not found in assets. Convert it with paddle_lite_opt " +
                    "and place it there — see README.md.",
                t
            )
        }
        return outFile
    }

    /** Runs detection + recognition over [srcBitmap] and returns text lines in reading order. */
    fun run(srcBitmap: Bitmap): OcrResult {
        ensureLoaded()
        val det = detPredictor ?: throw OcrException("Detection model not loaded")
        val rec = recPredictor ?: throw OcrException("Recognition model not loaded")

        val detStart = System.currentTimeMillis()
        val boxesOnOriginal = runDetection(det, srcBitmap)
        val detMs = System.currentTimeMillis() - detStart
        logger.log("Detection: ${boxesOnOriginal.size} candidate box(es) in ${detMs}ms")

        val recStart = System.currentTimeMillis()
        val lines = ArrayList<OcrLine>()
        for (box in boxesOnOriginal) {
            try {
                val (text, conf) = runRecognition(rec, srcBitmap, box)
                if (text.isNotBlank()) {
                    lines.add(OcrLine(box, text, conf))
                }
            } catch (t: Throwable) {
                // One bad crop shouldn't take down the whole result set.
                logger.logError("Recognition failed for box $box, skipping", t)
            }
        }
        val recMs = System.currentTimeMillis() - recStart
        logger.log("Recognition: ${lines.size} line(s) with text in ${recMs}ms")

        return OcrResult(lines, detMs, recMs)
    }

    private fun runDetection(predictor: PaddlePredictor, srcBitmap: Bitmap): List<Rect> {
        val detInput = preprocessor.prepareForDetection(srcBitmap)

        val inputTensor = predictor.getInput(0)
        inputTensor.resize(longArrayOf(1, 3, detInput.inputH.toLong(), detInput.inputW.toLong()))
        inputTensor.setData(detInput.chw)

        predictor.run()

        val outputTensor = predictor.getOutput(0)
        val outShape = outputTensor.shape() // expected [1, 1, H, W]
        val outH = outShape[outShape.size - 2].toInt()
        val outW = outShape[outShape.size - 1].toInt()
        val probMap = outputTensor.floatData

        val mapBoxes = dbPostProcess.extractBoxes(probMap, outW, outH)

        // Map boxes from the (possibly downscaled/padded) probability map back to original pixels.
        // The prob map is typically the same resolution as the model input; scaleX/scaleY already
        // account for input-vs-original scaling, and mapW/mapH vs inputW/inputH covers any stride
        // rounding the model applies internally.
        val mapToInputX = detInput.inputW.toFloat() / outW
        val mapToInputY = detInput.inputH.toFloat() / outH

        return mapBoxes.map { r ->
            Rect(
                (r.left * mapToInputX * detInput.scaleX).toInt(),
                (r.top * mapToInputY * detInput.scaleY).toInt(),
                (r.right * mapToInputX * detInput.scaleX).toInt().coerceAtMost(srcBitmap.width),
                (r.bottom * mapToInputY * detInput.scaleY).toInt().coerceAtMost(srcBitmap.height)
            )
        }
    }

    private fun runRecognition(predictor: PaddlePredictor, srcBitmap: Bitmap, box: Rect): Pair<String, Float> {
        val (chw, width, height) = preprocessor.prepareForRecognition(srcBitmap, box)

        val inputTensor = predictor.getInput(0)
        inputTensor.resize(longArrayOf(1, 3, height.toLong(), width.toLong()))
        // Only the first 3*width*height floats of the reused buffer are valid for this call.
        inputTensor.setData(chw.copyOfRange(0, 3 * width * height))

        predictor.run()

        val outputTensor = predictor.getOutput(0)
        val outShape = outputTensor.shape() // expected [1, timesteps, numClasses]
        val timesteps = outShape[1].toInt()
        val numClasses = outShape[2].toInt()
        val logits = outputTensor.floatData

        return labelDecoder.decode(logits, timesteps, numClasses)
    }

    /** Releases native resources. Call from onDestroy(). */
    fun release() {
        // The Paddle Lite Java API does not expose an explicit predictor.release() in all
        // releases; nulling the references lets the GC + JNI finalizers reclaim native memory.
        // If your version of the jar exposes predictor.release()/destroy(), call it here first.
        detPredictor = null
        recPredictor = null
    }
}
