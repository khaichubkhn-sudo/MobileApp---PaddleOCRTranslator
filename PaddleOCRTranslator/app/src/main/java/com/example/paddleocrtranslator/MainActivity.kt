package com.example.paddleocrtranslator

import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.paddleocrtranslator.ocr.OcrException
import com.example.paddleocrtranslator.ocr.PaddleOcrEngine
import com.example.paddleocrtranslator.util.DebugLogger
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var imagePreview: ImageView
    private lateinit var btnPickImage: Button
    private lateinit var btnRunOcr: Button
    private lateinit var btnTranslate: Button
    private lateinit var editExtractedText: EditText
    private lateinit var debugLogView: TextView

    private lateinit var logger: DebugLogger
    private lateinit var ocrEngine: PaddleOcrEngine

    // Single background thread, created once and reused for every OCR run — this is the
    // "at most 1-2 threads total" boundary at the app level (Paddle Lite's own internal
    // compute threads, set to <=2, are separate and run inside this worker's call to run()).
    private val ocrExecutor = Executors.newSingleThreadExecutor()

    private var selectedImageUri: Uri? = null
    private var selectedBitmap: Bitmap? = null

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            onImagePicked(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Last-resort safety net: log and show a toast instead of letting an exception we
        // didn't anticipate take the whole app down.
        val previousHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                if (::logger.isInitialized) {
                    logger.logError("UNCAUGHT on ${thread.name}", throwable)
                }
            } catch (ignored: Throwable) {
                // logging itself must never throw further
            }
            // Defer to the platform handler so the OS still records/reports it normally;
            // remove this line only if you explicitly want to attempt to keep running.
            previousHandler?.uncaughtException(thread, throwable)
        }

        setContentView(R.layout.activity_main)

        imagePreview = findViewById(R.id.imagePreview)
        btnPickImage = findViewById(R.id.btnPickImage)
        btnRunOcr = findViewById(R.id.btnRunOcr)
        btnTranslate = findViewById(R.id.btnTranslate)
        editExtractedText = findViewById(R.id.editExtractedText)
        debugLogView = findViewById(R.id.textDebugLog)

        logger = DebugLogger(debugLogView)
        ocrEngine = PaddleOcrEngine(applicationContext, logger, threads = 2)

        btnPickImage.setOnClickListener { safely("pick image") { pickImageLauncher.launch("image/*") } }
        btnRunOcr.setOnClickListener { safely("run OCR") { runOcrOnSelectedImage() } }
        btnTranslate.setOnClickListener { safely("send to translate") { sendToGoogleTranslate() } }

        logger.log("App started")
        logger.logMemory(this, "startup")
    }

    override fun onDestroy() {
        safely("shutdown") {
            ocrExecutor.shutdown()
            ocrEngine.release()
            selectedBitmap?.recycle()
            selectedBitmap = null
        }
        super.onDestroy()
    }

    private fun onImagePicked(uri: Uri) {
        selectedImageUri = uri
        btnRunOcr.isEnabled = false
        btnTranslate.isEnabled = false
        editExtractedText.setText("")

        ocrExecutor.execute {
            try {
                val loaded = com.example.paddleocrtranslator.ocr.ImagePreprocessor()
                    .loadAndValidate(contentResolver, uri)

                runOnUiThread {
                    selectedBitmap?.recycle()
                    selectedBitmap = loaded.bitmap
                    imagePreview.setImageBitmap(loaded.bitmap)
                    btnRunOcr.isEnabled = true
                    logger.log("Image loaded: ${loaded.bitmap.width}x${loaded.bitmap.height}, ${loaded.originalBytes / 1024}KB")
                    logger.logMemory(this, "after image load")
                }
            } catch (t: Throwable) {
                handleFailure("Could not load image", t)
            }
        }
    }

    private fun runOcrOnSelectedImage() {
        val bitmap = selectedBitmap
        if (bitmap == null) {
            Toast.makeText(this, "Pick an image first", Toast.LENGTH_SHORT).show()
            return
        }

        btnRunOcr.isEnabled = false
        logger.log("Running OCR...")

        ocrExecutor.execute {
            try {
                val result = ocrEngine.run(bitmap)
                runOnUiThread {
                    editExtractedText.setText(result.joinedText())
                    btnTranslate.isEnabled = result.lines.isNotEmpty()
                    btnRunOcr.isEnabled = true
                    logger.log("OCR done: ${result.lines.size} line(s), det=${result.detectMs}ms rec=${result.recognizeMs}ms")
                    logger.logMemory(this, "after OCR")
                    if (result.lines.isEmpty()) {
                        Toast.makeText(this, "No text detected", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (t: Throwable) {
                handleFailure("OCR failed", t)
                runOnUiThread { btnRunOcr.isEnabled = true }
            }
        }
    }

    private fun sendToGoogleTranslate() {
        val text = editExtractedText.text?.toString()?.trim().orEmpty()
        if (text.isEmpty()) {
            Toast.makeText(this, "Nothing to translate yet", Toast.LENGTH_SHORT).show()
            return
        }

        // Preferred: hand the text to the Google Translate app directly via ACTION_SEND.
        val appIntent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, text)
            setPackage("com.google.android.apps.translate")
        }

        try {
            startActivity(appIntent)
            logger.log("Handed off ${text.length} char(s) to Google Translate app")
            return
        } catch (e: ActivityNotFoundException) {
            logger.log("Google Translate app not installed, falling back to browser")
        } catch (t: Throwable) {
            logger.logError("Google Translate app launch failed, falling back to browser", t)
        }

        // Fallback: open translate.google.com with the text pre-filled, if the app isn't installed.
        try {
            val encoded = Uri.encode(text)
            val webIntent = Intent(
                Intent.ACTION_VIEW,
                Uri.parse("https://translate.google.com/?sl=auto&tl=en&text=$encoded&op=translate")
            )
            startActivity(webIntent)
        } catch (t: Throwable) {
            handleFailure("Could not open Google Translate", t)
        }
    }

    /** Runs [block], catching everything so a UI-thread tap handler can never crash the app. */
    private inline fun safely(what: String, block: () -> Unit) {
        try {
            block()
        } catch (t: Throwable) {
            handleFailure("Failed to $what", t)
        }
    }

    private fun handleFailure(message: String, t: Throwable) {
        val detail = if (t is OcrException) t.message ?: message else message
        logger.logError(detail, t)
        runOnUiThread {
            Toast.makeText(this, detail, Toast.LENGTH_LONG).show()
        }
    }
}
