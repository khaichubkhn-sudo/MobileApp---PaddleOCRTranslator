# Paddle OCR Translator (Android)

Android app: pick an image → run OCR locally with Paddle Lite → review/edit the text →
hand it off to Google Translate via intent.

## Building via GitHub Actions (recommended)

Push this repo to GitHub and the workflow at `.github/workflows/build.yml` builds a debug
APK automatically on every push to `main`. The Paddle Lite Java library and native
libraries must already be present in the repository; the workflow validates them, then
downloads and converts the OCR models and downloads the character dictionary before
running `gradle assembleDebug`. The finished APK is attached to the run as the
`paddle-ocr-translator-debug-apk` artifact.

The model URLs and dictionary URL are hardcoded in the workflow's `env:` block (they're
  stable, canonical PaddleOCR doc links) — edit those directly if you want a different
  language/model pair.

If you'd rather build locally instead of pushing to GitHub Actions, follow Steps 1–4 below
by hand.

## What's implemented here vs. what you still need to add

This project is a complete, buildable Android Studio app skeleton with real logic for
every requirement below. What it does **not** include is the ~150MB of binary Paddle Lite
libraries and trained model weights — those aren't something that can be embedded in
generated source, and PaddleOCR's model host (`paddleocr.bj.bcebos.com`) isn't reachable
from this environment. You need to download three things and drop them into the project
before it will build/run (Steps 1–3 below). Everything else — UI, preprocessing, box
detection, CTC decoding, memory management, translate intent, crash-proofing — is written.

| Requirement | Where it's handled |
|---|---|
| Paddle Lite OCR, 64-bit ARM only | `app/build.gradle` (`abiFilters "arm64-v8a"`), `app/libs/` |
| Image picker, size < 1MB | `MainActivity.onImagePicked` + `ImagePreprocessor.loadAndValidate` |
| Memory bounded across repeated runs (~300MB) | see "Memory design" below |
| Pre-process image to model's input requirements | `ImagePreprocessor.prepareForDetection/Recognition` |
| Debug info in GUI | bottom log panel, `util/DebugLogger.kt` |
| Never crash on unexpected exceptions | `MainActivity.safely{}`, try/catch in every pipeline stage, global `UncaughtExceptionHandler` |
| Max 1–2 threads | `PaddleOcrEngine(threads=2)` → `MobileConfig.setThreads()`, single-thread `ExecutorService` for app-level work |
| Google Translate via intent | `MainActivity.sendToGoogleTranslate` (`ACTION_SEND` to the app, browser fallback) |

## Step 1 — Get the Paddle Lite Java library (64-bit ARM)

1. Go to https://github.com/PaddlePaddle/Paddle-Lite/releases
2. Download the **Android Java full-publish** package for `arm64-v8a` (64-bit ARM), not
  the `armeabi-v7a` package. The asset is typically named something like
  `inference_lite_lib.android.arm64.xxx.tar.gz` (or listed under Android Java demos —
  naming has changed across releases, look for "arm64" + "java").
3. Unpack it. Inside you'll find a `java/` folder containing `PaddlePredictor.jar`
  (some releases ship an `.aar` instead — either works) and native libraries under
  `java/libs/arm64-v8a/`, including `libpaddle_lite_jni.so` and
  `libc++_shared.so`.
4. Copy `PaddlePredictor.jar` (or the `.aar`) into `app/libs/` in this project.
5. Copy both native libraries into `app/src/main/jniLibs/arm64-v8a/` (create that
  folder). Omitting `libc++_shared.so` causes `dlopen failed: libc++_shared.so not
  found` when OCR starts.

## Step 2 — Get + convert the OCR models

The app expects two Paddle-Lite-optimized (`.nb`, naive_buffer format) models:
detection and recognition. PaddleOCR's own model zoo ships regular inference models that
need one conversion pass:

```bash
# 1. Download PP-OCRv3 mobile (small, good default) inference models
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar
tar xf ch_PP-OCRv3_det_slim_infer.tar
tar xf ch_PP-OCRv3_rec_slim_infer.tar

# 2. Get the `paddle_lite_opt` converter (pip install, easiest option)
pip install paddlelite

# 3. Convert both to naive_buffer .nb files
paddle_lite_opt \
  --model_file=./ch_PP-OCRv3_det_slim_infer/inference.pdmodel \
  --param_file=./ch_PP-OCRv3_det_slim_infer/inference.pdiparams \
  --optimize_out_type=naive_buffer \
  --optimize_out=./det_db

paddle_lite_opt \
  --model_file=./ch_PP-OCRv3_rec_slim_infer/inference.pdmodel \
  --param_file=./ch_PP-OCRv3_rec_slim_infer/inference.pdiparams \
  --optimize_out_type=naive_buffer \
  --optimize_out=./rec_crnn
```

This produces `det_db.nb` and `rec_crnn.nb`. Copy both into
`app/src/main/assets/models/` (replacing the placeholder `.txt` file there).

If you need a different language (Japanese, Korean, etc. — matching your CJK translator
project), swap in the matching PP-OCR model from the same model zoo page:
https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md

## Step 3 — Get the character dictionary

Download the dictionary that matches your recognition model, e.g. for the Chinese model
above:
https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/ppocr_keys_v1.txt
(use the "Raw" button). Save it as `app/src/main/assets/labels/ppocr_keys_v1.txt`
(replacing the placeholder `.txt` file there).

## Step 4 — Build

Open the project root in Android Studio (or `./gradlew assembleDebug` from the CLI) and
build normally. Minimum SDK 23.

## Memory design (target: run repeatedly within ~300MB)

- **Predictors are created once and reused.** `PaddleOcrEngine.ensureLoaded()` only builds
  the two `PaddlePredictor` instances the first time; every subsequent image reuses them.
  Rebuilding a predictor per image (re-parsing the model graph) is the single biggest
  source of run-over-run memory growth, so this is the most important optimization here.
- **Reused float buffers.** `ImagePreprocessor` allocates its detection/recognition
  normalization buffers once as instance fields and reuses them on every call instead of
  allocating a new `FloatArray` per image/box.
- **Reused flood-fill buffers.** `DbPostProcess` keeps its `visited`/stack arrays sized to
  the largest map seen so far and reuses them, rather than reallocating per detection run.
- **Bitmaps are recycled immediately** after each stage no longer needs them (resized
  intermediates, per-box crops), not left for the GC to find later.
- **Debug log is bounded** to the last 200 lines (`DebugLogger`), so leaving the app running
  and running OCR many times doesn't grow the log into a memory problem of its own.
- The debug panel calls `logMemory()` after image load and after each OCR run so you can
  watch Java-heap and native-heap usage over repeated runs and confirm it's flat, not
  climbing.

## Threading

Per your requirement, the app never uses more than 2 compute threads:
- `PaddleOcrEngine(threads = 2)` clamps to `[1,2]` and calls `MobileConfig.setThreads()`,
  which controls Paddle Lite's own internal inference threads.
- All app-level OCR work (image decode, preprocess, calling the predictors, postprocess)
  runs on a single `Executors.newSingleThreadExecutor()` created once in `MainActivity` and
  reused for every click, instead of spawning a new thread per operation.

## Known simplification: detection box shapes

The official PaddleOCR pipeline extracts rotated polygons from the detector's probability
map (contours → min-area-rect → "unclip" via the Clipper library), which normally requires
either OpenCV or a bundled C++ clipper. To keep this project pure Kotlin/Java with no extra
native build step, `DbPostProcess` instead does iterative connected-component labeling and
returns **axis-aligned** boxes. This works well for roughly horizontal text (photographed
documents, signage, screenshots — the common case) but won't tightly fit heavily rotated or
curved text lines the way the official pipeline does. If you need that, port
`deploy/lite/db_post_process.cc` + `clipper.cpp` from the PaddleOCR repo's `deploy/lite/`
folder in as JNI code instead of this class.

## Project layout

```
app/src/main/java/com/example/paddleocrtranslator/
  MainActivity.kt              - UI, image picking, orchestration, translate intent, crash-proofing
  ocr/PaddleOcrEngine.kt        - owns the two PaddlePredictor instances, runs the pipeline
  ocr/ImagePreprocessor.kt      - decode/validate/EXIF-rotate, resize+normalize for both models
  ocr/DbPostProcess.kt          - probability map -> bounding boxes (connected components)
  ocr/CtcLabelDecoder.kt        - CTC greedy decode using the character dictionary
  ocr/OcrResult.kt              - result data classes
  util/DebugLogger.kt           - bounded ring-buffer logger feeding the on-screen debug panel
app/src/main/assets/models/     - drop det_db.nb + rec_crnn.nb here (Step 2)
app/src/main/assets/labels/     - drop ppocr_keys_v1.txt here (Step 3)
app/libs/                       - drop PaddlePredictor.jar/.aar here (Step 1)
```
