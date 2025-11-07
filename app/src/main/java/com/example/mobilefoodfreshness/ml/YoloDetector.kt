package com.example.mobilefoodfreshness.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.example.mobilefoodfreshness.util.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import org.tensorflow.lite.task.vision.detector.Detection as TfDetection
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.support.image.TensorImage

class YoloDetector(ctx: Context) {
    private var detector: ObjectDetector? = null

    init {
        try {
            val opts = ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(0.35f)
                .setMaxResults(10)
                .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
                .build()

            detector = ObjectDetector.createFromFileAndOptions(
                ctx,
                "yolov8n_food.tflite",
                opts
            )

            Log.i("YoloDetector", "TFLite model loaded successfully.")
        } catch (e: Exception) {
            Log.w("YoloDetector", "No TFLite model found. Using mock output.")
        }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        detector?.let { det ->
            val results = det.detect(TensorImage.fromBitmap(bitmap))
            return results.map { d ->
                val c = d.categories.first()
                val r = d.boundingBox
                Detection(
                    label = c.label,      // ✅ changed from categoryName → label
                    score = c.score,
                    bbox = Rect(r.left.toInt(), r.top.toInt(), r.right.toInt(), r.bottom.toInt()),
                    trackId = -1,
                    fresh = null
                )
            }
        }

        // If no model found, return mock detection
        val w = bitmap.width
        val h = bitmap.height
        val bw = (w * 0.35).toInt()
        val bh = (h * 0.35).toInt()
        return listOf(
            Detection(
                label = "mock",
                score = 0.9f,
                bbox = Rect(100, 100, 100 + bw, 100 + bh),
                trackId = -1,
                fresh = 0.6f
            )
        )
    }
}
