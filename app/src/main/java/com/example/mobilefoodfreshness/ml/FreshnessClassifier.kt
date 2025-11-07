package com.example.mobilefoodfreshness.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.DataType
import kotlin.random.Random

class FreshnessClassifier(ctx: Context) {
    private var interpreter: Interpreter? = null
    private val inputSize = 224
    init {
        try {
            val model = FileUtil.loadMappedFile(ctx, "freshness_cls.tflite")
            interpreter = Interpreter(model)
        } catch (e: Exception) {
            Log.w("FreshnessClassifier", "No model found. Using mock score.")
        }
    }

    fun score(bitmap: Bitmap): Float {
        interpreter?.let {
            val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            val input = TensorImage.fromBitmap(resized).buffer
            val output = TensorBuffer.createFixedSize(intArrayOf(1, 1), DataType.FLOAT32)
            it.run(input, output.buffer)
            return output.floatArray.first()
        }
        return (0.3f + Random.nextFloat() * 0.7f)
    }
}
