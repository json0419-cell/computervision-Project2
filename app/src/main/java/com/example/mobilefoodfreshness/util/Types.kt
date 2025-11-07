package com.example.mobilefoodfreshness.util

import android.graphics.Bitmap
import android.graphics.Rect
import androidx.camera.core.ImageProxy
import android.graphics.YuvImage
import android.graphics.ImageFormat
import android.graphics.BitmapFactory
import java.nio.ByteBuffer
import android.graphics.Rect as ARect

data class Detection(
    val label: String,
    val score: Float,
    val bbox: Rect,
    val trackId: Int,
    val fresh: Float?
)

fun ImageProxy.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer
    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()
    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = java.io.ByteArrayOutputStream()
    yuv.compressToJpeg(ARect(0, 0, width, height), 90, out)
    return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
}


