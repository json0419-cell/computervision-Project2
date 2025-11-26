package com.example.mobilefoodfreshness.camera

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class DottedSquareView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        style = Paint.Style.STROKE // We only want the outline
        strokeWidth = 5f // Adjust the thickness as needed
        // Define the dash pattern: 10f is the "on" interval (dot/dash length), 
        // and 10f is the "off" interval (gap length)
        pathEffect = DashPathEffect(floatArrayOf(10f, 10f), 0f)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val width = width.toFloat()
        val height = height.toFloat()
        val side = Math.min(width, height) * 0.5f // Square side is 50% of the smaller dimension
        val left = (width - side) / 2
        val top = (height - side) / 2
        val right = left + side
        val bottom = top + side

        // Draw the rectangle (square) in the center of the canvas
        canvas.drawRect(left, top, right, bottom, paint)
    }
}
