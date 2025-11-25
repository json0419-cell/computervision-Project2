package com.example.mobilefoodfreshness.ircamera

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import androidx.appcompat.widget.AppCompatImageView
import kotlin.math.min

data class BoundingBox(
    val rect: RectF,          // Original image coordinates (x1, y1, x2, y2)
    val className: String,    // Food name
    val score: Float?,        // Confidence score (0~1)
    val rawLabel: String      // Raw label from backend (debug helper)
)

class BoundingBoxImageView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : AppCompatImageView(context, attrs) {

    private var boxes: List<BoundingBox> = emptyList()

    // Paint used for drawing bounding boxes
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    // Semi-transparent background behind the text label
    private val textBgPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#AA000000")
        isAntiAlias = true
    }

    // Paint used for the text itself
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        isAntiAlias = true
    }

    // Assign a distinct color per class
    private val classColorMap = mutableMapOf<String, Int>()
    private val colorPalette = listOf(
        Color.parseColor("#FF5252"), // red
        Color.parseColor("#FF9800"), // orange
        Color.parseColor("#FFEB3B"), // yellow
        Color.parseColor("#4CAF50"), // green
        Color.parseColor("#03A9F4"), // light blue
        Color.parseColor("#3F51B5"), // indigo
        Color.parseColor("#9C27B0"), // purple
        Color.parseColor("#795548")  // brown
    )

    private var colorIndex = 0

    fun setBoxes(newBoxes: List<BoundingBox>) {
        boxes = newBoxes
        invalidate()
    }

    private fun colorForClass(className: String): Int {
        val key = className.ifBlank { "unknown" }
        return classColorMap.getOrPut(key) {
            val c = colorPalette[colorIndex % colorPalette.size]
            colorIndex++
            c
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (boxes.isEmpty()) return
        val d = drawable ?: return

        val imgW = d.intrinsicWidth.toFloat()
        val imgH = d.intrinsicHeight.toFloat()
        if (imgW <= 0f || imgH <= 0f) return

        val viewW = width.toFloat()
        val viewH = height.toFloat()

        // ====== Mirror ImageView fitCenter math so boxes align ======
        val scale = min(viewW / imgW, viewH / imgH)
        val dx = (viewW - imgW * scale) / 2f
        val dy = (viewH - imgH * scale) / 2f

        for (box in boxes) {
            val cls = box.className.ifBlank { "unknown" }
            val color = colorForClass(cls)
            boxPaint.color = color

            // Convert original image coordinates into view coordinates
            val scaledRect = RectF(
                box.rect.left * scale + dx,
                box.rect.top * scale + dy,
                box.rect.right * scale + dx,
                box.rect.bottom * scale + dy
            )

            // Draw the rectangle
            canvas.drawRect(scaledRect, boxPaint)

            // Text: class name (replace underscores) + confidence
            val labelName = cls.replace("_", " ")
            val scoreStr = box.score?.let {
                " %.2f".format(it)
            }.orEmpty()
            val text = "$labelName$scoreStr"

            // Measure text bounds for the background rect
            val textBounds = Rect()
            textPaint.getTextBounds(text, 0, text.length, textBounds)
            val padding = 6f
            val textLeft = scaledRect.left
            val textTop = scaledRect.top - textBounds.height() - padding * 2

            val bgTop = if (textTop < 0) 0f else textTop
            val bgRect = RectF(textLeft, bgTop, textLeft + textBounds.width() + padding * 2, bgTop + textBounds.height() + padding * 2)

            // Draw the background pill plus the text
            canvas.drawRoundRect(bgRect, 8f, 8f, textBgPaint)
            canvas.drawText(
                text,
                bgRect.left + padding,
                bgRect.bottom - padding,
                textPaint
            )
        }
    }
}
