package com.example.mobilefoodfreshness.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.example.mobilefoodfreshness.util.Detection
import kotlin.math.roundToInt

class OverlayView @JvmOverloads constructor(
    ctx: Context, attrs: AttributeSet? = null
) : View(ctx, attrs) {
    private val boxes = mutableListOf<Detection>()
    private var frameW = 0
    private var frameH = 0

    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 6f }
    private val textPaint = Paint().apply { textSize = 36f; isAntiAlias = true; style = Paint.Style.FILL }

    fun update(newBoxes: List<Detection>, w: Int, h: Int) {
        boxes.clear(); boxes.addAll(newBoxes); frameW = w; frameH = h; invalidate()
    }

    fun currentSummary(): String {
        if (boxes.isEmpty()) return "No items detected."
        return boxes.joinToString("\n") {
            val status = when {
                (it.fresh ?: 0f) >= 0.7f -> "fresh"
                (it.fresh ?: 0f) >= 0.4f -> "slightly spoiled"
                else -> "spoiled"
            }
            "#${it.trackId} ${it.label} - $status (${((it.fresh ?: 0f)*100).roundToInt()}%)"
        }
    }

    override fun onDraw(canvas: Canvas) {
        if (frameW == 0 || frameH == 0) return
        val sx = width / frameW.toFloat()
        val sy = height / frameH.toFloat()
        for (d in boxes) {
            val color = when {
                (d.fresh ?: 0f) >= 0.7f -> Color.GREEN
                (d.fresh ?: 0f) >= 0.4f -> Color.YELLOW
                else -> Color.RED
            }
            boxPaint.color = color
            textPaint.color = color
            val r = RectF(d.bbox.left*sx, d.bbox.top*sy, d.bbox.right*sx, d.bbox.bottom*sy)
            canvas.drawRect(r, boxPaint)
            val label = "${d.label}: ${(d.score*100).roundToInt()}% fresh=${((d.fresh?:0f)*100).roundToInt()}%"
            canvas.drawText(label, r.left, (r.top - 8).coerceAtLeast(24f), textPaint)
        }
    }
}
