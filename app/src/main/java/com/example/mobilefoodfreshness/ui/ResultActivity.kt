package com.example.mobilefoodfreshness.ui

import android.graphics.Bitmap
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.mobilefoodfreshness.R

class ResultActivity : AppCompatActivity() {
    companion object { var bitmapCache: Bitmap? = null }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)
        val text = intent.getStringExtra("summary") ?: "No summary"
        val tv = findViewById<android.widget.TextView>(R.id.tvSummary)
        val img = findViewById<android.widget.ImageView>(R.id.imgSnapshot)
        tv.text = text
        img.setImageBitmap(bitmapCache)
    }
}
