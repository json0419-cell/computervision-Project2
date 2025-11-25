package com.example.mobilefoodfreshness.ui

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.mobilefoodfreshness.MainActivity
import com.example.mobilefoodfreshness.R

class ResultActivity : AppCompatActivity() {
    companion object { var bitmapCache: Bitmap? = null }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)
        val text = intent.getStringExtra("summary") ?: "No summary"
        val tv = findViewById<android.widget.TextView>(R.id.title)
        val img = findViewById<android.widget.ImageView>(R.id.capturedImage)
        tv.text = text
        img.setImageBitmap(bitmapCache)

        findViewById<android.widget.Button>(R.id.back).setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
        }
    }
}
