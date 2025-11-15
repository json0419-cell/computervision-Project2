package com.example.mobilefoodfreshness

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import com.example.mobilefoodfreshness.ircamera.RobotPickerActivity
import org.opencv.android.OpenCVLoader


class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (OpenCVLoader.initDebug()) {
            Log.i("OpenCV", "OpenCV loaded successfully.")
        } else {
            Log.e("OpenCV", "Failed to load OpenCV.")
        }

        // use local camera
        findViewById<Button>(R.id.btnLocal).setOnClickListener {
            startActivity(Intent(this, LocalCameraActivity::class.java))
        }

        // remote camera
        findViewById<Button>(R.id.btnIR).setOnClickListener {
            startActivity(Intent(this, RobotPickerActivity::class.java))
        }
    }
}
