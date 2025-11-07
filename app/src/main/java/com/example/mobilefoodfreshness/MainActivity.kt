
package com.example.mobilefoodfreshness

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.commit
import com.example.mobilefoodfreshness.camera.CameraFragment
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

        if (savedInstanceState == null) {
            supportFragmentManager.commit {
                replace(R.id.container, CameraFragment())
            }
        }
    }
}
