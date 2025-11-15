
package com.example.mobilefoodfreshness

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.commit
import com.example.mobilefoodfreshness.camera.CameraFragment
import org.opencv.android.OpenCVLoader

class LocalCameraActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_local_camera)

        if (savedInstanceState == null) {
            supportFragmentManager.commit {
                replace(R.id.container, CameraFragment())
            }
        }
    }
}
