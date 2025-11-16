package com.example.mobilefoodfreshness.camera

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.fragment.app.Fragment
import com.example.mobilefoodfreshness.R
import com.example.mobilefoodfreshness.ui.ResultActivity
import com.example.mobilefoodfreshness.ui.OverlayView
import com.example.mobilefoodfreshness.ml.YoloDetector
import com.example.mobilefoodfreshness.ml.FreshnessClassifier
import com.example.mobilefoodfreshness.tracking.BoxTracker
import java.util.concurrent.Executors

class CameraFragment : Fragment() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private val executor = Executors.newSingleThreadExecutor()

    private lateinit var detector: YoloDetector
    private lateinit var classifier: FreshnessClassifier
    private val tracker = BoxTracker()
    private var lastFrame: Bitmap? = null

    private val reqPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, s: Bundle?): View {
        val v = inflater.inflate(R.layout.fragment_camera, container, false)
        previewView = v.findViewById(R.id.previewView)
        overlay = v.findViewById(R.id.overlay)
        v.findViewById<android.widget.Button>(R.id.btnFinish).setOnClickListener {
            val intent = Intent(requireContext(), ResultActivity::class.java)
            intent.putExtra("summary", overlay.currentSummary())
            ResultActivity.bitmapCache = lastFrame
            startActivity(intent)
        }
        return v
    }

    override fun onResume() {
        super.onResume()
        reqPermission.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        detector = YoloDetector(requireContext())
        classifier = FreshnessClassifier(requireContext())

        val provider = ProcessCameraProvider.getInstance(requireContext()).get()
        val preview = Preview.Builder()
            .setTargetResolution(Size(1280, 720)).build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(1280, 720))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        analysis.setAnalyzer(executor) { img ->
            val bmp = img.toBitmap()
            lastFrame = bmp
            img.close()
        }

        provider.unbindAll()
        provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
    }
}
