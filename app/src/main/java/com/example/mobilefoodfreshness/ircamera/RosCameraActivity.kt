package com.example.mobilefoodfreshness.ircamera

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.example.mobilefoodfreshness.R
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.progressindicator.CircularProgressIndicator
import kotlinx.coroutines.*
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException

class RosCameraActivity : AppCompatActivity() {

    private val ui = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private lateinit var preview: ImageView
    private lateinit var status: TextView
    private lateinit var btnShutter: FloatingActionButton

    private lateinit var loadingOverlay: FrameLayout
    private lateinit var centerLoading: CircularProgressIndicator

    private var ros: RosBridgeClient? = null

    @Volatile
    private var latestFrame: Bitmap? = null

    @Volatile
    private var isPreviewFrozen: Boolean = false

    // Python backend endpoint
    private val apiUrl = "http://149.165.159.197:8080/infer"

    private var lastUploadedJpeg: ByteArray? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ros_camera)

        preview = findViewById(R.id.preview)
        status = findViewById(R.id.status)
        btnShutter = findViewById(R.id.btnShutter)
        loadingOverlay = findViewById(R.id.loadingOverlay)
        centerLoading = findViewById(R.id.centerLoading)

        val ip = intent.getStringExtra("robot_ip") ?: "192.168.41.210"
        val port = intent.getIntExtra("robot_port", 9001)
        val topic = intent.getStringExtra("camera_topic") ?: "/bot5/camera_node/image/compressed"
        val wsUrl = "ws://$ip:$port"

        ros = RosBridgeClient(
            wsUrl = wsUrl,
            topic = topic,
            onImage = { jpeg, _ -> handleFrame(jpeg) },
            onState = { ok, err ->
                ui.launch {
                    status.text =
                        if (ok) "Connected to $wsUrl\nTopic: $topic"
                        else "Disconnected: $err"
                }
            }
        )
        ros?.connect()

        btnShutter.setOnClickListener { onCaptureAndSend() }
    }

    private var lastTs = 0L

    private fun handleFrame(jpeg: ByteArray) {
        if (isPreviewFrozen) return

        val now = System.currentTimeMillis()
        if (now - lastTs < 40) return
        lastTs = now

        val bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: return
        latestFrame = bmp

        ui.launch { preview.setImageBitmap(bmp) }
    }

    private fun onCaptureAndSend() {
        val bmp = latestFrame ?: run {
            Toast.makeText(this, "No frame available to send", Toast.LENGTH_SHORT).show()
            return
        }

        isPreviewFrozen = true
        btnShutter.isEnabled = false

        loadingOverlay.visibility = View.VISIBLE
        centerLoading.show()

        ui.launch(Dispatchers.IO) {
            val jpegBytes = ByteArrayOutputStream().use { bos ->
                bmp.compress(Bitmap.CompressFormat.JPEG, 90, bos)
                bos.toByteArray()
            }
            lastUploadedJpeg = jpegBytes

            val mediaType = "image/jpeg".toMediaType()
            val imageBody = jpegBytes.toRequestBody(mediaType)

            val body = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", "image.jpg", imageBody)
                .addFormDataPart("draw_mask", "false")
                .addFormDataPart("box_threshold", "0.4")
                .addFormDataPart("text_threshold", "0.3")
                .build()

            val req = Request.Builder()
                .url(apiUrl)
                .post(body)
                .build()

            ApiClient.http.newCall(req).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    ui.launch {
                        loadingOverlay.visibility = View.GONE
                        isPreviewFrozen = false
                        btnShutter.isEnabled = true

                        Toast.makeText(
                            this@RosCameraActivity,
                            "Upload failed: ${e.message}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }

                override fun onResponse(call: Call, response: Response) {
                    val bodyStr = response.body?.string().orEmpty()
                    ui.launch {
                        btnShutter.isEnabled = true

                        if (!response.isSuccessful) {
                            loadingOverlay.visibility = View.GONE
                            isPreviewFrozen = false
                            showResultDialog("HTTP ${response.code}", bodyStr.take(1000))
                            return@launch
                        }

                        val jpeg = lastUploadedJpeg ?: ByteArray(0)
                        val imageUri = saveJpegToCache(jpeg)

                        startActivity(
                            Intent(
                                this@RosCameraActivity,
                                IRCameraResultActivity::class.java    // ✅ New results screen
                            ).apply {
                                putExtra("image_uri", imageUri.toString())
                                putExtra("raw_json", bodyStr)      // Full JSON payload from backend
                                putExtra("jpeg_bytes", jpeg)       // Original JPEG for refresh logic
                                putExtra("needs_rotation", false)
                                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            }
                        )
                    }
                }
            })
        }
    }

    private fun saveJpegToCache(bytes: ByteArray): Uri {
        val file = File(cacheDir, "last_capture.jpg")
        file.outputStream().use { it.write(bytes) }
        return FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)
    }

    private fun showResultDialog(title: String, message: String) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }

    override fun onResume() {
        super.onResume()
        isPreviewFrozen = false
        loadingOverlay.visibility = View.GONE
        centerLoading.hide()
        btnShutter.isEnabled = true
    }

    override fun onDestroy() {
        super.onDestroy()
        ros?.close()
        ui.cancel()
    }
}
