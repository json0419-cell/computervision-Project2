package com.example.mobilefoodfreshness.ircamera

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.example.mobilefoodfreshness.R
import kotlinx.coroutines.*
import okhttp3.Call
import okhttp3.Callback
import okhttp3.Response
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException

class RosCameraActivity : AppCompatActivity() {
    private val ui = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private lateinit var preview: ImageView
    private lateinit var status: TextView
    private lateinit var btnShutter: com.google.android.material.floatingactionbutton.FloatingActionButton
    private lateinit var uploadProgress: com.google.android.material.progressindicator.CircularProgressIndicator

    private var ros: RosBridgeClient? = null
    @Volatile private var latestFrame: Bitmap? = null

    private val apiUrl = "https://vertex-proxy-646427707803.us-central1.run.app/infer"
    private val prompt =
        "Extract all food items visible in the image and return a JSON array of {name, edible}. Output JSON only."

    // keep last uploaded bytes so we can forward to result screen
    private var lastUploadedJpeg: ByteArray? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ros_camera)

        preview = findViewById(R.id.preview)
        status = findViewById(R.id.status)
        btnShutter = findViewById(R.id.btnShutter)
        uploadProgress = findViewById(R.id.uploadProgress)

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
                    status.text = if (ok) "Connected to $wsUrl\nTopic: $topic" else "Disconnected: $err"
                }
            }
        )
        ros?.connect()

        btnShutter.setOnClickListener { onCaptureAndSend() }
    }

    private var lastTs = 0L
    private fun handleFrame(jpeg: ByteArray) {
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

        btnShutter.isEnabled = false
        uploadProgress.show()
        uploadProgress.visibility = android.view.View.VISIBLE

        ui.launch(Dispatchers.IO) {
            val jpegBytes = ByteArrayOutputStream().use { bos ->
                bmp.compress(Bitmap.CompressFormat.JPEG, 90, bos)
                bos.toByteArray()
            }
            lastUploadedJpeg = jpegBytes

            val req = ApiClient.buildRequest(apiUrl, jpegBytes, prompt)
            ApiClient.http.newCall(req).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    ui.launch {
                        uploadProgress.visibility = android.view.View.GONE
                        btnShutter.isEnabled = true
                        Toast.makeText(this@RosCameraActivity, "Upload failed: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                }

                override fun onResponse(call: Call, response: Response) {
                    val bodyStr = response.body?.string().orEmpty()
                    ui.launch {
                        uploadProgress.visibility = android.view.View.GONE
                        btnShutter.isEnabled = true

                        if (!response.isSuccessful) {
                            showResultDialog("HTTP ${response.code}", bodyStr.take(1000))
                            return@launch
                        }

                        // sanitize to a clean JSON array: handles markdown/code fences/extra text/BOM etc.
                        val cleanJson = coerceJsonArray(bodyStr)
                        if (cleanJson == null) {
                            showResultDialog("JSON parse failed", bodyStr.take(1000))
                            return@launch
                        }

                        val imageUri = saveJpegToCache(lastUploadedJpeg ?: ByteArray(0))
                        startActivity(Intent(this@RosCameraActivity, ResultActivity::class.java).apply {
                            putExtra("image_uri", imageUri.toString())
                            putExtra("result_json", cleanJson)  // pass sanitized JSON array
                            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        })
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

    // --- JSON coercion: make best effort to extract the first valid JSON array ---
    private fun coerceJsonArray(raw: String): String? {
        // remove UTF-8 BOM & zero-width spaces
        val s = raw.replace("\uFEFF", "").replace("\u200B", "").trim()

        // 1) ```json ... ``` fenced block
        val fenceStart = s.indexOf("```")
        if (fenceStart >= 0) {
            val fenceEnd = s.indexOf("```", fenceStart + 3)
            if (fenceEnd > fenceStart) {
                val inside = s.substring(fenceStart + 3, fenceEnd).trim()
                val cleaned = inside.removePrefix("json").removePrefix("JSON").trim()
                val arr = extractBalancedArray(cleaned)
                if (arr != null) return arr
            }
        }

        // 2) pure array or array with leading/trailing noise
        extractBalancedArray(s)?.let { return it }

        // 3) give up
        return null
    }

    private fun extractBalancedArray(text: String): String? {
        val start = text.indexOf('[')
        if (start < 0) return null
        var depth = 0
        var inString = false
        var esc = false
        for (i in start until text.length) {
            val c = text[i]
            if (inString) {
                if (esc) { esc = false }
                else if (c == '\\') { esc = true }
                else if (c == '"') { inString = false }
                continue
            } else {
                when (c) {
                    '"' -> inString = true
                    '[' -> depth++
                    ']' -> {
                        depth--
                        if (depth == 0) {
                            val candidate = text.substring(start, i + 1)
                            return try {
                                org.json.JSONArray(candidate) // validate
                                candidate
                            } catch (_: Exception) { null }
                        }
                    }
                }
            }
        }
        return null
    }

    override fun onDestroy() {
        super.onDestroy()
        ros?.close()
        ui.cancel()
    }
}
