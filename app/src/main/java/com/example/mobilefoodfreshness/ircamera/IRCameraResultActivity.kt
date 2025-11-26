package com.example.mobilefoodfreshness.ircamera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.mobilefoodfreshness.R
import com.google.android.material.progressindicator.CircularProgressIndicator
import kotlinx.coroutines.*
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

class IRCameraResultActivity : AppCompatActivity() {

    private val ui = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private lateinit var image: BoundingBoxImageView
    private lateinit var title: TextView
    private lateinit var list: RecyclerView
    private lateinit var btnRefresh: Button
    private lateinit var btnBack: Button

    // loading overlay
    private lateinit var loadingOverlay: FrameLayout
    private lateinit var centerLoading: CircularProgressIndicator

    private val apiUrl = "http://149.165.159.197:8080/infer"

    // Entry points: either IR camera or local camera flow
    // 1) image_uri (preferred)
    // 2) jpeg_bytes (legacy compatibility path)
    private var imageUri: Uri? = null
    private var originalJpeg: ByteArray? = null

    private var origImageWidth: Int = 0
    private var origImageHeight: Int = 0
    private var imageRotated90: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        image = findViewById(R.id.capturedImage)
        title = findViewById(R.id.title)
        list = findViewById(R.id.resultList)
        btnRefresh = findViewById(R.id.refreshresults)
        btnBack = findViewById(R.id.back)

        loadingOverlay = findViewById(R.id.resultLoadingOverlay)
        centerLoading = findViewById(R.id.resultCenterLoading)

        val rawJson = intent.getStringExtra("raw_json").orEmpty()
        originalJpeg = intent.getByteArrayExtra("jpeg_bytes")

        // Prefer showing the original image via Uri
        val uriStr = intent.getStringExtra("image_uri")
        if (uriStr != null) {
            val uri = Uri.parse(uriStr)
            imageUri = uri // (also fixes resend() using imageUri)

            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            // store original (unrotated) size
            origImageWidth = bitmap.width
            origImageHeight = bitmap.height

            val matrix = Matrix()
            matrix.postRotate(90f) // or any desired angle
            val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            bitmap.recycle() // Release the original bitmap

            image.setImageBitmap(rotatedBitmap)
            imageRotated90 = true
        } else {
            // Fallback when Uri is unavailable: decode from jpeg_bytes
            originalJpeg?.let { bytes ->
                val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

                // store original size (no rotation in this path)
                origImageWidth = bmp.width
                origImageHeight = bmp.height

                image.setImageBitmap(bmp)
                imageRotated90 = false
            }
        }

        // ① items -> FoodItem(name, score)
        val items = parseItems(rawJson)
        title.text = "Detected items: ${items.size}"
        list.layoutManager = LinearLayoutManager(this)
        list.addItemDecoration(
            DividerItemDecoration(this, DividerItemDecoration.VERTICAL)
        )
        list.adapter = FoodAdapter(items)

        // ② boxes_xyxy + labels + per_item_status -> BoundingBox (filter placeholders)
        val boxes = parseBoundingBoxes(rawJson)
        image.setBoxes(boxes)

        btnBack.setOnClickListener {
            finish()
        }

        btnRefresh.setOnClickListener {
            resend()
            Toast.makeText(this, "Refreshing…", Toast.LENGTH_SHORT).show()
        }
    }

    // ----------------- Parse items (core rule: reuse edible/ediblt to derive score) -----------------
    private fun parseItems(raw: String): List<FoodItem> {
        val out = mutableListOf<FoodItem>()
        try {
            val obj = JSONObject(raw)
            val arr = obj.optJSONArray("items") ?: JSONArray()

            for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue

                val name = o.optString("name", null)
                    ?.takeIf { it.isNotBlank() }

                val score: Float? = when {
                    o.has("score") && !o.isNull("score") -> {
                        o.optDouble("score").toFloat()
                    }
                    o.has("edible") && !o.isNull("edible") -> {
                        val v = o.get("edible")
                        when (v) {
                            is Number -> v.toFloat()
                            is Boolean -> if (v) 8.5f else 2.0f
                            is String -> v.toFloatOrNull()
                            else -> null
                        }
                    }
                    o.has("ediblt") && !o.isNull("ediblt") -> {
                        val v = o.get("ediblt")
                        when (v) {
                            is Number -> v.toFloat()
                            is Boolean -> if (v) 8.5f else 2.0f
                            is String -> v.toFloatOrNull()
                            else -> null
                        }
                    }
                    else -> null
                }

                out.add(FoodItem(name, score))
            }
        } catch (_: Exception) {
        }
        return out
    }

    private fun buildRotationMatrixForBoxes(): Matrix? {
        if (!imageRotated90 || origImageWidth == 0 || origImageHeight == 0) {
            return null
        }

        val m = Matrix()
        // Same as in onCreate: postRotate(90f)
        m.postRotate(90f)

        // Mimic what Bitmap.createBitmap does: translate so the image fits at (0,0)
        val src = RectF(0f, 0f, origImageWidth.toFloat(), origImageHeight.toFloat())
        val dst = RectF()
        m.mapRect(dst, src)
        m.postTranslate(-dst.left, -dst.top)

        return m
    }

    // ----------------- Parse boxes_xyxy, apply Python placeholder filtering, include className + score -----------------
    private fun parseBoundingBoxes(raw: String): List<BoundingBox> {
        val out = mutableListOf<BoundingBox>()
        try {
            val obj = JSONObject(raw)

            val boxesArr = obj.optJSONArray("boxes_xyxy") ?: return out
            val labelsArr = obj.optJSONArray("labels") ?: JSONArray()
            val statusArr = obj.optJSONArray("per_item_status") ?: JSONArray()
            val itemsArr = obj.optJSONArray("items") ?: JSONArray()

            // Collect placeholder names from per_item_status
            val placeholderNames = mutableSetOf<String>()
            for (i in 0 until statusArr.length()) {
                val s = statusArr.optJSONObject(i) ?: continue
                if (s.optString("found_by") == "placeholder") {
                    val nm = s.optString("name", null)
                    if (!nm.isNullOrBlank()) placeholderNames.add(nm)
                }
            }

            // Build rotation matrix (if the displayed image is rotated 90°)
            val rotationMatrix = buildRotationMatrixForBoxes()

            for (i in 0 until boxesArr.length()) {
                val boxArr = boxesArr.optJSONArray(i) ?: continue
                if (boxArr.length() < 4) continue

                val label = labelsArr.optString(i, "")

                // Parse score from label string
                val parsedScore = parseScoreFromLabel(label)

                // Rule 1: score <= 0.02 => placeholder
                val isPlaceholderByScore =
                    (!parsedScore.isNaN()) && (parsedScore <= 0.02f)

                // Rule 2: label text contains any placeholder name
                var isPlaceholderByName = false
                val normLab = label.lowercase()
                for (nm in placeholderNames) {
                    if (nm.isNotBlank() && normLab.contains(nm.lowercase())) {
                        isPlaceholderByName = true
                        break
                    }
                }

                if (isPlaceholderByScore || isPlaceholderByName) {
                    continue   // Skip placeholder boxes
                }

                val x1 = boxArr.optDouble(0).toFloat()
                val y1 = boxArr.optDouble(1).toFloat()
                val x2 = boxArr.optDouble(2).toFloat()
                val y2 = boxArr.optDouble(3).toFloat()

                // Rectangle in original image coordinates
                val rect = RectF(x1, y1, x2, y2)

                // ⭐ Rotate rect to match the rotated bitmap, if needed
                rotationMatrix?.mapRect(rect)

                // Split into className + score for drawing
                val (classNameRaw, scoreForDraw) = splitLabelAndScore(label)
                val className = classNameRaw.replace('_', ' ')  // ⭐ Replace _ with spaces for display

                val s = itemsArr.optJSONObject(i) ?: continue
                val freshnessScore = "freshness ${s.optString("edible", "-1")}"

                out.add(
                    BoundingBox(
                        rect = rect,
                        className = className,
                        score = scoreForDraw,
                        freshnessScore = freshnessScore,
                        rawLabel = label
                    )
                )
            }
        } catch (_: Exception) {
        }
        return out
    }

    private fun parseScoreFromLabel(label: String): Float {
        val regex = Regex("([-+]?\\d*\\.\\d+|\\d+)$")
        val m = regex.find(label.trim()) ?: return Float.NaN
        return m.groupValues.getOrNull(1)?.toFloatOrNull() ?: Float.NaN
    }

    private fun splitLabelAndScore(label: String): Pair<String, Float?> {
        val trimmed = label.trim()
        if (trimmed.isEmpty()) return "" to null

        val regex = Regex("([-+]?\\d*\\.\\d+|\\d+)$")
        val m = regex.find(trimmed)

        val score = m?.groupValues?.getOrNull(1)?.toFloatOrNull()
        val namePart = if (m != null) {
            trimmed.substring(0, m.range.first).trim()
        } else {
            trimmed
        }

        val className = if (namePart.isEmpty()) "unknown" else namePart
        return className to score
    }

    // ----------------- Refresh: read JPEG from imageUri first, otherwise fall back to originalJpeg -----------------
    private fun resend() {
        // 1) If available, read bytes from the Uri
        val jpegBytes: ByteArray? = imageUri?.let { uri ->
            try {
                contentResolver.openInputStream(uri)?.use { it.readBytes() }
            } catch (e: Exception) {
                null
            }
        } ?: originalJpeg   // 2) Otherwise fall back to the legacy byte[]

        if (jpegBytes == null) {
            Toast.makeText(this, "No JPEG to resend", Toast.LENGTH_SHORT).show()
            return
        }

        loadingOverlay.visibility = View.VISIBLE
        centerLoading.show()
        btnRefresh.isEnabled = false

        ui.launch(Dispatchers.IO) {
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
                        centerLoading.hide()
                        btnRefresh.isEnabled = true

                        showResultDialog("HTTP 500", "Refresh failed: ${e.message}")
                    }
                }

                override fun onResponse(call: Call, response: Response) {
                    val bodyStr = response.body?.string().orEmpty()
                    ui.launch {
                        loadingOverlay.visibility = View.GONE
                        centerLoading.hide()
                        btnRefresh.isEnabled = true

                        if (!response.isSuccessful) {
                            showResultDialog("HTTP ${response.code}", bodyStr.take(1000))
                            return@launch
                        }

                        val items = parseItems(bodyStr)
                        title.text = "Detected items: ${items.size}"
                        list.adapter = FoodAdapter(items)

                        val boxes = parseBoundingBoxes(bodyStr)
                        image.setBoxes(boxes)
                    }
                }
            })
        }
    }

    private fun showResultDialog(title: String, message: String) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        ui.cancel()
    }
}
