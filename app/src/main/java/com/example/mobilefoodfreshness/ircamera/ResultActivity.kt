package com.example.mobilefoodfreshness.ircamera

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.mobilefoodfreshness.R
import com.example.mobilefoodfreshness.ui.ResultActivity.Companion.bitmapCache
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import okhttp3.Call
import okhttp3.Callback
import okhttp3.Response
import java.io.ByteArrayOutputStream
import java.io.IOException

data class FoodItem(
    val name: String?,
    val edible: Boolean?
)

private val ui = CoroutineScope(SupervisorJob() + Dispatchers.Main)

private var bitmapCacheParameter: Bitmap? = null

private val apiUrl = "https://vertex-proxy-646427707803.us-central1.run.app/infer"

private val prompt =
    "Extract all food items visible in the image and return a JSON array of {name, edible}. Output JSON only."


class ResultActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        val title = findViewById<TextView>(R.id.title)
        val image = findViewById<ImageView>(R.id.capturedImage)
        val list = findViewById<RecyclerView>(R.id.resultList)

        // image
        intent.getStringExtra("image_uri")?.let { uriStr ->
            image.setImageURI(Uri.parse(uriStr))
        }

        // parse items
        val raw = intent.getStringExtra("result_json").orEmpty()
        val items = parseItems(raw)
        title.text = "Detected items: ${items.size}"

        list.layoutManager = LinearLayoutManager(this)
        list.addItemDecoration(DividerItemDecoration(this, DividerItemDecoration.VERTICAL))
        list.adapter = FoodAdapter(items)

        val btnRefresh: Button = findViewById(R.id.refreshresults)
        btnRefresh.setOnClickListener {
            resend()
            Toast.makeText(this, "refresh initiated", Toast.LENGTH_SHORT).show()
        }
    }

    private fun parseItems(raw: String): List<FoodItem> {
        val out = mutableListOf<FoodItem>()
        try {
            val arr = org.json.JSONArray(raw.trim())
            for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue
                val name = if (o.has("name")) o.optString("name", null) else null
                val edible = if (o.has("edible")) o.optBoolean("edible") else null
                out.add(FoodItem(name, edible))
            }
        } catch (_: Exception) { }
        return out
    }

    private fun resend() {

        ui.launch(Dispatchers.IO) {
            val jpegBytes = ByteArrayOutputStream().use { bos ->
                bitmapCache?.compress(Bitmap.CompressFormat.JPEG, 90, bos)
                bos.toByteArray()
            }

            val req = ApiClient.buildRequest(apiUrl, jpegBytes, prompt)
            ApiClient.http.newCall(req).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    showResultDialog("HTTP 500", "Failed!")
                }

                override fun onResponse(call: Call, response: Response) {
                    val bodyStr = response.body?.string().orEmpty()
                    ui.launch {

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

                        val title = findViewById<TextView>(R.id.title)
                        val list = findViewById<RecyclerView>(R.id.resultList)

                        val items = parseItems(cleanJson)
                        title.text = "Detected items: ${items.size}"

                        list.layoutManager = LinearLayoutManager(this@ResultActivity)
                        list.addItemDecoration(DividerItemDecoration(this@ResultActivity, DividerItemDecoration.VERTICAL))
                        list.adapter = FoodAdapter(items)
                    }
                }
            })
        }
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

    private fun showResultDialog(title: String, message: String) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }

}
