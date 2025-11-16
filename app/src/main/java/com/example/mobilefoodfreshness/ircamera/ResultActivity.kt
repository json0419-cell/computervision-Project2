package com.example.mobilefoodfreshness.ircamera

import android.net.Uri
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.mobilefoodfreshness.R

data class FoodItem(
    val name: String?,
    val edible: Boolean?
)


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


}
