package com.example.mobilefoodfreshness.ircamera

import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mobilefoodfreshness.R

data class FoodItem(
    val name: String?,
    val score: Float?    // 0–10 freshness score from backend
)

class FoodAdapter(private val data: List<FoodItem>) :
    RecyclerView.Adapter<FoodAdapter.VH>() {

    class VH(v: View) : RecyclerView.ViewHolder(v) {
        val name: TextView = v.findViewById(R.id.itemName)
        val scoreLabel: TextView = v.findViewById(R.id.itemScoreLabel)
        val bar1: View = v.findViewById(R.id.bar1)
        val bar2: View = v.findViewById(R.id.bar2)
        val bar3: View = v.findViewById(R.id.bar3)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val v = LayoutInflater.from(parent.context)
            .inflate(R.layout.row_food_item, parent, false)
        return VH(v)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val item = data[position]

        // ⭐ Replace "_" with spaces only for display purposes
        val displayName = item.name
            ?.replace("_", " ")
            ?.takeIf { it.isNotBlank() }
            .orEmpty()
        holder.name.text = displayName

        // Reset all bars to the inactive gray state by default
        val inactive = Color.parseColor("#555555")
        holder.bar1.setBackgroundColor(inactive)
        holder.bar2.setBackgroundColor(inactive)
        holder.bar3.setBackgroundColor(inactive)

        // Three color bands representing freshness levels
        val red = Color.parseColor("#D32F2F")
        val yellow = Color.parseColor("#FBC02D")
        val green = Color.parseColor("#2E7D32")

        val score = item.score
        if (score == null || score.isNaN()) {
            holder.scoreLabel.text = "Unknown"
            return
        }

        when {
            score <= 4f -> {
                // spoiled / inedible
                holder.scoreLabel.text = "Spoiled"
                holder.bar1.setBackgroundColor(red)
            }
            score <= 7f -> {
                // okay
                holder.scoreLabel.text = "Okay"
                holder.bar1.setBackgroundColor(yellow)
                holder.bar2.setBackgroundColor(yellow)
            }
            else -> {
                // fresh (8–10)
                holder.scoreLabel.text = "Fresh"
                holder.bar1.setBackgroundColor(green)
                holder.bar2.setBackgroundColor(green)
                holder.bar3.setBackgroundColor(green)
            }
        }
    }

    override fun getItemCount(): Int = data.size
}
