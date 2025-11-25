package com.example.mobilefoodfreshness.ircamera

import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mobilefoodfreshness.R

class FoodAdapter(private val data: List<FoodItem>) :
    RecyclerView.Adapter<FoodAdapter.VH>() {

    class VH(v: View) : RecyclerView.ViewHolder(v) {
        val name: TextView = v.findViewById(R.id.itemName)
        val edible: TextView = v.findViewById(R.id.itemEdible)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val v = LayoutInflater.from(parent.context)
            .inflate(R.layout.row_food_item, parent, false)
        return VH(v)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val item = data[position]

        // name
        holder.name.text = item.name?.takeIf { it.isNotBlank() }.orEmpty()
        holder.name.visibility = View.VISIBLE

        // edible
        when (item.edible) {
            true -> {
                holder.edible.visibility = View.VISIBLE
                holder.edible.text = "✅"
                holder.name.setTextColor(Color.parseColor("#0B8043"))
            }
            false -> {
                holder.edible.visibility = View.VISIBLE
                holder.edible.text = "❌"
                holder.name.setTextColor(Color.parseColor("#B00020"))
            }
            else -> {
                holder.edible.text = ""
                holder.edible.visibility = View.INVISIBLE
            }
        }
    }

    override fun getItemCount(): Int = data.size
}
