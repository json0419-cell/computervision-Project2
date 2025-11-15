package com.example.mobilefoodfreshness.ircamera

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.CheckedTextView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mobilefoodfreshness.R

class FoodAdapter(private val data: List<FoodItem>) :
    RecyclerView.Adapter<FoodAdapter.VH>() {

    class VH(view: View) : RecyclerView.ViewHolder(view) {
        val name: TextView = view.findViewById(R.id.itemName)
        val edible: CheckedTextView = view.findViewById(R.id.itemEdible)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val v = LayoutInflater.from(parent.context)
            .inflate(R.layout.row_food_item, parent, false)
        return VH(v)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val item = data[position]
        holder.name.text = item.name
        holder.edible.isChecked = item.edible
        holder.edible.text = if (item.edible) "Edible" else "Not edible"
    }

    override fun getItemCount(): Int = data.size
}
