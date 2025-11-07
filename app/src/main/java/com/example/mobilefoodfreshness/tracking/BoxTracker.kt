package com.example.mobilefoodfreshness.tracking

import android.graphics.Rect
import com.example.mobilefoodfreshness.util.Detection
import kotlin.math.max
import kotlin.math.min

class BoxTracker {
    private var nextId = 1
    private val tracks = mutableMapOf<Int, Detection>()
    fun update(dets: List<Detection>): List<Detection> {
        val results = mutableListOf<Detection>()
        for (d in dets) {
            val id = nextId++
            results.add(d.copy(trackId = id))
            tracks[id] = d
        }
        return results
    }
}

