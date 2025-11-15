package com.example.mobilefoodfreshness.ircamera

import android.util.Base64
import okhttp3.*
import okio.ByteString
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class RosBridgeClient(
    private val wsUrl: String,                 // ws://<ip>:<port>
    private val topic: String,                 // /bot5/camera_node/image/compressed
    private val onImage: (bytes: ByteArray, format: String?) -> Unit,
    private val onState: (connected: Boolean, error: String?) -> Unit = {_,_->}
) : WebSocketListener() {

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var socket: WebSocket? = null

    fun connect() {
        socket = client.newWebSocket(Request.Builder().url(wsUrl).build(), this)
    }

    fun close() {
        socket?.close(1000, "bye")
    }

    override fun onOpen(webSocket: WebSocket, response: Response) {
        onState(true, null)
        val sub = JSONObject().apply {
            put("op", "subscribe")
            put("topic", topic)
            put("type", "sensor_msgs/CompressedImage")
            put("queue_length", 1)
        }
        webSocket.send(sub.toString())
    }

    override fun onFailure(webSocket: WebSocket, t: Throwable, r: Response?) {
        onState(false, t.message)
    }

    override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
        onState(false, "closed: $reason")
    }

    override fun onMessage(webSocket: WebSocket, text: String) {
        try {
            val root = JSONObject(text)
            if (root.optString("op") != "publish") return
            val msg = root.optJSONObject("msg") ?: return
            val format = msg.optString("format", null)
            val dataB64 = msg.optString("data", "")
            if (dataB64.isNotEmpty()) {
                val bytes = Base64.decode(dataB64, Base64.DEFAULT)
                onImage(bytes, format)
            }
        } catch (_: Exception) { /* ignore single frame error */ }
    }

    override fun onMessage(webSocket: WebSocket, bytes: ByteString) { /* rarely used */ }
}
