package com.example.mobilefoodfreshness.ircamera

import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import java.util.concurrent.TimeUnit

object ApiClient {
    val http: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    fun buildRequest(url: String, jpegBytes: ByteArray, prompt: String): Request {
        val imageBody = RequestBody.create("image/jpeg".toMediaTypeOrNull(), jpegBytes)
        val form = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", "capture.jpg", imageBody)
            .addFormDataPart("prompt", prompt)
            .build()

        return Request.Builder()
            .url(url)
            .addHeader("Accept", "application/json")
            .post(form)
            .build()
    }
}
