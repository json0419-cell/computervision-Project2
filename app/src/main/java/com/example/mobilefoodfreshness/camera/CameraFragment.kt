package com.example.mobilefoodfreshness.camera

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.example.mobilefoodfreshness.R
import com.example.mobilefoodfreshness.ui.OverlayView
import com.example.mobilefoodfreshness.ircamera.ApiClient
import com.example.mobilefoodfreshness.ircamera.IRCameraResultActivity
import com.google.android.material.progressindicator.CircularProgressIndicator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment : Fragment() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var btnFinish: Button

    // Static image overlay displayed after tapping Finish (freezes the frame)
    private lateinit var freezeImage: ImageView

    private lateinit var blackOverlayView: View

    // Loading overlay for the local camera preview
    private lateinit var localLoadingOverlay: FrameLayout
    private lateinit var localCenterLoading: CircularProgressIndicator

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var previewUseCase: Preview? = null
    private lateinit var cameraExecutor: ExecutorService

    // The most recent camera frame
    private var lastFrame: Bitmap? = null
    // Flag indicating upload in progress (halts lastFrame updates + keeps frozen preview)
    private var isCaptureInProgress = false

    // Shares the same URL as the IR camera flow
    private val apiUrl = "http://149.165.159.197:8080/infer"

    private val reqPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else {
            Toast.makeText(requireContext(), "Camera permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        s: Bundle?
    ): View {
        val v = inflater.inflate(R.layout.fragment_camera, container, false)
        previewView = v.findViewById(R.id.previewView)
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        overlay = v.findViewById(R.id.overlay)
        btnFinish = v.findViewById(R.id.btnFinish)
        freezeImage = v.findViewById(R.id.freezeImage)
        freezeImage.scaleType = ImageView.ScaleType.FIT_CENTER
        blackOverlayView = v.findViewById(R.id.blackOverlayView)
        blackOverlayView.visibility = View.GONE

        localLoadingOverlay = v.findViewById(R.id.localLoadingOverlay)
        localCenterLoading = v.findViewById(R.id.localCenterLoading)
        localCenterLoading.isIndeterminate = true   // Force indeterminate mode in code

        cameraExecutor = Executors.newSingleThreadExecutor()

        btnFinish.setOnClickListener {
            onCaptureAndSend()
        }

        return v
    }

    override fun onResume() {
        super.onResume()
        // Reset state whenever we return to this screen
        isCaptureInProgress = false
        freezeImage.setImageDrawable(null)
        freezeImage.visibility = View.GONE
        localLoadingOverlay.visibility = View.GONE
        localCenterLoading.hide()
        btnFinish.isEnabled = true
        blackOverlayView.visibility = View.GONE

        reqPermission.launch(Manifest.permission.CAMERA)
    }

    override fun onPause() {
        super.onPause()
        cameraProvider?.unbindAll()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        cameraExecutor.shutdown()
    }

    // ---------------- CameraX initialization ----------------

    private fun startCamera() {
        val ctx = requireContext()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            val provider = cameraProvider ?: return@addListener

            previewUseCase = Preview.Builder()
                .setTargetResolution(Size(1280, 720))
                .build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { img ->
                        // Skip updating lastFrame while uploading (effectively freeze that frame)
                        if (!isCaptureInProgress) {
                            val bmp = img.toBitmap()
                            lastFrame = bmp
                        }
                        img.close()
                    }
                }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    previewUseCase,
                    imageAnalysis
                )
            } catch (e: Exception) {
                e.printStackTrace()
                Toast.makeText(ctx, "Failed to start camera: ${e.message}", Toast.LENGTH_LONG)
                    .show()
            }

        }, ContextCompat.getMainExecutor(ctx))
    }

    // ImageProxy -> Bitmap (keeps original orientation so bounding boxes stay aligned)
    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage =
            android.graphics.YuvImage(
                nv21,
                android.graphics.ImageFormat.NV21,
                width,
                height,
                null
            )
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 90, out)
        val imageBytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    // Downscale a bitmap so the longest edge does not exceed maxSize
    private fun downscaleBitmap(src: Bitmap, maxSize: Int = 1024): Bitmap {
        val w = src.width
        val h = src.height
        val maxSide = maxOf(w, h)
        if (maxSide <= maxSize) return src

        val scale = maxSize.toFloat() / maxSide.toFloat()
        val matrix = Matrix().apply { postScale(scale, scale) }
        return Bitmap.createBitmap(src, 0, 0, w, h, matrix, true)
    }

    // Bitmap -> JPEG bytes
    private fun bitmapToJpegBytes(bmp: Bitmap, quality: Int = 90): ByteArray {
        return ByteArrayOutputStream().use { bos ->
            bmp.compress(Bitmap.CompressFormat.JPEG, quality, bos)
            bos.toByteArray()
        }
    }

    // Save JPEG bytes into cache and return a Uri (for ResultActivity consumption)
    private fun saveJpegToCache(jpegBytes: ByteArray): Uri {
        val ctx = requireContext()
        val file = File(ctx.cacheDir, "local_last_capture.jpg")
        FileOutputStream(file).use { it.write(jpegBytes) }
        return FileProvider.getUriForFile(
            ctx,
            "${ctx.packageName}.fileprovider",
            file
        )
    }

    // Slim down backend JSON, keeping only items / boxes_xyxy / labels / per_item_status
    private fun buildSlimJson(full: String): String {
        return try {
            val obj = JSONObject(full)
            val slim = JSONObject()
            slim.put("items", obj.optJSONArray("items") ?: JSONArray())
            slim.put("boxes_xyxy", obj.optJSONArray("boxes_xyxy") ?: JSONArray())
            slim.put("labels", obj.optJSONArray("labels") ?: JSONArray())
            slim.put("per_item_status", obj.optJSONArray("per_item_status") ?: JSONArray())
            slim.toString()
        } catch (e: Exception) {
            full.take(5000)
        }
    }

    // ---------------- On Finish: capture frame + freeze preview + call server ----------------

    private fun onCaptureAndSend() {
        if (isCaptureInProgress) return

        val frame = lastFrame
        if (frame == null) {
            Toast.makeText(requireContext(), "No frame available", Toast.LENGTH_SHORT).show()
            return
        }

        // Mark upload in progress so the analyzer stops updating lastFrame
        isCaptureInProgress = true

        // Freeze preview by drawing the current frame into freezeImage over the preview
        freezeImage.setImageBitmap(frame)
        freezeImage.rotation = 90f
        freezeImage.visibility = View.VISIBLE
        blackOverlayView.visibility = View.VISIBLE

        // Show loading overlay and disable the button
        btnFinish.isEnabled = false
        localLoadingOverlay.visibility = View.VISIBLE
        localCenterLoading.show()

        // Downscale, compress, and upload
        viewLifecycleOwner.lifecycleScope.launch(Dispatchers.IO) {
            try {
                val downscaled = downscaleBitmap(frame, maxSize = 1024)
                val jpegBytes = bitmapToJpegBytes(downscaled, 90)

                // Build multipart/form-data request body
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
                        viewLifecycleOwner.lifecycleScope.launch(Dispatchers.Main) {
                            isCaptureInProgress = false
                            btnFinish.isEnabled = true
                            localCenterLoading.hide()
                            localLoadingOverlay.visibility = View.GONE
                            // On failure, unfreeze so the user sees the live preview again
                            freezeImage.setImageDrawable(null)
                            freezeImage.visibility = View.GONE

                            Toast.makeText(
                                requireContext(),
                                "Upload failed: ${e.message}",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }

                    override fun onResponse(call: Call, response: Response) {
                        val bodyStr = response.body?.string().orEmpty()
                        viewLifecycleOwner.lifecycleScope.launch(Dispatchers.Main) {
                            isCaptureInProgress = false
                            btnFinish.isEnabled = true
                            localCenterLoading.hide()
                            localLoadingOverlay.visibility = View.GONE

                            if (!response.isSuccessful) {
                                // On HTTP errors also unfreeze so the user can retry
                                freezeImage.setImageDrawable(null)
                                freezeImage.visibility = View.GONE

                                AlertDialog.Builder(requireContext())
                                    .setTitle("HTTP ${response.code}")
                                    .setMessage(bodyStr.take(1000))
                                    .setPositiveButton("OK", null)
                                    .show()
                                return@launch
                            }

                            // Slim JSON to avoid TransactionTooLarge when passing via Intent
                            val safeJson = buildSlimJson(bodyStr)
                            val imageUri = saveJpegToCache(jpegBytes)

                            val intent = Intent(requireContext(), IRCameraResultActivity::class.java)
                            intent.putExtra("raw_json", safeJson)
                            intent.putExtra("image_uri", imageUri.toString())
                            intent.putExtra("needs_rotation",true)
                            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

                            startActivity(intent)
                            // Keep the frozen image until we return; onResume will reset everything
                        }
                    }
                })
            } catch (e: Exception) {
                e.printStackTrace()
                viewLifecycleOwner.lifecycleScope.launch(Dispatchers.Main) {
                    isCaptureInProgress = false
                    btnFinish.isEnabled = true
                    localCenterLoading.hide()
                    localLoadingOverlay.visibility = View.GONE
                    freezeImage.setImageDrawable(null)
                    freezeImage.visibility = View.GONE

                    Toast.makeText(
                        requireContext(),
                        "Capture error: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }
}
