package com.example.mobilefoodfreshness.ircamera

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.calib3d.Calib3d

/**
 * Fisheye distortion correction utility
 * Uses OpenCV fisheye::undistortImage to remove fisheye distortion
 */
object FisheyeCorrection {
    
    // Default camera matrix (K) - these should be calibrated for your specific camera
    // Format: [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    // These are example values - you should replace them with your camera's calibration parameters
    private val defaultCameraMatrix: Mat = Mat(3, 3, CvType.CV_64FC1).apply {
        // Row 0: [fx, 0, cx]
        put(0, 0, 500.0, 0.0, 320.0)
        // Row 1: [0, fy, cy]
        put(1, 0, 0.0, 500.0, 240.0)
        // Row 2: [0, 0, 1]
        put(2, 0, 0.0, 0.0, 1.0)
    }
    
    // Default distortion coefficients (D) for fisheye model
    // Format: [k1, k2, k3, k4] for fisheye model
    // These are example values - you should replace them with your camera's calibration parameters
    private val defaultDistCoeffs: Mat = Mat(1, 4, CvType.CV_64FC1).apply {
        // [k1, k2, k3, k4]
        put(0, 0, 0.1, 0.05, 0.0, 0.0)
    }
    
    private var cameraMatrix: Mat = defaultCameraMatrix
    private var distCoeffs: Mat = defaultDistCoeffs
    
    /**
     * Set camera calibration parameters
     * @param K Camera matrix (3x3)
     * @param D Distortion coefficients (1x4 for fisheye)
     */
    fun setCalibrationParams(K: Mat, D: Mat) {
        cameraMatrix = K
        distCoeffs = D
    }
    
    /**
     * Undistort a bitmap image to remove fisheye distortion
     * @param input Bitmap with fisheye distortion
     * @return Undistorted bitmap
     */
    fun undistort(input: Bitmap): Bitmap {
        if (input.isRecycled) return input
        
        val srcMat = Mat()
        Utils.bitmapToMat(input, srcMat)
        
        // Use Calib3d.undistort method for distortion correction
        val dstMat = Mat()
        Calib3d.undistort(srcMat, dstMat, cameraMatrix, distCoeffs)
        
        val output = Bitmap.createBitmap(
            dstMat.width(),
            dstMat.height(),
            Bitmap.Config.ARGB_8888
        )
        Utils.matToBitmap(dstMat, output)
        
        srcMat.release()
        dstMat.release()
        
        return output
    }
    
    /**
     * Quick undistort with default parameters (for testing)
     * Adjust the default parameters based on your camera
     */
    fun quickUndistort(input: Bitmap): Bitmap {
        return undistort(input)
    }
}

