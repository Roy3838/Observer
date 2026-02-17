package com.plugin.screencapture

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.util.Log
import androidx.activity.result.ActivityResult
import app.tauri.annotation.ActivityCallback
import app.tauri.annotation.Command
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin

private const val TAG = "ScreenCapturePlugin"

@TauriPlugin
class ScreenCapturePlugin(private val activity: Activity) : Plugin(activity) {

    companion object {
        // Store projection data in companion object - accessible by service in same process
        @Volatile
        var mediaProjectionResultCode: Int = Activity.RESULT_CANCELED
        @Volatile
        var mediaProjectionData: Intent? = null
        @Volatile
        var isDataReady: Boolean = false
    }

    @Command
    fun startCapture(invoke: Invoke) {
        Log.d(TAG, "startCapture called")

        activity.runOnUiThread {
            try {
                // Request media projection permission
                val mediaProjectionManager = activity.getSystemService(
                    Context.MEDIA_PROJECTION_SERVICE
                ) as MediaProjectionManager

                val captureIntent = mediaProjectionManager.createScreenCaptureIntent()

                // Start activity for result using Tauri's API
                startActivityForResult(invoke, captureIntent, "onMediaProjectionResult")
                Log.d(TAG, "MediaProjection permission dialog shown")

            } catch (e: Exception) {
                Log.e(TAG, "Error starting capture: ${e.message}", e)
                invoke.reject("Failed to start capture: ${e.message}")
            }
        }
    }

    @Command
    fun stopCapture(invoke: Invoke) {
        Log.d(TAG, "stopCapture called")

        try {
            // Stop the foreground service
            val stopIntent = Intent(activity, ScreenCaptureService::class.java).apply {
                action = ScreenCaptureService.ACTION_STOP
            }
            activity.stopService(stopIntent)

            // Clear stored projection data
            isDataReady = false
            mediaProjectionResultCode = Activity.RESULT_CANCELED
            mediaProjectionData = null

            Log.d(TAG, "Screen capture stopped")
            invoke.resolve()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping capture: ${e.message}", e)
            invoke.reject("Failed to stop capture: ${e.message}")
        }
    }

    @Command
    fun getFrame(invoke: Invoke) {
        // Deprecated - frontend uses get_broadcast_status instead
        invoke.reject("Use get_broadcast_status Tauri command instead")
    }

    @ActivityCallback
    private fun onMediaProjectionResult(invoke: Invoke, result: ActivityResult) {
        Log.d(TAG, "MediaProjection result: resultCode=${result.resultCode}, data=${result.data}")

        if (result.resultCode == Activity.RESULT_OK && result.data != null) {
            // Store the result in companion object for the service to read
            mediaProjectionResultCode = result.resultCode
            mediaProjectionData = result.data
            isDataReady = true

            Log.d(TAG, "Stored projection data: resultCode=$mediaProjectionResultCode, data=$mediaProjectionData")

            // Start the foreground service
            startForegroundService()

            // Return true as a JSON value (Rust expects bool)
            invoke.resolve(JSObject().put("value", true))
        } else {
            Log.w(TAG, "MediaProjection permission denied or cancelled")
            invoke.reject("Screen capture permission denied")
        }
    }

    private fun startForegroundService() {
        val serviceIntent = Intent(activity, ScreenCaptureService::class.java).apply {
            action = ScreenCaptureService.ACTION_START
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            activity.startForegroundService(serviceIntent)
        } else {
            activity.startService(serviceIntent)
        }
        Log.d(TAG, "Foreground service started")
    }
}
