package com.plugin.devicelock

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import android.util.Log
import android.webkit.WebView
import androidx.activity.result.ActivityResult
import app.tauri.annotation.ActivityCallback
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.Plugin

private const val TAG = "DeviceLockPlugin"

@InvokeArg
class LockArgs {
    var reason: String = "Observer locked your phone"
    var maxMinutes: Int = 30
    var allowEmergencyUnlock: Boolean = true
}

@TauriPlugin
class DeviceLockPlugin(private val activity: Activity) : Plugin(activity) {

    override fun load(webView: WebView) {
        super.load(webView)
        // Re-show an unexpired lock if the app was restarted mid-lock,
        // or record a timeout if it expired while we were dead.
        activity.runOnUiThread { DeviceLockController.restoreIfNeeded(activity.applicationContext) }
    }

    @Command
    fun getStatus(invoke: Invoke) {
        invoke.resolve(DeviceLockController.status(activity))
    }

    @Command
    fun requestPermission(invoke: Invoke) {
        if (DeviceLockController.canDrawOverlays(activity)) {
            invoke.resolve(DeviceLockController.status(activity))
            return
        }
        val intent = Intent(
            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
            Uri.parse("package:${activity.packageName}")
        )
        startActivityForResult(invoke, intent, "onOverlayPermissionResult")
    }

    @ActivityCallback
    private fun onOverlayPermissionResult(invoke: Invoke, result: ActivityResult) {
        // The settings screen always returns RESULT_CANCELED; check the real state.
        invoke.resolve(DeviceLockController.status(activity))
    }

    @Command
    fun lock(invoke: Invoke) {
        val args = invoke.parseArgs(LockArgs::class.java)
        if (!DeviceLockController.canDrawOverlays(activity)) {
            invoke.reject("\"Display over other apps\" permission not granted. Call requestPermission() first.")
            return
        }
        activity.runOnUiThread {
            try {
                DeviceLockController.lock(
                    activity.applicationContext,
                    args.reason,
                    args.maxMinutes,
                    args.allowEmergencyUnlock
                )
                invoke.resolve(DeviceLockController.status(activity))
            } catch (e: Exception) {
                Log.e(TAG, "lock failed", e)
                DeviceLockController.unlock(activity.applicationContext, DeviceLockController.CAUSE_MANUAL)
                invoke.reject("Failed to lock: ${e.message}")
            }
        }
    }

    @Command
    fun unlock(invoke: Invoke) {
        activity.runOnUiThread {
            DeviceLockController.unlock(activity.applicationContext, DeviceLockController.CAUSE_MANUAL)
            invoke.resolve(DeviceLockController.status(activity))
        }
    }
}
