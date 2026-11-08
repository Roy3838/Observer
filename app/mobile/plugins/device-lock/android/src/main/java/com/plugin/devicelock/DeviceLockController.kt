package com.plugin.devicelock

import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import androidx.core.content.ContextCompat
import app.tauri.plugin.JSObject

private const val TAG = "DeviceLockController"

/**
 * Owns the lock: persisted state, the overlay window and the expiry timer.
 *
 * The overlay is added straight to the WindowManager (not by the service) so
 * it is visible before the foreground service starts. On Android 14+ that
 * visible overlay is what lets a SYSTEM_ALERT_WINDOW app start a foreground
 * service from the background.
 *
 * Safety: the overlay lives in our process, so if the process dies the lock
 * disappears with it. The service only exists to keep the process alive and
 * to re-show an unexpired lock after a sticky restart.
 *
 * All methods must be called on the main thread.
 */
object DeviceLockController {
    const val CAUSE_MANUAL = "manual"
    const val CAUSE_TIMEOUT = "timeout"
    const val CAUSE_EMERGENCY = "emergency"

    private const val PREFS = "observer_device_lock"
    private const val KEY_LOCKED_UNTIL = "lockedUntil"
    private const val KEY_REASON = "reason"
    private const val KEY_ALLOW_EMERGENCY = "allowEmergency"
    private const val KEY_LAST_UNLOCK = "lastUnlock"

    private val handler = Handler(Looper.getMainLooper())
    private var overlay: LockOverlayView? = null
    private var appContext: Context? = null
    private val expiryRunnable = Runnable { appContext?.let { unlock(it, CAUSE_TIMEOUT) } }

    fun canDrawOverlays(context: Context): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.M || Settings.canDrawOverlays(context)

    fun lockedUntil(context: Context): Long =
        prefs(context).getLong(KEY_LOCKED_UNTIL, 0L)

    fun isLocked(context: Context): Boolean =
        lockedUntil(context) > System.currentTimeMillis()

    fun lock(context: Context, reason: String, maxMinutes: Int, allowEmergencyUnlock: Boolean) {
        val app = context.applicationContext
        val until = System.currentTimeMillis() + maxMinutes.coerceIn(1, 720) * 60_000L
        prefs(app).edit()
            .putLong(KEY_LOCKED_UNTIL, until)
            .putString(KEY_REASON, reason)
            .putBoolean(KEY_ALLOW_EMERGENCY, allowEmergencyUnlock)
            .apply()
        show(app)
        startService(app)
    }

    fun unlock(context: Context, cause: String) {
        val app = context.applicationContext
        val wasLocked = prefs(app).getLong(KEY_LOCKED_UNTIL, 0L) != 0L
        handler.removeCallbacks(expiryRunnable)
        removeOverlay(app)
        val editor = prefs(app).edit().remove(KEY_LOCKED_UNTIL)
        if (wasLocked) editor.putString(KEY_LAST_UNLOCK, cause)
        editor.apply()
        app.stopService(Intent(app, DeviceLockService::class.java))
        Log.i(TAG, "Unlocked ($cause)")
    }

    /** Re-show an unexpired lock, or finish an expired one. Returns true if still locked. */
    fun restoreIfNeeded(context: Context): Boolean {
        val app = context.applicationContext
        val until = lockedUntil(app)
        if (until == 0L) return false
        if (until <= System.currentTimeMillis() || !canDrawOverlays(app)) {
            unlock(app, CAUSE_TIMEOUT)
            return false
        }
        show(app)
        return true
    }

    fun status(context: Context): JSObject {
        val until = lockedUntil(context)
        val locked = until > System.currentTimeMillis()
        val obj = JSObject()
        obj.put("supported", true)
        obj.put("authorized", canDrawOverlays(context))
        obj.put("locked", locked)
        if (locked) obj.put("lockedUntil", until)
        prefs(context).getString(KEY_LAST_UNLOCK, null)?.let { obj.put("lastUnlock", it) }
        return obj
    }

    private fun show(app: Context) {
        appContext = app
        val p = prefs(app)
        val until = p.getLong(KEY_LOCKED_UNTIL, 0L)
        val reason = p.getString(KEY_REASON, null) ?: "Observer locked your phone"
        val allowEmergency = p.getBoolean(KEY_ALLOW_EMERGENCY, true)

        val view = overlay ?: LockOverlayView(app) { unlock(app, CAUSE_EMERGENCY) }.also {
            windowManager(app).addView(it, layoutParams())
            overlay = it
        }
        view.bind(reason, until, allowEmergency)

        handler.removeCallbacks(expiryRunnable)
        handler.postDelayed(expiryRunnable, (until - System.currentTimeMillis()).coerceAtLeast(0L))
    }

    private fun removeOverlay(app: Context) {
        overlay?.let {
            it.stop()
            try {
                windowManager(app).removeView(it)
            } catch (e: IllegalArgumentException) {
                Log.w(TAG, "Overlay already detached", e)
            }
        }
        overlay = null
    }

    private fun startService(app: Context) {
        try {
            ContextCompat.startForegroundService(app, Intent(app, DeviceLockService::class.java))
        } catch (e: Exception) {
            // e.g. ForegroundServiceStartNotAllowedException. The overlay still
            // works while the process is alive; it's just easier for the OS to kill.
            Log.w(TAG, "Could not start foreground service; lock is overlay-only", e)
        }
    }

    @Suppress("DEPRECATION")
    private fun layoutParams(): WindowManager.LayoutParams {
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            WindowManager.LayoutParams.TYPE_PHONE
        }
        return WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            type,
            // Focusable on purpose: the overlay swallows Back and all touches.
            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                layoutInDisplayCutoutMode =
                    WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
            }
            systemUiVisibility = View.SYSTEM_UI_FLAG_LAYOUT_STABLE
        }
    }

    private fun windowManager(context: Context) =
        context.getSystemService(Context.WINDOW_SERVICE) as WindowManager

    private fun prefs(context: Context) =
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
}
