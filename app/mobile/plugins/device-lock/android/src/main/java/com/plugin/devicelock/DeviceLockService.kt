package com.plugin.devicelock

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import java.text.DateFormat
import java.util.Date

/**
 * Keeps the process (and therefore the overlay) alive while locked. On a
 * sticky restart after the OS killed us, re-shows the lock if it hasn't
 * expired yet, otherwise stops.
 */
class DeviceLockService : Service() {

    companion object {
        private const val CHANNEL_ID = "observer_device_lock"
        private const val NOTIFICATION_ID = 0x0B5E
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // startForeground must be called promptly after startForegroundService.
        try {
            ServiceCompat.startForeground(
                this,
                NOTIFICATION_ID,
                buildNotification(),
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
                } else {
                    0
                }
            )
        } catch (e: Exception) {
            // Background FGS start denied (e.g. sticky restart on Android 12+).
            // The overlay alone still works while the process lives.
            android.util.Log.w("DeviceLockService", "startForeground failed", e)
            stopSelf()
            return START_NOT_STICKY
        }

        Handler(Looper.getMainLooper()).post {
            if (!DeviceLockController.restoreIfNeeded(this)) stopSelf()
        }
        return START_STICKY
    }

    private fun buildNotification(): android.app.Notification {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(NotificationManager::class.java)
            if (nm.getNotificationChannel(CHANNEL_ID) == null) {
                nm.createNotificationChannel(
                    NotificationChannel(CHANNEL_ID, "Phone lock", NotificationManager.IMPORTANCE_LOW)
                )
            }
        }

        val until = DeviceLockController.lockedUntil(this)
        val text = if (until > 0) {
            "Unlocks at " + DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(until))
        } else {
            "Locked"
        }

        val launch = packageManager.getLaunchIntentForPackage(packageName)?.let {
            PendingIntent.getActivity(this, 0, it, PendingIntent.FLAG_IMMUTABLE)
        }

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_lock_lock)
            .setContentTitle("Phone locked by Observer")
            .setContentText(text)
            .setOngoing(true)
            .setContentIntent(launch)
            .build()
    }
}
