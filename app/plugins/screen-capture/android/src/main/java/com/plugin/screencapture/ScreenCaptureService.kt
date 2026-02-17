package com.plugin.screencapture

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.util.DisplayMetrics
import android.util.Log
import android.view.WindowManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL

private const val TAG = "ScreenCaptureService"
private const val NOTIFICATION_CHANNEL_ID = "screen_capture_channel"
private const val NOTIFICATION_ID = 1001

class ScreenCaptureService : Service() {

    companion object {
        const val ACTION_START = "com.plugin.screencapture.ACTION_START"
        const val ACTION_STOP = "com.plugin.screencapture.ACTION_STOP"
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_DATA = "data"

        private const val FRAMES_URL = "http://127.0.0.1:8080/frames"
        private const val BROADCAST_START_URL = "http://127.0.0.1:8080/broadcast/start"
        private const val BROADCAST_STOP_URL = "http://127.0.0.1:8080/broadcast/stop"
        private const val JPEG_QUALITY = 60 // Match iOS quality (0.6)
    }

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null

    private var screenWidth = 0
    private var screenHeight = 0
    private var screenDensity = 0

    private lateinit var handlerThread: HandlerThread
    private lateinit var handler: Handler
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private var isCapturing = false
    private var lastFrameTime = 0L
    private val minFrameInterval = 33L // ~30 fps max

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")

        // Initialize handler thread for ImageReader callbacks
        handlerThread = HandlerThread("ScreenCaptureThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        // Get screen metrics
        val windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val windowMetrics = windowManager.currentWindowMetrics
            screenWidth = windowMetrics.bounds.width()
            screenHeight = windowMetrics.bounds.height()
        } else {
            @Suppress("DEPRECATION")
            val displayMetrics = DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getMetrics(displayMetrics)
            screenWidth = displayMetrics.widthPixels
            screenHeight = displayMetrics.heightPixels
        }
        screenDensity = resources.displayMetrics.densityDpi

        Log.d(TAG, "Screen dimensions: ${screenWidth}x${screenHeight}, density: $screenDensity")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "onStartCommand: action=${intent?.action}")

        // On Android 14+, we MUST call startForeground immediately
        // Call it first before any other operations
        if (intent?.action == ACTION_START) {
            try {
                startForegroundWithNotification()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start foreground: ${e.message}", e)
                stopSelf()
                return START_NOT_STICKY
            }
        }

        when (intent?.action) {
            ACTION_START -> {
                try {
                    // Read from companion object (same process, more reliable than Intent extras)
                    val resultCode = ScreenCapturePlugin.mediaProjectionResultCode
                    val data = ScreenCapturePlugin.mediaProjectionData
                    val isReady = ScreenCapturePlugin.isDataReady

                    Log.d(TAG, "Reading from companion: isReady=$isReady, resultCode=$resultCode, data=$data")

                    if (isReady && resultCode != android.app.Activity.RESULT_CANCELED && data != null) {
                        startCapture(resultCode, data)
                    } else {
                        Log.e(TAG, "Invalid result code or data: isReady=$isReady, resultCode=$resultCode, data=$data")
                        stopSelf()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error in onStartCommand: ${e.message}", e)
                    stopSelf()
                }
            }
            ACTION_STOP -> {
                stopCapture()
                stopSelf()
            }
        }

        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startForegroundWithNotification() {
        createNotificationChannel()

        val notification = buildNotification()

        // On Android 14+, must specify the foreground service type
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        Log.d(TAG, "Foreground notification started")
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "Screen Capture",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Screen capture is active"
                setShowBadge(false)
            }

            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(): Notification {
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, NOTIFICATION_CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }

        // Try to launch the main app activity when notification is tapped
        val launchIntent = packageManager.getLaunchIntentForPackage(packageName)
        val pendingIntent = if (launchIntent != null) {
            PendingIntent.getActivity(
                this, 0, launchIntent,
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
            )
        } else null

        return builder
            .setContentTitle("Observer Screen Capture")
            .setContentText("Screen capture is active")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setOngoing(true)
            .setContentIntent(pendingIntent)
            .build()
    }

    private fun startCapture(resultCode: Int, data: Intent) {
        Log.d(TAG, "Starting capture with resultCode: $resultCode")

        try {
            val mediaProjectionManager = getSystemService(
                Context.MEDIA_PROJECTION_SERVICE
            ) as MediaProjectionManager

            mediaProjection = mediaProjectionManager.getMediaProjection(resultCode, data)

            if (mediaProjection == null) {
                Log.e(TAG, "Failed to get MediaProjection")
                stopSelf()
                return
            }

            // Register callback for projection stop
            mediaProjection?.registerCallback(mediaProjectionCallback, handler)

            // Create ImageReader
            imageReader = ImageReader.newInstance(
                screenWidth,
                screenHeight,
                PixelFormat.RGBA_8888,
                2 // Max images - keep small to avoid memory issues
            )

            imageReader?.setOnImageAvailableListener(imageAvailableListener, handler)

            // Create VirtualDisplay
            virtualDisplay = mediaProjection?.createVirtualDisplay(
                "ScreenCapture",
                screenWidth,
                screenHeight,
                screenDensity,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader?.surface,
                null,
                handler
            )

            isCapturing = true

            // Notify server that broadcast started
            notifyBroadcastStart()

            Log.d(TAG, "Screen capture started successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Error starting capture: ${e.message}", e)
            stopCapture()
            stopSelf()
        }
    }

    private val mediaProjectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            Log.d(TAG, "MediaProjection stopped by system")
            stopCapture()
        }
    }

    private val imageAvailableListener = ImageReader.OnImageAvailableListener { reader ->
        if (!isCapturing) return@OnImageAvailableListener

        val currentTime = System.currentTimeMillis()
        if (currentTime - lastFrameTime < minFrameInterval) {
            // Skip frame to maintain frame rate limit
            reader.acquireLatestImage()?.close()
            return@OnImageAvailableListener
        }
        lastFrameTime = currentTime

        var image: Image? = null
        try {
            image = reader.acquireLatestImage()
            if (image == null) return@OnImageAvailableListener

            val jpeg = convertToJpeg(image)
            if (jpeg != null) {
                postFrameToServer(jpeg)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame: ${e.message}")
        } finally {
            image?.close()
        }
    }

    private fun convertToJpeg(image: Image): ByteArray? {
        try {
            val planes = image.planes
            val buffer = planes[0].buffer
            val pixelStride = planes[0].pixelStride
            val rowStride = planes[0].rowStride
            val rowPadding = rowStride - pixelStride * screenWidth

            // Create bitmap with padding
            val bitmapWidth = screenWidth + rowPadding / pixelStride
            val bitmap = Bitmap.createBitmap(
                bitmapWidth,
                screenHeight,
                Bitmap.Config.ARGB_8888
            )
            bitmap.copyPixelsFromBuffer(buffer)

            // Crop to actual screen size if there's padding
            val croppedBitmap = if (rowPadding > 0) {
                Bitmap.createBitmap(bitmap, 0, 0, screenWidth, screenHeight).also {
                    bitmap.recycle()
                }
            } else {
                bitmap
            }

            // Compress to JPEG
            val outputStream = ByteArrayOutputStream()
            croppedBitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, outputStream)
            croppedBitmap.recycle()

            return outputStream.toByteArray()

        } catch (e: Exception) {
            Log.e(TAG, "Error converting to JPEG: ${e.message}", e)
            return null
        }
    }

    private fun postFrameToServer(jpeg: ByteArray) {
        serviceScope.launch {
            try {
                val url = URL(FRAMES_URL)
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", "application/octet-stream")
                connection.connectTimeout = 2000
                connection.readTimeout = 2000

                connection.outputStream.use { it.write(jpeg) }
                connection.responseCode // Trigger request
                connection.disconnect()

            } catch (e: Exception) {
                // Fire-and-forget, ignore errors (matches iOS behavior)
                Log.v(TAG, "Frame post error (ignored): ${e.message}")
            }
        }
    }

    private fun notifyBroadcastStart() {
        serviceScope.launch {
            try {
                val url = URL(BROADCAST_START_URL)
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.connectTimeout = 2000
                connection.responseCode
                connection.disconnect()
                Log.d(TAG, "Notified server: broadcast started")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to notify broadcast start: ${e.message}")
            }
        }
    }

    private fun notifyBroadcastStop() {
        serviceScope.launch {
            try {
                val url = URL(BROADCAST_STOP_URL)
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.connectTimeout = 2000
                connection.responseCode
                connection.disconnect()
                Log.d(TAG, "Notified server: broadcast stopped")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to notify broadcast stop: ${e.message}")
            }
        }
    }

    private fun stopCapture() {
        Log.d(TAG, "Stopping capture")
        isCapturing = false

        // Notify server
        notifyBroadcastStop()

        try {
            virtualDisplay?.release()
            virtualDisplay = null

            imageReader?.close()
            imageReader = null

            mediaProjection?.unregisterCallback(mediaProjectionCallback)
            mediaProjection?.stop()
            mediaProjection = null

            Log.d(TAG, "Capture stopped and resources released")

        } catch (e: Exception) {
            Log.e(TAG, "Error stopping capture: ${e.message}", e)
        }
    }

    override fun onDestroy() {
        Log.d(TAG, "Service destroyed")
        stopCapture()
        serviceScope.cancel()
        handlerThread.quitSafely()
        super.onDestroy()
    }
}
