package com.plugin.devicelock

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Handler
import android.os.Looper
import android.util.TypedValue
import android.view.Gravity
import android.view.KeyEvent
import android.view.MotionEvent
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import java.text.DateFormat
import java.util.Date

/**
 * Full-screen block screen, built in code so the plugin needs no resources.
 * Swallows every touch and the Back key. Emergency unlock = hold the button
 * for [HOLD_MS].
 */
@SuppressLint("ViewConstructor")
class LockOverlayView(
    context: Context,
    private val onEmergencyUnlock: () -> Unit
) : FrameLayout(context) {

    companion object {
        private const val HOLD_MS = 5_000L
    }

    private val handler = Handler(Looper.getMainLooper())
    private var lockedUntil = 0L
    private var holdStartedAt = 0L

    private val reasonView = text(18f, Color.parseColor("#D0D0DA"))
    private val countdownView = text(40f, Color.WHITE, bold = true)
    private val untilView = text(14f, Color.parseColor("#9090A0"))
    private val emergencyButton = text(15f, Color.parseColor("#E0E0E8")).apply {
        setPadding(dp(24), dp(14), dp(24), dp(14))
        background = GradientDrawable().apply {
            cornerRadius = dp(28).toFloat()
            setStroke(dp(1), Color.parseColor("#55556A"))
        }
    }

    private var running = false

    private val tick = object : Runnable {
        override fun run() {
            if (!running) return
            updateCountdown()
            updateHold()
            if (!running) return // updateHold may have triggered the unlock
            handler.postDelayed(this, if (holdStartedAt > 0) 100L else 1_000L)
        }
    }

    init {
        setBackgroundColor(Color.parseColor("#F20B0B12"))
        isFocusable = true
        isFocusableInTouchMode = true
        isClickable = true

        val column = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(dp(32), 0, dp(32), 0)
        }
        column.addView(text(44f, Color.WHITE).apply { text = "🔒" })
        column.addView(space(16))
        column.addView(text(26f, Color.WHITE, bold = true).apply { text = "Locked by Observer" })
        column.addView(space(12))
        column.addView(reasonView)
        column.addView(space(40))
        column.addView(countdownView)
        column.addView(space(4))
        column.addView(untilView)
        column.addView(space(56))
        column.addView(emergencyButton, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.CENTER_HORIZONTAL })

        addView(column, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT, Gravity.CENTER))

        emergencyButton.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> holdStartedAt = System.currentTimeMillis()
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> holdStartedAt = 0L
            }
            restartTick()
            true
        }
    }

    fun bind(reason: String, until: Long, allowEmergency: Boolean) {
        reasonView.text = reason
        lockedUntil = until
        untilView.text = "Unlocks at " + DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(until))
        emergencyButton.visibility = if (allowEmergency) VISIBLE else GONE
        holdStartedAt = 0L
        running = true
        restartTick()
        requestFocus()
    }

    fun stop() {
        running = false
        handler.removeCallbacks(tick)
    }

    private fun restartTick() {
        if (!running) return
        handler.removeCallbacks(tick)
        handler.post(tick)
    }

    override fun onDetachedFromWindow() {
        stop()
        super.onDetachedFromWindow()
    }

    // Swallow Back and other keys; let volume keys through.
    override fun dispatchKeyEvent(event: KeyEvent): Boolean = when (event.keyCode) {
        KeyEvent.KEYCODE_VOLUME_UP, KeyEvent.KEYCODE_VOLUME_DOWN, KeyEvent.KEYCODE_VOLUME_MUTE ->
            super.dispatchKeyEvent(event)
        else -> true
    }

    private fun updateCountdown() {
        val left = ((lockedUntil - System.currentTimeMillis()) / 1000).coerceAtLeast(0)
        val h = left / 3600
        val m = (left % 3600) / 60
        val s = left % 60
        countdownView.text = if (h > 0) "%d:%02d:%02d".format(h, m, s) else "%d:%02d".format(m, s)
    }

    private fun updateHold() {
        if (holdStartedAt == 0L) {
            emergencyButton.text = "Hold for emergency unlock"
            return
        }
        val remaining = HOLD_MS - (System.currentTimeMillis() - holdStartedAt)
        if (remaining <= 0) {
            holdStartedAt = 0L
            onEmergencyUnlock()
        } else {
            emergencyButton.text = "Keep holding… ${(remaining + 999) / 1000}"
        }
    }

    private fun text(sp: Float, color: Int, bold: Boolean = false) = TextView(context).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sp)
        setTextColor(color)
        gravity = Gravity.CENTER
        if (bold) typeface = Typeface.DEFAULT_BOLD
    }

    private fun space(height: Int) = android.view.View(context).apply {
        layoutParams = LinearLayout.LayoutParams(1, dp(height))
    }

    private fun dp(v: Int): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics).toInt()
}
