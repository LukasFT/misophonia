package dk.ku.misophonia

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class WaveformView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    private val paint = Paint().apply {
        color = Color.parseColor("#4CAF50") // Green
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }
    
    private val maxDataPoints = 128
    private val data = FloatArray(maxDataPoints)
    private var head = 0

    fun addValue(value: Float) {
        data[head] = value
        head = (head + 1) % maxDataPoints
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        val centerY = h / 2f
        
        if (w <= 0f || h <= 0f) return

        val step = w / (maxDataPoints - 1)
        
        // Draw a center line
        paint.alpha = 64
        paint.strokeWidth = 1f
        canvas.drawLine(0f, centerY, w, centerY, paint)
        
        paint.alpha = 255
        paint.strokeWidth = 4f
        
        for (i in 0 until maxDataPoints - 1) {
            val idx1 = (head + i) % maxDataPoints
            val idx2 = (head + i + 1) % maxDataPoints
            
            // Normalize value to some reasonable range. 
            // RMS is typically 0 to 1, but often much smaller.
            // Let's amplify it slightly for visibility.
            val amp = 2.0f 
            val v1 = (data[idx1] * amp).coerceAtMost(1.0f) * (h / 2f)
            val v2 = (data[idx2] * amp).coerceAtMost(1.0f) * (h / 2f)
            
            val x1 = i * step
            val x2 = (i + 1) * step
            
            canvas.drawLine(x1, centerY - v1, x2, centerY - v2, paint)
            canvas.drawLine(x1, centerY + v1, x2, centerY + v2, paint)
        }
    }
}
