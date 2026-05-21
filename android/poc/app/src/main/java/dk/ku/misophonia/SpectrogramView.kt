package dk.ku.misophonia

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class SpectrogramView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val maxColumns = 256
    private var bitmap: Bitmap? = null
    private var canvasBitmap: Canvas? = null
    private val paint = Paint()
    private var colIndex = 0
    private var fftSize = 0

    // Reuse Rect objects to avoid allocations in onDraw
    private val srcRect1 = Rect()
    private val dstRect1 = RectF()
    private val srcRect2 = Rect()
    private val dstRect2 = RectF()

    fun update(magnitudes: FloatArray) {
        if (magnitudes.isEmpty()) return
        
        if (bitmap == null || bitmap!!.height != magnitudes.size) {
            fftSize = magnitudes.size
            bitmap = Bitmap.createBitmap(maxColumns, fftSize, Bitmap.Config.ARGB_8888)
            canvasBitmap = Canvas(bitmap!!)
            canvasBitmap?.drawColor(Color.BLACK)
        }

        for (i in magnitudes.indices) {
            // Increased gain for visibility. Magnitudes are often < 1.0
            val mag = magnitudes[i]
            // Scale and log for better dynamic range visualization
            val intensity = (Math.log10(mag.toDouble() * 1000.0 + 1.0) * 80).coerceIn(0.0, 255.0).toInt()
            
            // Vibrant color mapping: Deep Purple -> Red -> Orange -> Yellow -> White
            val r = (intensity * 2.0).coerceIn(0.0, 255.0).toInt()
            val g = (intensity * 1.2 - 50).coerceIn(0.0, 255.0).toInt()
            val b = (intensity * 0.8).coerceIn(0.0, 255.0).toInt()
            
            paint.color = Color.rgb(r, g, b)
            canvasBitmap?.drawPoint(colIndex.toFloat(), (fftSize - 1 - i).toFloat(), paint)
        }

        colIndex = (colIndex + 1) % maxColumns
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val b = bitmap ?: return
        
        val w = width.toFloat()
        val h = height.toFloat()
        
        // Split and wrap the bitmap for a scrolling effect
        srcRect1.set(colIndex, 0, maxColumns, fftSize)
        dstRect1.set(0f, 0f, w * (maxColumns - colIndex) / maxColumns, h)
        
        srcRect2.set(0, 0, colIndex, fftSize)
        dstRect2.set(w * (maxColumns - colIndex) / maxColumns, 0f, w, h)
        
        canvas.drawBitmap(b, srcRect1, dstRect1, null)
        canvas.drawBitmap(b, srcRect2, dstRect2, null)
    }
}
