package dk.ku.misophonia

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.*
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.math.max

class MainActivity : Activity() {
    private val tag = "MisophoniaPoc"

    private val sampleRate = 44_100
    private val outputChannels = 2

    // Your export used label-index 3. Change this to test another trigger class.
    private val labelIndex = 3

    // If the checkpoint predicts isolated trigger audio, listen to mix - prediction.
    // If the checkpoint predicts clean_mix directly, set this to false.
    private val playResidual = true

    private lateinit var status: TextView
    private lateinit var button: Button

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession
    private lateinit var inputShapes: Map<String, LongArray>

    private var chunkSamples = 416
    private var label = FloatArray(0)
    private var encBuf = FloatArray(0)
    private var decBuf = FloatArray(0)
    private var outBuf = FloatArray(0)

    private val running = AtomicBoolean(false)
    private var worker: Thread? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        status = TextView(this).apply {
            text = "Ready"
            textSize = 14f
            setPadding(24, 24, 24, 24)
        }

        button = Button(this).apply {
            text = "Start live demo"
            setOnClickListener {
                if (running.get()) stopDemo() else startDemo()
            }
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(status)
            addView(button)
        }

        setContentView(layout)

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 7)
        }

        initOnnx()
    }

    private fun initOnnx() {
        val modelFile = copyAssetToCache("misophonia_anc_step.onnx")
        val metaText = assets.open("misophonia_anc_step.mobile_metadata.json")
            .bufferedReader()
            .use { it.readText() }

        val meta = JSONObject(metaText)
        chunkSamples = meta.getInt("chunk_samples")

        val shapesObj = meta.getJSONObject("input_shapes")
        inputShapes = mapOf(
            "mix" to jsonLongArray(shapesObj.getJSONArray("mix")),
            "label" to jsonLongArray(shapesObj.getJSONArray("label")),
            "enc_buf" to jsonLongArray(shapesObj.getJSONArray("enc_buf")),
            "dec_buf" to jsonLongArray(shapesObj.getJSONArray("dec_buf")),
            "out_buf" to jsonLongArray(shapesObj.getJSONArray("out_buf")),
        )

        env = OrtEnvironment.getEnvironment()
        session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())

        label = FloatArray(sizeOf(inputShapes.getValue("label")))
        label[labelIndex] = 1.0f

        encBuf = FloatArray(sizeOf(inputShapes.getValue("enc_buf")))
        decBuf = FloatArray(sizeOf(inputShapes.getValue("dec_buf")))
        outBuf = FloatArray(sizeOf(inputShapes.getValue("out_buf")))

        logStatus(
            "ONNX loaded\n" +
                "chunkSamples=$chunkSamples\n" +
                "sampleRate=$sampleRate\n" +
                "labelIndex=$labelIndex\n" +
                "playResidual=$playResidual"
        )
    }

    private fun startDemo() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 7)
            return
        }

        running.set(true)
        button.text = "Stop"
        worker = thread(start = true, name = "audio-onnx-loop") {
            try {
                audioLoop()
            } catch (t: Throwable) {
                Log.e(tag, "audio loop failed", t)
                logStatus("ERROR: ${t.message}")
                running.set(false)
            }
        }
    }

    private fun stopDemo() {
        running.set(false)
        button.text = "Start live demo"
    }

    private fun audioLoop() {
        val preferredInput = chooseInputDevice()
        val preferStereo = preferredInput?.channelCounts?.let { counts ->
            counts.isEmpty() || counts.contains(2)
        } ?: false

        var recorderAndChannels = makeRecorder(stereo = preferStereo, preferredInput)
        if (recorderAndChannels.first.state != AudioRecord.STATE_INITIALIZED) {
            recorderAndChannels.first.release()
            recorderAndChannels = makeRecorder(stereo = false, preferredInput)
        }

        val recorder = recorderAndChannels.first
        val inputChannels = recorderAndChannels.second

        val track = makeAudioTrack()

        preferredInput?.let { recorder.setPreferredDevice(it) }

        logStatus(
            "Starting\n" +
                "inputChannels=$inputChannels\n" +
                "preferredInputType=${preferredInput?.type}\n" +
                "preferredInputChannels=${preferredInput?.channelCounts?.joinToString()}"
        )

        val readBuf = FloatArray(chunkSamples * inputChannels)
        val mix = FloatArray(outputChannels * chunkSamples)
        val playBuf = FloatArray(outputChannels * chunkSamples)

        recorder.startRecording()
        track.play()

        while (running.get()) {
            val n = recorder.read(
                readBuf,
                0,
                readBuf.size,
                AudioRecord.READ_BLOCKING
            )

            if (n <= 0) {
                Log.w(tag, "AudioRecord.read returned $n")
                continue
            }

            if (n < readBuf.size) {
                for (i in n until readBuf.size) readBuf[i] = 0.0f
            }

            if (inputChannels == 1) {
                for (i in 0 until chunkSamples) {
                    val s = readBuf[i]
                    mix[i] = s
                    mix[chunkSamples + i] = s
                }
            } else {
                for (i in 0 until chunkSamples) {
                    mix[i] = readBuf[2 * i]
                    mix[chunkSamples + i] = readBuf[2 * i + 1]
                }
            }

            val pred = runOnnxStep(mix)

            for (i in 0 until chunkSamples) {
                val modelL = pred[i]
                val modelR = pred[chunkSamples + i]

                val yL = if (playResidual) mix[i] - modelL else modelL
                val yR = if (playResidual) mix[chunkSamples + i] - modelR else modelR

                playBuf[2 * i] = yL.coerceIn(-1.0f, 1.0f)
                playBuf[2 * i + 1] = yR.coerceIn(-1.0f, 1.0f)
            }

            track.write(
                playBuf,
                0,
                playBuf.size,
                AudioTrack.WRITE_BLOCKING
            )
        }

        recorder.stop()
        recorder.release()

        track.stop()
        track.release()

        logStatus("Stopped")
    }

    private fun runOnnxStep(mix: FloatArray): FloatArray {
        val tensors = linkedMapOf<String, OnnxTensor>()

        try {
            tensors["mix"] = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(mix),
                inputShapes.getValue("mix")
            )
            tensors["label"] = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(label),
                inputShapes.getValue("label")
            )
            tensors["enc_buf"] = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(encBuf),
                inputShapes.getValue("enc_buf")
            )
            tensors["dec_buf"] = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(decBuf),
                inputShapes.getValue("dec_buf")
            )
            tensors["out_buf"] = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(outBuf),
                inputShapes.getValue("out_buf")
            )

            session.run(tensors).use { result ->
                val x = tensorToFloatArray(result.get(0) as OnnxTensor)
                encBuf = tensorToFloatArray(result.get(1) as OnnxTensor)
                decBuf = tensorToFloatArray(result.get(2) as OnnxTensor)
                outBuf = tensorToFloatArray(result.get(3) as OnnxTensor)
                return x
            }
        } finally {
            tensors.values.forEach { it.close() }
        }
    }

    private fun makeRecorder(
        stereo: Boolean,
        preferredInput: AudioDeviceInfo?
    ): Pair<AudioRecord, Int> {
        val channelMask = if (stereo) {
            AudioFormat.CHANNEL_IN_STEREO
        } else {
            AudioFormat.CHANNEL_IN_MONO
        }

        val inputChannels = if (stereo) 2 else 1

        val minBytes = AudioRecord.getMinBufferSize(
            sampleRate,
            channelMask,
            AudioFormat.ENCODING_PCM_FLOAT
        )

        val bufferBytes = max(
            if (minBytes > 0) minBytes else 0,
            chunkSamples * inputChannels * 4 * 8
        )

        val format = AudioFormat.Builder()
            .setSampleRate(sampleRate)
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(channelMask)
            .build()

        val recorder = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
            .setAudioFormat(format)
            .setBufferSizeInBytes(bufferBytes)
            .build()

        preferredInput?.let { recorder.setPreferredDevice(it) }

        return recorder to inputChannels
    }

    private fun makeAudioTrack(): AudioTrack {
        val minBytes = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_STEREO,
            AudioFormat.ENCODING_PCM_FLOAT
        )

        val bufferBytes = max(
            if (minBytes > 0) minBytes else 0,
            chunkSamples * outputChannels * 4 * 8
        )

        val format = AudioFormat.Builder()
            .setSampleRate(sampleRate)
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
            .build()

        val attributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
            .build()

        return AudioTrack.Builder()
            .setAudioAttributes(attributes)
            .setAudioFormat(format)
            .setBufferSizeInBytes(bufferBytes)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
    }

    private fun chooseInputDevice(): AudioDeviceInfo? {
        val audioManager = getSystemService(AudioManager::class.java)
        val inputs = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS).toList()

        return inputs.firstOrNull {
            it.type == AudioDeviceInfo.TYPE_USB_HEADSET ||
                it.type == AudioDeviceInfo.TYPE_USB_DEVICE ||
                it.type == AudioDeviceInfo.TYPE_WIRED_HEADSET
        } ?: inputs.firstOrNull()
    }

    private fun copyAssetToCache(assetName: String): File {
        val outFile = File(cacheDir, assetName)
        assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        return outFile
    }

    private fun tensorToFloatArray(tensor: OnnxTensor): FloatArray {
        val buffer = tensor.getFloatBuffer()
        buffer.rewind()
        val out = FloatArray(buffer.remaining())
        buffer.get(out)
        return out
    }

    private fun jsonLongArray(arr: JSONArray): LongArray {
        return LongArray(arr.length()) { i -> arr.getLong(i) }
    }

    private fun sizeOf(shape: LongArray): Int {
        return shape.fold(1L) { acc, v -> acc * v }.toInt()
    }

    private fun logStatus(msg: String) {
        Log.i(tag, msg)
        runOnUiThread {
            status.text = msg
        }
    }

    override fun onDestroy() {
        running.set(false)
        super.onDestroy()
    }
}
