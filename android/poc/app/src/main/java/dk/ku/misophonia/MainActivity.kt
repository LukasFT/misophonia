package dk.ku.misophonia

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.content.pm.PackageManager
import android.media.*
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.math.max
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private val tag = "MisophoniaPoc"

    private val sampleRate = 44_100
    private val outputChannels = 2

    private val classNames = listOf(
        "chewing_gum",
        "clearing_throat",
        "human_breathing",
        "knife_cutting",
        "plastic_crumpling",
        "swallowing",
        "typing",
        "water_drops"
    )

    private var currentLabelIndex = 0

    private lateinit var status: TextView
    private lateinit var button: Button
    private lateinit var classSpinner: Spinner
    private lateinit var inputLevelBar: ProgressBar
    private lateinit var outputLevelBar: ProgressBar

    private lateinit var env: OrtEnvironment
    private var session: OrtSession? = null
    private var inputShapes: Map<String, LongArray>? = null

    private var chunkSamples = 416
    private var label = FloatArray(8)
    private var encBuf = FloatArray(0)
    private var decBuf = FloatArray(0)
    private var outBuf = FloatArray(0)

    private val running = AtomicBoolean(false)
    private var worker: Thread? = null

    @Volatile
    private var latestInputRms = 0f
    @Volatile
    private var latestOutputRms = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setupUI()

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 7)
        }

        try {
            initOnnx()
        } catch (e: Exception) {
            Log.e(tag, "Failed to init ONNX", e)
            status.text = "Error loading model: ${e.message}"
        }

        startUpdateTimer()
    }

    private fun setupUI() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(64, 64, 64, 64)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        val scroll = ScrollView(this).apply {
            addView(root)
        }

        status = TextView(this).apply {
            text = "Initializing..."
            textSize = 16f
            setPadding(0, 0, 0, 48)
        }
        root.addView(status)

        root.addView(TextView(this).apply {
            text = "Target Trigger Class"
            textSize = 14f
        })

        classSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                classNames
            )
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    updateLabelIndex(position)
                }
                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
            setPadding(0, 24, 0, 48)
        }
        root.addView(classSpinner)

        root.addView(TextView(this).apply { text = "Input Level (Raw Mic)" })
        inputLevelBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            setPadding(0, 16, 0, 32)
        }
        root.addView(inputLevelBar)

        root.addView(TextView(this).apply { text = "Output Level (Filtered)" })
        outputLevelBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            setPadding(0, 16, 0, 48)
        }
        root.addView(outputLevelBar)

        button = Button(this).apply {
            text = "Start Live Filtering"
            setOnClickListener {
                if (running.get()) stopDemo() else startDemo()
            }
        }
        root.addView(button)

        setContentView(scroll)
    }

    private fun updateLabelIndex(index: Int) {
        currentLabelIndex = index
        label.fill(0f)
        if (index in label.indices) {
            label[index] = 1.0f
        }
        Log.i(tag, "Selected class: ${classNames[index]}")
    }

    private fun initOnnx() {
        val modelFile = copyAssetToCache("misophonia_anc_step.onnx")
        val metaText = assets.open("misophonia_anc_step.mobile_metadata.json").bufferedReader().use { it.readText() }

        val meta = JSONObject(metaText)
        chunkSamples = meta.getInt("chunk_samples")

        val shapesObj = meta.getJSONObject("input_shapes")
        val shapes = mutableMapOf<String, LongArray>()
        arrayOf("mix", "label", "enc_buf", "dec_buf", "out_buf").forEach { key ->
            shapes[key] = jsonLongArray(shapesObj.getJSONArray(key))
        }
        inputShapes = shapes

        env = OrtEnvironment.getEnvironment()
        session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())

        label = FloatArray(sizeOf(shapes.getValue("label")))
        updateLabelIndex(currentLabelIndex)

        encBuf = FloatArray(sizeOf(shapes.getValue("enc_buf")))
        decBuf = FloatArray(sizeOf(shapes.getValue("dec_buf")))
        outBuf = FloatArray(sizeOf(shapes.getValue("out_buf")))

        status.text = "Model: misophonia_anc_step.onnx\nChunk size: $chunkSamples\nSample rate: $sampleRate"
    }

    private fun startUpdateTimer() {
        thread(start = true, isDaemon = true) {
            while (!isDestroyed) {
                Thread.sleep(50)
                if (running.get()) {
                    val inP = (latestInputRms * 100).toInt().coerceIn(0, 100)
                    val outP = (latestOutputRms * 100).toInt().coerceIn(0, 100)
                    runOnUiThread {
                        inputLevelBar.progress = inP
                        outputLevelBar.progress = outP
                    }
                }
            }
        }
    }

    private fun startDemo() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 7)
            return
        }

        if (session == null) {
            Toast.makeText(this, "Model not loaded", Toast.LENGTH_SHORT).show()
            return
        }

        running.set(true)
        button.text = "Stop"
        worker = thread(start = true, name = "audio-worker") {
            try {
                audioLoop()
            } catch (t: Throwable) {
                Log.e(tag, "Audio loop error", t)
                runOnUiThread {
                    Toast.makeText(this, "Error: ${t.message}", Toast.LENGTH_LONG).show()
                    stopDemo()
                }
            }
        }
    }

    private fun stopDemo() {
        running.set(false)
        button.text = "Start Live Filtering"
        inputLevelBar.progress = 0
        outputLevelBar.progress = 0
    }

    private fun audioLoop() {
        val preferredInput = chooseInputDevice()
        val preferStereo = preferredInput?.channelCounts?.let { it.isEmpty() || it.contains(2) } ?: false

        var (recorder, inputChannels) = makeRecorder(stereo = preferStereo, preferredInput)
        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            val fallback = makeRecorder(stereo = false, preferredInput)
            recorder = fallback.first
            inputChannels = fallback.second
        }

        val track = makeAudioTrack()

        val readBuf = FloatArray(chunkSamples * inputChannels)
        val mixInput = FloatArray(outputChannels * chunkSamples)
        val playBuf = FloatArray(outputChannels * chunkSamples)

        recorder.startRecording()
        track.play()

        try {
            while (running.get()) {
                val n = recorder.read(readBuf, 0, readBuf.size, AudioRecord.READ_BLOCKING)
                if (n <= 0) continue

                if (inputChannels == 1) {
                    for (i in 0 until chunkSamples) {
                        val s = readBuf[i]
                        mixInput[i] = s
                        mixInput[chunkSamples + i] = s
                    }
                } else {
                    for (i in 0 until chunkSamples) {
                        mixInput[i] = readBuf[2 * i]
                        mixInput[chunkSamples + i] = readBuf[2 * i + 1]
                    }
                }

                latestInputRms = calculateRms(mixInput)

                val filtered = runOnnxStep(mixInput) ?: continue
                latestOutputRms = calculateRms(filtered)

                for (i in 0 until chunkSamples) {
                    playBuf[2 * i] = filtered[i].coerceIn(-1.0f, 1.0f)
                    playBuf[2 * i + 1] = filtered[chunkSamples + i].coerceIn(-1.0f, 1.0f)
                }

                track.write(playBuf, 0, playBuf.size, AudioTrack.WRITE_BLOCKING)
            }
        } finally {
            recorder.stop()
            recorder.release()
            track.stop()
            track.release()
        }
    }

    private fun calculateRms(audio: FloatArray): Float {
        var sum = 0f
        for (x in audio) sum += x * x
        return sqrt(sum / audio.size)
    }

    private fun runOnnxStep(mix: FloatArray): FloatArray? {
        val sess = session ?: return null
        val shapes = inputShapes ?: return null
        
        val tensors = mutableMapOf<String, OnnxTensor>()
        try {
            val currentLabel = label.copyOf()
            tensors["mix"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(mix), shapes.getValue("mix"))
            tensors["label"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(currentLabel), shapes.getValue("label"))
            tensors["enc_buf"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(encBuf), shapes.getValue("enc_buf"))
            tensors["dec_buf"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(decBuf), shapes.getValue("dec_buf"))
            tensors["out_buf"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(outBuf), shapes.getValue("out_buf"))

            sess.run(tensors).use { result ->
                val x = tensorToFloatArray(result.get(0) as OnnxTensor)
                encBuf = tensorToFloatArray(result.get(1) as OnnxTensor)
                decBuf = tensorToFloatArray(result.get(2) as OnnxTensor)
                outBuf = tensorToFloatArray(result.get(3) as OnnxTensor)
                return x
            }
        } catch (e: Exception) {
            Log.e(tag, "Inference error", e)
            return null
        } finally {
            tensors.values.forEach { it.close() }
        }
    }

    private fun makeRecorder(stereo: Boolean, preferredInput: AudioDeviceInfo?): Pair<AudioRecord, Int> {
        val channelMask = if (stereo) AudioFormat.CHANNEL_IN_STEREO else AudioFormat.CHANNEL_IN_MONO
        val inputChannels = if (stereo) 2 else 1
        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelMask, AudioFormat.ENCODING_PCM_FLOAT)
        val bufferSize = max(minBufferSize, chunkSamples * inputChannels * 4 * 10)

        val recorder = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
            .setAudioFormat(AudioFormat.Builder()
                .setSampleRate(sampleRate)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(channelMask)
                .build())
            .setBufferSizeInBytes(bufferSize)
            .build()

        preferredInput?.let { recorder.setPreferredDevice(it) }
        return recorder to inputChannels
    }

    private fun makeAudioTrack(): AudioTrack {
        val minBufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_FLOAT)
        val bufferSize = max(minBufferSize, chunkSamples * outputChannels * 4 * 10)

        return AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                .build())
            .setAudioFormat(AudioFormat.Builder()
                .setSampleRate(sampleRate)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                .build())
            .setBufferSizeInBytes(bufferSize)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
    }

    private fun chooseInputDevice(): AudioDeviceInfo? {
        val audioManager = getSystemService(AudioManager::class.java)
        val inputs = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
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

    private fun jsonLongArray(arr: org.json.JSONArray): LongArray {
        return LongArray(arr.length()) { i -> arr.getLong(i) }
    }

    private fun sizeOf(shape: LongArray): Int {
        return shape.fold(1L) { acc, v -> acc * v }.toInt()
    }

    override fun onDestroy() {
        running.set(false)
        session?.close()
        env.close()
        super.onDestroy()
    }
}
