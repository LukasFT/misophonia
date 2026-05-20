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
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.math.max
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private val logTag = "MisophoniaPoc"

    private val sampleRate = 44_100
    private val outputChannels = 2

    private var classNames = listOf(
        "chewing_gum",
        "clearing_throat",
        "human_breathing",
        "knife_cutting",
        "plastic_crumpling",
        "swallowing",
        "typing",
        "water_drops"
    )

    private val playbackModes = listOf(
        "Microphone input",
        "Model output",
        "Model output subtracted (Input - Model)"
    )

    private var currentPlaybackModeIndex = 0
    private var currentLabelIndex = 0
    private var currentModelIndex = -1

    private lateinit var status: TextView
    private lateinit var controlButton: Button
    private lateinit var classSpinner: Spinner
    private lateinit var playbackSpinner: Spinner
    private lateinit var modelSpinner: Spinner
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

    private var modelsList = mutableListOf<JSONObject>()

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
            env = OrtEnvironment.getEnvironment()
            loadManifest()
        } catch (e: Exception) {
            Log.e(logTag, "Failed to load manifest", e)
            status.text = "Error loading manifest: ${e.message}"
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

        // Model Selection Spinner
        root.addView(TextView(this).apply { text = "Select Model"; textSize = 14f })
        modelSpinner = Spinner(this).apply {
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    if (position != currentModelIndex) {
                        loadModel(position)
                    }
                }
                override fun onNothingSelected(p0: AdapterView<*>?) {}
            }
            setPadding(0, 16, 0, 32)
        }
        root.addView(modelSpinner)

        // Playback Mode Spinner
        root.addView(TextView(this).apply { text = "Playback Mode"; textSize = 14f })
        playbackSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, playbackModes)
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    currentPlaybackModeIndex = position
                    Log.i(logTag, "Playback mode: ${playbackModes[position]}")
                }
                override fun onNothingSelected(p0: AdapterView<*>?) {}
            }
            setPadding(0, 16, 0, 32)
        }
        root.addView(playbackSpinner)

        // Trigger Class Spinner
        root.addView(TextView(this).apply { text = "Target Trigger Class"; textSize = 14f })
        classSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, classNames)
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    updateLabelIndex(position)
                }
                override fun onNothingSelected(p0: AdapterView<*>?) {}
            }
            setPadding(0, 16, 0, 48)
        }
        root.addView(classSpinner)

        root.addView(TextView(this).apply { text = "Input Level (Raw Mic)" })
        inputLevelBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            setPadding(0, 16, 0, 32)
        }
        root.addView(inputLevelBar)

        root.addView(TextView(this).apply { text = "Output Level (Playback Signal)" })
        outputLevelBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            setPadding(0, 16, 0, 48)
        }
        root.addView(outputLevelBar)

        controlButton = Button(this).apply {
            text = "Start Live Audio"
            setOnClickListener {
                if (running.get()) stopDemo() else startDemo()
            }
        }
        root.addView(controlButton)

        setContentView(scroll)
    }

    private fun loadManifest() {
        val manifestText = assets.open("misophonia_anc_models.json").bufferedReader().use { it.readText() }
        val manifest = JSONObject(manifestText)
        
        val classes = manifest.getJSONArray("class_names")
        classNames = (0 until classes.length()).map { classes.getString(it) }
        
        // Update class spinner
        runOnUiThread {
            classSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, classNames)
            classSpinner.setSelection(manifest.optInt("default_class_index", 0))
        }

        val models = manifest.getJSONArray("models")
        val displayNames = mutableListOf<String>()
        modelsList.clear()
        for (i in 0 until models.length()) {
            val m = models.getJSONObject(i)
            modelsList.add(m)
            displayNames.add(m.getString("display_name"))
        }

        runOnUiThread {
            modelSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, displayNames)
            if (modelsList.isNotEmpty()) {
                modelSpinner.setSelection(0)
            }
        }
    }

    private fun loadModel(index: Int) {
        val wasRunning = running.get()
        if (wasRunning) {
            stopDemo()
        }
        
        session?.close()
        session = null
        
        currentModelIndex = index
        val modelMeta = modelsList[index]
        val onnxAssetName = modelMeta.getString("onnx_asset_name")
        
        try {
            val modelFile = copyAssetToCache(onnxAssetName)
            chunkSamples = modelMeta.getInt("chunk_samples")
            
            val shapesObj = modelMeta.getJSONObject("input_shapes")
            val shapes = mutableMapOf<String, LongArray>()
            arrayOf("mix", "label", "enc_buf", "dec_buf", "out_buf").forEach { key ->
                shapes[key] = jsonLongArray(shapesObj.getJSONArray(key))
            }
            inputShapes = shapes
            
            session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
            
            label = FloatArray(sizeOf(shapes.getValue("label")))
            updateLabelIndex(currentLabelIndex)
            
            encBuf = FloatArray(sizeOf(shapes.getValue("enc_buf")))
            decBuf = FloatArray(sizeOf(shapes.getValue("dec_buf")))
            outBuf = FloatArray(sizeOf(shapes.getValue("out_buf")))
            
            status.text = "Model Loaded: $onnxAssetName\nChunk size: $chunkSamples"
            
            if (wasRunning) {
                startDemo()
            }
        } catch (e: Exception) {
            Log.e(logTag, "Failed to load model $onnxAssetName", e)
            status.text = "Error loading model: ${e.message}"
        }
    }

    private fun updateLabelIndex(index: Int) {
        currentLabelIndex = index
        label.fill(0f)
        if (index in label.indices) {
            label[index] = 1.0f
        }
        Log.i(logTag, "Selected class: ${classNames.getOrNull(index) ?: index}")
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
        controlButton.text = "Stop"
        
        worker = thread(start = true, name = "audio-worker") {
            try {
                audioLoop()
            } catch (t: Throwable) {
                Log.e(logTag, "Audio loop error", t)
                runOnUiThread {
                    Toast.makeText(this, "Error: ${t.message}", Toast.LENGTH_LONG).show()
                    stopDemo()
                }
            }
        }
    }

    private fun stopDemo() {
        running.set(false)
        controlButton.text = "Start Live Audio"
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

                // Determine playback signal based on selected mode
                val outputSignal = when (currentPlaybackModeIndex) {
                    0 -> mixInput // Microphone input
                    1 -> runOnnxStep(mixInput) // Model output
                    2 -> {
                        val filtered = runOnnxStep(mixInput)
                        if (filtered != null) {
                            val sub = FloatArray(mixInput.size)
                            for (i in sub.indices) sub[i] = mixInput[i] - filtered[i]
                            sub
                        } else null
                    }
                    else -> null
                }

                if (outputSignal != null) {
                    latestOutputRms = calculateRms(outputSignal)
                    for (i in 0 until chunkSamples) {
                        playBuf[2 * i] = outputSignal[i].coerceIn(-1.0f, 1.0f)
                        playBuf[2 * i + 1] = outputSignal[chunkSamples + i].coerceIn(-1.0f, 1.0f)
                    }
                    track.write(playBuf, 0, playBuf.size, AudioTrack.WRITE_BLOCKING)
                }
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
            Log.e(logTag, "Inference error", e)
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

    private fun jsonLongArray(arr: JSONArray): LongArray {
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
