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
import android.content.Intent
import android.graphics.Typeface
import androidx.core.content.FileProvider
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
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
    private var currentModelIndex = -1
    private var selectedClasses = BooleanArray(0)

    private lateinit var status: TextView
    private lateinit var classSelectBtn: Button
    private lateinit var playbackSpinner: Spinner
    private lateinit var modelSpinner: Spinner
    private lateinit var inputWaveform: WaveformView
    private lateinit var outputWaveform: WaveformView
    private lateinit var inputSpectrogram: SpectrogramView
    private lateinit var outputSpectrogram: SpectrogramView

    private lateinit var liveBtn: Button
    private lateinit var recordBtn: Button
    private lateinit var disabledBtn: Button
    private lateinit var debugLayout: LinearLayout
    private lateinit var debugInfo: TextView
    private lateinit var debugToggle: Button

    private var currentMode = SessionMode.DISABLED
    private var recordingInputFile: File? = null
    private var recordingOutputFile: File? = null
    private var recordingInputStream: DataOutputStream? = null
    private var recordingOutputStream: DataOutputStream? = null
    private val latencyHistory = java.util.Collections.synchronizedList(mutableListOf<Double>())

    enum class SessionMode { DISABLED, LIVE, RECORD }

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
    @Volatile
    private var latestInputFft: FloatArray? = null
    @Volatile
    private var latestOutputFft: FloatArray? = null
    @Volatile
    private var latestLatencyMs = 0.0

    private val fft = FastFourierTransform(512)

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
            setPadding(64, 128, 64, 64)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        val scroll = ScrollView(this).apply {
            addView(root)
        }

        root.addView(TextView(this).apply {
            text = "MisoSoupression"
            textSize = 28f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(0, 16, 0, 32)
            textAlignment = View.TEXT_ALIGNMENT_CENTER
        })

        status = TextView(this).apply {
            text = "Initializing..."
            textSize = 14f
            setPadding(0, 0, 0, 32)
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

        // Trigger Class Selection
        root.addView(TextView(this).apply { text = "Target Trigger Classes"; textSize = 14f })
        classSelectBtn = Button(this).apply {
            text = "Select Classes"
            setOnClickListener { showClassSelectionDialog() }
            setPadding(0, 16, 0, 48)
        }
        root.addView(classSelectBtn)

        val btnRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            weightSum = 3f
        }
        
        liveBtn = Button(this).apply {
            text = "Live"
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener { setSessionMode(SessionMode.LIVE) }
        }
        recordBtn = Button(this).apply {
            text = "Record"
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener { setSessionMode(SessionMode.RECORD) }
        }
        disabledBtn = Button(this).apply {
            text = "Disabled"
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener { setSessionMode(SessionMode.DISABLED) }
        }
        
        btnRow.addView(liveBtn)
        btnRow.addView(recordBtn)
        btnRow.addView(disabledBtn)
        root.addView(btnRow)

        root.addView(TextView(this).apply { text = "Input Waveform (Mic)"; setPadding(0, 32, 0, 0) })
        inputWaveform = WaveformView(this).apply {
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 250).apply {
                setMargins(0, 16, 0, 32)
            }
        }
        root.addView(inputWaveform)

        root.addView(TextView(this).apply { text = "Input Spectrogram" })
        inputSpectrogram = SpectrogramView(this).apply {
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 400).apply {
                setMargins(0, 16, 0, 48)
            }
        }
        root.addView(inputSpectrogram)

        root.addView(TextView(this).apply { text = "Output Waveform (Signal)" })
        outputWaveform = WaveformView(this).apply {
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 250).apply {
                setMargins(0, 16, 0, 32)
            }
        }
        root.addView(outputWaveform)

        root.addView(TextView(this).apply { text = "Output Spectrogram" })
        outputSpectrogram = SpectrogramView(this).apply {
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 400).apply {
                setMargins(0, 16, 0, 48)
            }
        }
        root.addView(outputSpectrogram)

        debugLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            setBackgroundColor(0x11000000)
            setPadding(32, 32, 32, 32)
        }
        debugInfo = TextView(this).apply {
            textSize = 12f
            text = "Debug Info..."
        }
        debugLayout.addView(debugInfo)
        root.addView(debugLayout)

        debugToggle = Button(this).apply {
            text = "Show Debug"
            setOnClickListener {
                if (debugLayout.visibility == View.VISIBLE) {
                    debugLayout.visibility = View.GONE
                    text = "Show Debug"
                } else {
                    debugLayout.visibility = View.VISIBLE
                    text = "Hide Debug"
                }
            }
        }
        root.addView(debugToggle)

        setContentView(scroll)
        updateButtonColors()
    }

    private fun setSessionMode(mode: SessionMode) {
        if (mode == currentMode) return
        
        val oldMode = currentMode
        currentMode = mode // Update mode FIRST
        
        if (oldMode == SessionMode.RECORD && (mode == SessionMode.LIVE || mode == SessionMode.DISABLED)) {
            stopRecordingAndShare()
        }

        updateButtonColors()

        if (mode != SessionMode.DISABLED && !running.get()) {
            startDemo()
        } else if (mode == SessionMode.DISABLED && running.get()) {
            stopDemo()
        }
    }

    private fun updateButtonColors() {
        runOnUiThread {
            liveBtn.alpha = if (currentMode == SessionMode.LIVE) 1.0f else 0.5f
            recordBtn.alpha = if (currentMode == SessionMode.RECORD) 1.0f else 0.5f
            disabledBtn.alpha = if (currentMode == SessionMode.DISABLED) 1.0f else 0.5f
        }
    }

    private fun stopRecordingAndShare() {
        try {
            recordingInputStream?.close()
            recordingOutputStream?.close()
            recordingInputStream = null
            recordingOutputStream = null
            
            val pcmIn = recordingInputFile ?: return
            val pcmOut = recordingOutputFile ?: return
            
            val timestamp = System.currentTimeMillis()
            val baseDir = externalCacheDir ?: cacheDir
            val wavInFile = File(baseDir, "input_$timestamp.wav")
            val wavOutFile = File(baseDir, "output_$timestamp.wav")
            val csvFile = File(baseDir, "latency_$timestamp.csv")
            
            pcmToWav(pcmIn, wavInFile)
            pcmToWav(pcmOut, wavOutFile)
            
            // Save latency history to CSV
            csvFile.printWriter().use { out ->
                out.println("Chunk,LatencyMs")
                synchronized(latencyHistory) {
                    latencyHistory.forEachIndexed { index, latency ->
                        out.println("$index,$latency")
                    }
                }
            }
            
            val inUri = FileProvider.getUriForFile(this, "dk.ku.misophonia.fileprovider", wavInFile)
            val outUri = FileProvider.getUriForFile(this, "dk.ku.misophonia.fileprovider", wavOutFile)
            val csvUri = FileProvider.getUriForFile(this, "dk.ku.misophonia.fileprovider", csvFile)
            
            val uris = arrayListOf(inUri, outUri, csvUri)
            val shareIntent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                type = "*/*"
                putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "Share Results"))
            
        } catch (e: Exception) {
            Log.e(logTag, "Error sharing results", e)
            runOnUiThread {
                Toast.makeText(this, "Sharing failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun pcmToWav(pcmFile: File, wavFile: File) {
        val totalAudioLen = pcmFile.length()
        val totalDataLen = totalAudioLen + 36
        val longSampleRate = sampleRate.toLong()
        val channels = 2
        val byteRate = 16 * sampleRate * channels / 8

        val header = ByteArray(44)
        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1 // PCM
        header[21] = 0
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (longSampleRate and 0xff).toByte()
        header[25] = ((longSampleRate shr 8) and 0xff).toByte()
        header[26] = ((longSampleRate shr 16) and 0xff).toByte()
        header[27] = ((longSampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = (2 * 16 / 8).toByte()
        header[33] = 0
        header[34] = 16
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
        header[43] = ((totalAudioLen shr 24) and 0xff).toByte()

        FileOutputStream(wavFile).use { out ->
            out.write(header)
            FileInputStream(pcmFile).use { input ->
                input.copyTo(out)
            }
        }
    }

    private fun loadManifest() {
        val manifestText = assets.open("misophonia_anc_models.json").bufferedReader().use { it.readText() }
        val manifest = JSONObject(manifestText)
        
        val classes = manifest.getJSONArray("class_names")
        classNames = (0 until classes.length()).map { classes.getString(it) }
        selectedClasses = BooleanArray(classNames.size)
        
        val defaultIdx = manifest.optInt("default_class_index", 0)
        if (defaultIdx in selectedClasses.indices) {
            selectedClasses[defaultIdx] = true
        }
        
        updateClassButtonText()

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

    private fun showClassSelectionDialog() {
        val builder = androidx.appcompat.app.AlertDialog.Builder(this)
        builder.setTitle("Select Trigger Classes")
        
        builder.setMultiChoiceItems(classNames.toTypedArray(), selectedClasses) { _, index, isChecked ->
            selectedClasses[index] = isChecked
        }
        
        builder.setPositiveButton("OK") { _, _ ->
            updateLabelsFromSelection()
            updateClassButtonText()
        }
        builder.setNegativeButton("Cancel", null)
        builder.show()
    }

    private fun updateClassButtonText() {
        val selected = classNames.filterIndexed { index, _ -> selectedClasses[index] }
        classSelectBtn.text = if (selected.isEmpty()) {
            "None Selected"
        } else if (selected.size <= 2) {
            selected.joinToString(", ")
        } else {
            "${selected.size} Classes Selected"
        }
    }

    private fun updateLabelsFromSelection() {
        label.fill(0f)
        for (i in selectedClasses.indices) {
            if (selectedClasses[i] && i in label.indices) {
                label[i] = 1.0f
            }
        }
        val selected = classNames.filterIndexed { index, _ -> selectedClasses[index] }
        Log.i(logTag, "Selected classes: $selected")
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
            updateLabelsFromSelection()
            
            encBuf = FloatArray(sizeOf(shapes.getValue("enc_buf")))
            decBuf = FloatArray(sizeOf(shapes.getValue("dec_buf")))
            outBuf = FloatArray(sizeOf(shapes.getValue("out_buf")))
            
            status.text = "Status: Model Ready"
            
            if (wasRunning) {
                startDemo()
            }
        } catch (e: Exception) {
            Log.e(logTag, "Failed to load model $onnxAssetName", e)
            status.text = "Error loading model: ${e.message}"
        }
    }

    private fun startUpdateTimer() {
        thread(start = true, isDaemon = true) {
            while (!isDestroyed) {
                Thread.sleep(50)
                if (running.get()) {
                    val inRms = latestInputRms
                    val outRms = latestOutputRms
                    val inFft = latestInputFft
                    val outFft = latestOutputFft
                    val latency = latestLatencyMs
                    val modelName = modelsList.getOrNull(currentModelIndex)?.optString("display_name") ?: "None"
                    
                    val latencyVal = if (currentPlaybackModeIndex == 0) "N/A (Bypass)" else "${"%.2f".format(latency)} ms"
                    
                    runOnUiThread {
                        if (currentMode != SessionMode.RECORD) {
                            inputWaveform.addValue(inRms)
                            outputWaveform.addValue(outRms)
                            inFft?.let { inputSpectrogram.update(it) }
                            outFft?.let { outputSpectrogram.update(it) }
                        }
                        
                        debugInfo.text = """
                            Model: $modelName
                            Chunk Size: $chunkSamples
                            Inference Latency: $latencyVal
                            Mode: $currentMode
                        """.trimIndent()
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
            setSessionMode(SessionMode.DISABLED)
            return
        }

        if (currentMode == SessionMode.RECORD) {
            latencyHistory.clear()
            recordingInputFile = File(cacheDir, "input.pcm")
            recordingOutputFile = File(cacheDir, "output.pcm")
            recordingInputStream = DataOutputStream(FileOutputStream(recordingInputFile))
            recordingOutputStream = DataOutputStream(FileOutputStream(recordingOutputFile))
        }

        running.set(true)
        
        worker = thread(start = true, name = "audio-worker") {
            try {
                audioLoop()
            } catch (t: Throwable) {
                Log.e(logTag, "Audio loop error", t)
                runOnUiThread {
                    Toast.makeText(this, "Error: ${t.message}", Toast.LENGTH_LONG).show()
                    setSessionMode(SessionMode.DISABLED)
                }
            }
        }
    }

    private fun stopDemo() {
        running.set(false)
        if (currentMode == SessionMode.RECORD) {
            stopRecordingAndShare()
        }
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
                if (currentMode != SessionMode.RECORD) {
                    latestInputFft = calculateFft(mixInput)
                }

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
                    if (currentMode != SessionMode.RECORD) {
                        latestOutputFft = calculateFft(outputSignal)
                    }
                    
                    val shortBufIn = ShortArray(mixInput.size)
                    val shortBufOut = ShortArray(playBuf.size)
                    for (i in 0 until chunkSamples) {
                        val inL = mixInput[i].coerceIn(-1.0f, 1.0f)
                        val inR = mixInput[chunkSamples + i].coerceIn(-1.0f, 1.0f)
                        val outL = outputSignal[i].coerceIn(-1.0f, 1.0f)
                        val outR = outputSignal[chunkSamples + i].coerceIn(-1.0f, 1.0f)
                        
                        playBuf[2 * i] = outL
                        playBuf[2 * i + 1] = outR
                        
                        if (currentMode == SessionMode.RECORD) {
                            shortBufIn[2 * i] = (inL * 32767).toInt().toShort()
                            shortBufIn[2 * i + 1] = (inR * 32767).toInt().toShort()
                            shortBufOut[2 * i] = (outL * 32767).toInt().toShort()
                            shortBufOut[2 * i + 1] = (outR * 32767).toInt().toShort()
                        }
                    }
                    
                    if (currentMode == SessionMode.RECORD) {
                        try {
                            recordingInputStream?.let { for (s in shortBufIn) it.writeShort(s.toInt()) }
                            recordingOutputStream?.let { for (s in shortBufOut) it.writeShort(s.toInt()) }
                        } catch (e: Exception) {
                            Log.e(logTag, "Recording write error", e)
                        }
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

    private fun calculateFft(mix: FloatArray): FloatArray {
        // Average channels and pad to 512
        val real = FloatArray(512)
        val imag = FloatArray(512)
        for (i in 0 until chunkSamples) {
            real[i] = (mix[i] + mix[chunkSamples + i]) / 2f
        }
        
        fft.transform(real, imag)
        
        // Calculate magnitudes for the first half (Nyquist)
        val mags = FloatArray(256)
        for (i in 0 until 256) {
            mags[i] = sqrt(real[i] * real[i] + imag[i] * imag[i])
        }
        return mags
    }

    private fun runOnnxStep(mix: FloatArray): FloatArray? {
        val sess = session ?: return null
        val shapes = inputShapes ?: return null
        
        val tensors = mutableMapOf<String, OnnxTensor>()
        try {
            val startTime = System.nanoTime()
            val currentLabel = label.copyOf()
            tensors["mix"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(mix), shapes.getValue("mix"))
            tensors["label"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(currentLabel), shapes.getValue("label"))
            tensors["enc_buf"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(encBuf), shapes.getValue("enc_buf"))
            tensors["dec_buf"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(decBuf), shapes.getValue("dec_buf"))
            tensors["out_buf"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(outBuf), shapes.getValue("out_buf"))

            sess.run(tensors).use { result ->
                val endTime = System.nanoTime()
                val latency = (endTime - startTime) / 1_000_000.0
                latestLatencyMs = latency
                if (currentMode == SessionMode.RECORD) {
                    latencyHistory.add(latency)
                }

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

    class FastFourierTransform(private val n: Int) {
        private val cos = FloatArray(n / 2)
        private val sin = FloatArray(n / 2)

        init {
            for (i in 0 until n / 2) {
                cos[i] = kotlin.math.cos(2 * Math.PI * i / n).toFloat()
                sin[i] = kotlin.math.sin(2 * Math.PI * i / n).toFloat()
            }
        }

        fun transform(real: FloatArray, imag: FloatArray) {
            var j = 0
            for (i in 0 until n) {
                if (i < j) {
                    val tempReal = real[i]
                    real[i] = real[j]
                    real[j] = tempReal
                    val tempImag = imag[i]
                    imag[i] = imag[j]
                    imag[j] = tempImag
                }
                var m = n shr 1
                while (m >= 1 && j >= m) {
                    j -= m
                    m = m shr 1
                }
                j += m
            }

            var m = 2
            while (m <= n) {
                val halfM = m / 2
                val step = n / m
                for (k in 0 until n step m) {
                    for (i in 0 until halfM) {
                        val wr = cos[i * step]
                        val wi = -sin[i * step]
                        val tr = wr * real[k + i + halfM] - wi * imag[k + i + halfM]
                        val ti = wr * imag[k + i + halfM] + wi * real[k + i + halfM]
                        real[k + i + halfM] = real[k + i] - tr
                        imag[k + i + halfM] = imag[k + i] - ti
                        real[k + i] += tr
                        imag[k + i] += ti
                    }
                }
                m *= 2
            }
        }
    }

    override fun onDestroy() {
        running.set(false)
        session?.close()
        env.close()
        super.onDestroy()
    }
}
