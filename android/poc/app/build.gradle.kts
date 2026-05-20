plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dk.ku.misophonia"
    compileSdk = 35

    defaultConfig {
        applicationId = "dk.ku.misophonia"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.25.1")
}
