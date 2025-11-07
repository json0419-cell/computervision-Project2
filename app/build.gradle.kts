plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.example.mobilefoodfreshness"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.mobilefoodfreshness"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildFeatures { viewBinding = true }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    packaging {
        resources.excludes.add("META-INF/*")
    }
}

// ✅ Ensure Kotlin uses the same JDK as Java
kotlin {
    jvmToolchain(21)
}

dependencies {
    val camerax = "1.3.4"

    // ✅ CameraX dependencies
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // ✅ Fragment KTX (needed for commit{} and fragment extensions)
    implementation("androidx.fragment:fragment-ktx:1.6.2")

    // ✅ TensorFlow Lite dependencies (must match versions)
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0")

    // 👇 Recommended: upgrade task-vision to 0.4.5 to fix BaseOptions issue
    implementation("org.tensorflow:tensorflow-lite-task-vision:0.4.4")

    // ✅ OpenCV (QuickBird packaged AAR)
    implementation("com.quickbirdstudios:opencv:4.5.3")

    // ✅ AndroidX + Material essentials
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
}
