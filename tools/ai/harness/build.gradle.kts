// Runs the app's own com.netscape.vault.ai code on a desktop JVM against a freshly built AI pack.
// The same ai.onnxruntime Java API ships in onnxruntime-android, so this exercises the phone path.
plugins {
    kotlin("jvm") version "1.9.24"
    application
}
repositories { mavenCentral() }
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime:1.19.2")
    implementation("org.json:json:20240303")
}
val aiSrc: String = (findProperty("aiSrc") as String?)
    ?: error("pass -PaiSrc=<path to app/src/main/java/com/netscape/vault/ai>")
sourceSets {
    main {
        kotlin.srcDirs(aiSrc, "src")
        // AiService is the Android glue (Bitmap, Vault); everything else is plain Kotlin.
        kotlin.exclude("**/AiService.kt", "**/WriterService.kt")
    }
}
application { mainClass.set("HarnessKt") }
