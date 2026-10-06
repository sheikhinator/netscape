// Runs the app's WriterPrompt + StoryWriter (and, through JNI, writer_jni.cpp) on a desktop JVM.
plugins {
    kotlin("jvm") version "1.9.24"
    application
}
repositories { mavenCentral() }
val aiSrc: String = (findProperty("aiSrc") as String?)
    ?: error("pass -PaiSrc=<path to app/src/main/java/com/netscape/vault/ai>")
sourceSets {
    main {
        kotlin.srcDirs(aiSrc, "src")
        kotlin.include("**/WriterPrompt.kt", "**/StoryWriter.kt", "**/WriterHarness.kt")
    }
}
application { mainClass.set("WriterHarnessKt") }
