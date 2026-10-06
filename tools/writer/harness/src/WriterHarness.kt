import com.netscape.vault.ai.StoryWriter
import com.netscape.vault.ai.WriterPrompt
import com.netscape.vault.ai.WriterPrompt.Voice
import kotlin.system.exitProcess

/** usage: WriterHarness <writer.gguf>  (with -Dnetscape.nativeLib=<host libnetscape_writer_host.so>) */
fun main(args: Array<String>) {
    val samples = listOf(
        WriterPrompt.Item(true, 9 * 60_000L, "Full HD", 4, linkedMapOf(
            "Heat Level" to listOf("Explicit"), "Who's In It" to listOf("Couple"),
            "Positions" to listOf("Doggy Style", "Missionary"), "Acts" to listOf("Oral", "Dirty Talk"),
            "Setting & Place" to listOf("Shower"), "Mood & Vibe" to listOf("Rough")), emptyList()),
        WriterPrompt.Item(false, 0, "4K", 5, linkedMapOf(
            "Heat Level" to listOf("Sexy"), "Who's In It" to listOf("Solo Female"),
            "Outfit & Lingerie" to listOf("Stockings", "Heels"), "Body Focus" to listOf("Legs"),
            "Camera & Style" to listOf("Mirror Shot")), emptyList()),
        WriterPrompt.Item(true, 3 * 60_000L, "HD", 0, linkedMapOf(
            "Spicy Categories" to listOf("Quickies"), "Age" to listOf("Adult"),
            "Positions" to listOf("Against the Wall"), "Setting & Place" to listOf("Car"),
            "Mood & Vibe" to listOf("Spontaneous")), emptyList()),
        WriterPrompt.Item(true, 22 * 60_000L, "4K", 5, emptyMap(), emptyList())
    )
    val refusal = Regex("""(?i)\b(I can(no|')t|I cannot|I'm sorry|I am sorry|as an AI|I won't|not able to (help|assist)|inappropriate)\b""")
    val writer = StoryWriter(args[0], threads = Runtime.getRuntime().availableProcessors().coerceAtMost(4))
    var runs = 0
    var refusals = 0
    var malformed = 0
    var rejected = 0
    for (voice in Voice.values()) {
        for ((i, item) in samples.withIndex()) {
            if (voice == Voice.MINIMAL && i > 1) continue
            val raw = writer.write(WriterPrompt.build(item, voice, 2), maxTokens = 200, seed = 42 + i)
            val st = writer.lastStats!!
            val parsed = WriterPrompt.parse(raw)
            // Same post-processing the app applies before anything is shown.
            val accepted = WriterPrompt.isAcceptable(parsed)
            val r = WriterPrompt.enforce(parsed, item, 2)
            runs++
            val ok = r.title.isNotBlank() && r.description.length >= 30
            if (!ok) malformed++
            if (!accepted) rejected++
            if (refusal.containsMatchIn(raw)) refusals++
            val tps = if (st.millis > 0) st.tokens * 1000.0 / st.millis else 0.0
            val flags = (if (!ok) " · MALFORMED" else "") + (if (!accepted) " · REJECTED (app rewrites it)" else "")
            println("=== $voice · sample ${i + 1} · ${st.tokens} tokens · first token ${st.firstTokenMillis} ms · ${"%.1f".format(tps)} tok/s$flags")
            println("TITLE: ${r.title}")
            println("DESCRIPTION: ${r.description}")
        }
    }
    writer.close()
    println("runs=$runs malformed=$malformed refusals=$refusals rejected=$rejected")
    // A couple of odd outputs from a 1.5B model are tolerable; refusals or broken formatting across the board are not.
    // Rejected takes are rewritten by the app; if it happens often the user waits on retries.
    if (refusals > 1 || malformed > runs / 5 || rejected > runs / 5) {
        println("WRITER CHECK FAILED")
        exitProcess(1)
    }
    println("WRITER CHECK PASSED")
}
