import com.netscape.vault.ai.*
import org.json.JSONObject
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.system.exitProcess

fun load(f: File): Pixels {
    val img = ImageIO.read(f)
    return Pixels(img.width, img.height, img.getRGB(0, 0, img.width, img.height, null, 0, img.width))
}

/** usage: pack <packDir> <ref.json> <imgDir> */
fun main(args: Array<String>) {
    val pack = File(args[1])
    val ref = JSONObject(File(args[2]).readText())
    val imgs = File(args[3])
    val models = AiModels(pack)
    var failures = 0
    fun check(ok: Boolean, msg: String) {
        println((if (ok) "  ok   " else "  FAIL ") + msg)
        if (!ok) failures++
    }

    check(models.hasClassifier && models.hasNudeNet && models.hasClip, "manifest lists all three models")
    val prompts = ref.getJSONArray("prompts").let { a -> (0 until a.length()).map { a.getString(it) } }
    val clip = models.clip()!!
    val textEmb = prompts.associateWith { clip.embedText(it) }
    val images = ref.getJSONObject("images")

    for (name in images.keys().asSequence().sorted()) {
        println(name)
        val px = load(File(imgs, "$name.png"))
        val r = images.getJSONObject(name)

        val probs = models.classifier()!!.classify(px)
        val want = r.getJSONObject("nsfw")
        for (label in want.keys()) {
            val d = abs((probs[label] ?: -1f) - want.getDouble(label).toFloat())
            check(d < 0.02f, "nsfw[$label] kotlin=${"%.4f".format(probs[label])} python=${"%.4f".format(want.getDouble(label))}")
        }

        val img = clip.embedImage(px)
        val sims = prompts.map { ClipModel.dot(img, textEmb.getValue(it)) }
        val wantSims = r.getJSONArray("clip_sims").let { a -> (0 until a.length()).map { a.getDouble(it).toFloat() } }
        sims.indices.forEach { i ->
            check(abs(sims[i] - wantSims[i]) < 0.01f, "clip sim '${prompts[i]}' kotlin=${"%.4f".format(sims[i])} python=${"%.4f".format(wantSims[i])}")
        }
        check(sims.indexOf(sims.max()) == wantSims.indexOf(wantSims.max()), "same zero-shot winner")

        val dets = models.nudeNet()!!.detect(px)
        println("  nudenet: ${dets.size} detections")

        val suggestions = AiTagger.suggest(
            listOf(FrameResult(probs, dets, img)),
            mapOf("setting" to listOf("Bedroom", "Shower", "Beach", "Kitchen"), "heat" to listOf("Mild", "Sexy", "Explicit", "Animated")),
            emptySet(), 1
        ) { clip.embedText(it) }
        println("  tagger: " + suggestions.joinToString { "${it.tag}(${"%.2f".format(it.confidence)})" })
    }
    models.close()
    println(if (failures == 0) "ALL CHECKS PASSED" else "$failures CHECK(S) FAILED")
    exitProcess(if (failures == 0) 0 else 1)
}
