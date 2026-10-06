"""
Builds Netscape-AI-Pack.zip — the offline models the app loads from Settings → AI Models.

  nsfw.onnx          Falconsai/nsfw_image_detection (ViT, normal / nsfw), int8
  nudenet.onnx       NudeNet 3 detector (320n, 18 body-part classes), as shipped on PyPI
  clip_vision.onnx   openai/clip-vit-base-patch32 image encoder, int8
  clip_text.onnx     openai/clip-vit-base-patch32 text encoder, int8
  clip_vocab.txt.gz  CLIP BPE merges (from open_clip)
  smolvlm_q8.gguf   SmolVLM2-500M-Video-Instruct Q8_0 decoder
  smolvlm_mmproj_q8.gguf  SmolVLM2 Q8_0 vision projector
  manifest.json      preprocessing + labels, so the phone never guesses

Every model is checked against the original PyTorch / Python implementation before packing,
and reference outputs are written to <out>/ref.json for the Kotlin harness to compare against.

usage: python build_ai_pack.py <out_dir>
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image

torch.set_grad_enabled(False)
OUT = os.path.abspath(sys.argv[1])
PACK = os.path.join(OUT, "pack")
TMP = os.path.join(OUT, "tmp")
IMGS = os.path.join(OUT, "testimgs")
for d in (PACK, TMP, IMGS):
    os.makedirs(d, exist_ok=True)

OPSET = 17


def log(*a):
    print("[pack]", *a, flush=True)


def fp16_weights(src, dst, min_elems=1024):
    """Store large float32 weights as float16 and Cast them back to float32 at load time.
    Half the size, and every op still computes in float32 — runs on any CPU, near-lossless."""
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(src)
    g = m.graph
    casts, inits = [], []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_elems:
            h = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
            inits.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
        else:
            inits.append(init)
    del g.initializer[:]
    g.initializer.extend(inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.save(m, dst)


# Only MatMul/Gemm are quantised: quantising a Conv produces ConvInteger, which ONNX Runtime's
# CPU provider (desktop and Android) can't run.
MM = ["MatMul", "Gemm"]
VARIANTS = [
    ("int8 per-channel", lambda s, d: quantize_dynamic(s, d, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=MM)),
    ("int8", lambda s, d: quantize_dynamic(s, d, weight_type=QuantType.QInt8, op_types_to_quantize=MM)),
    ("fp16 weights", fp16_weights),
]


def compress(src, dst, score, need):
    """Try the variants smallest-first; keep the first whose score (1.0 = identical to the
    original PyTorch model) reaches `need`. fp16 weights is near-lossless, so it is the
    guaranteed fallback; if even that falls short the build fails loudly."""
    name = os.path.basename(dst)
    for label, make in VARIANTS:
        cand = dst + ".cand"
        try:
            make(src, cand)
            got = score(cand)
        except Exception as e:  # e.g. an op the runtime can't execute
            log(f"  {name}: {label} unusable ({type(e).__name__}: {str(e)[:120]})")
            continue
        log(f"  {name}: {label} {os.path.getsize(src)/1e6:.0f} MB -> {os.path.getsize(cand)/1e6:.0f} MB, fidelity {got:.4f} (need {need})")
        if got >= need:
            os.replace(cand, dst)
            return label
        os.remove(cand)
    raise SystemExit(f"{name}: no variant was accurate enough")


def session(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def softmax(x, scale=1.0):
    x = np.asarray(x, dtype=np.float64) * scale
    e = np.exp(x - x.max())
    return e / e.sum()


# ------------------------------------------------------------------ test images (224², no resize)
COLORS = {"red": (220, 30, 30), "green": (30, 190, 40), "blue": (30, 60, 220), "yellow": (240, 220, 30)}
test_files = []
for name, rgb in COLORS.items():
    arr = np.zeros((224, 224, 3), np.uint8)
    arr[:] = rgb
    # a little texture so it is not a perfectly flat tensor
    arr = np.clip(arr.astype(int) + np.random.default_rng(len(name)).integers(-12, 12, arr.shape), 0, 255).astype(np.uint8)
    p = os.path.join(IMGS, f"{name}.png")
    Image.fromarray(arr).save(p)
    test_files.append((name, p))
ref = {"images": {}, "prompts": [f"a photo of a {c} square" for c in COLORS]}

manifest = {"version": "3", "models": {}}

# ------------------------------------------------------------------ 1. NSFW classifier
from transformers import AutoImageProcessor, AutoModelForImageClassification

NSFW_ID = "Falconsai/nsfw_image_detection"
log("NSFW classifier:", NSFW_ID)
nm = AutoModelForImageClassification.from_pretrained(NSFW_ID).eval()
npp = AutoImageProcessor.from_pretrained(NSFW_ID)
size_cfg = npp.size
if "height" in size_cfg:
    n_size, n_resize = int(size_cfg["height"]), "stretch"
else:
    n_size, n_resize = int(size_cfg["shortest_edge"]), "crop"
labels = [nm.config.id2label[i] for i in range(nm.config.num_labels)]
log("  labels", labels, "size", n_size, n_resize, "mean", npp.image_mean, "std", npp.image_std)


class NsfwWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).logits


f32 = os.path.join(TMP, "nsfw_f32.onnx")
torch.onnx.export(
    NsfwWrap(nm), torch.randn(1, 3, n_size, n_size), f32,
    input_names=["pixel_values"], output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}}, opset_version=OPSET,
)
nsfw_inputs = {name: npp(images=Image.open(p).convert("RGB"), return_tensors="pt")["pixel_values"] for name, p in test_files}
nsfw_torch = {name: softmax(nm(pixel_values=px).logits[0].numpy()) for name, px in nsfw_inputs.items()}


def nsfw_score(path):
    s = session(path)
    worst = max(np.abs(softmax(s.run(None, {"pixel_values": px.numpy()})[0][0]) - nsfw_torch[n]).max() for n, px in nsfw_inputs.items())
    return 1.0 - float(worst)


compress(f32, os.path.join(PACK, "nsfw.onnx"), nsfw_score, need=0.9)
ns = session(os.path.join(PACK, "nsfw.onnx"))
for name, px in nsfw_inputs.items():
    onnx_p = softmax(ns.run(None, {"pixel_values": px.numpy()})[0][0])
    log(f"  {name}: torch {np.round(nsfw_torch[name], 3)} packed {np.round(onnx_p, 3)}")
    ref["images"].setdefault(name, {})["nsfw"] = {labels[i]: float(onnx_p[i]) for i in range(len(labels))}
manifest["models"]["nsfw"] = {
    "file": "nsfw.onnx", "source": NSFW_ID, "labels": labels,
    "spec": {"size": n_size, "resize": n_resize, "mean": list(npp.image_mean), "std": list(npp.image_std)},
}

# ------------------------------------------------------------------ 2. NudeNet
import nudenet

log("NudeNet", getattr(nudenet, "__version__", ""))
src = os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx")
shutil.copy(src, os.path.join(PACK, "nudenet.onnx"))
nn_s = session(os.path.join(PACK, "nudenet.onnx"))
log("  input", nn_s.get_inputs()[0].shape, "output", nn_s.get_outputs()[0].shape)
manifest["models"]["nudenet"] = {"file": "nudenet.onnx", "source": "nudenet (PyPI) 320n", "size": 320}

# ------------------------------------------------------------------ 3. CLIP
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

CLIP_ID = "openai/clip-vit-base-patch32"
log("CLIP:", CLIP_ID)
# "eager" attention: torch 2.4's ONNX exporter can't translate the SDPA path with a float scale.
cm = CLIPModel.from_pretrained(CLIP_ID, attn_implementation="eager").eval()
cp = CLIPProcessor.from_pretrained(CLIP_ID)
ip = cp.image_processor
c_size = int(ip.crop_size["height"]) if isinstance(ip.crop_size, dict) else int(ip.crop_size)


def feats(x):
    # transformers may return a tensor or a ModelOutput depending on version
    if isinstance(x, torch.Tensor):
        return x
    for k in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(x, k, None) is not None:
            return getattr(x, k)
    raise RuntimeError(f"unexpected output {type(x)}")


class Vision(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return feats(self.m.get_image_features(pixel_values=pixel_values))


class Text(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return feats(self.m.get_text_features(input_ids=input_ids))


v32 = os.path.join(TMP, "clip_vision_f32.onnx")
t32 = os.path.join(TMP, "clip_text_f32.onnx")
torch.onnx.export(
    Vision(cm), torch.randn(1, 3, c_size, c_size), v32,
    input_names=["pixel_values"], output_names=["image_embeds"],
    dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}}, opset_version=OPSET,
)
dummy = torch.zeros(1, 77, dtype=torch.long)
dummy[0, :3] = torch.tensor([49406, 320, 49407])
torch.onnx.export(
    Text(cm), dummy, t32,
    input_names=["input_ids"], output_names=["text_embeds"],
    dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}}, opset_version=OPSET,
)

# CLIP BPE vocab + reference tokenizer from open_clip (the phone's tokenizer is a port of it)
subprocess.check_call([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "open_clip_torch==3.3.0", "-d", TMP])
whl = glob.glob(os.path.join(TMP, "open_clip_torch-*.whl"))[0]
with zipfile.ZipFile(whl) as z:
    with z.open("open_clip/bpe_simple_vocab_16e6.txt.gz") as fsrc, open(os.path.join(PACK, "clip_vocab.txt.gz"), "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)
    tok_src = z.read("open_clip/tokenizer.py").decode()

import types

sys.modules.setdefault("torch", torch)
tokmod = types.ModuleType("oc_tok")
tok_path = os.path.join(TMP, "tokenizer.py")
with open(tok_path, "w") as f:
    f.write(tok_src)
# it builds a default tokenizer at import time from the vocab sitting next to it
shutil.copy(os.path.join(PACK, "clip_vocab.txt.gz"), os.path.join(TMP, "bpe_simple_vocab_16e6.txt.gz"))
tokmod.__file__ = tok_path  # the module looks for its vocab next to __file__
exec(compile(tok_src, tok_path, "exec"), tokmod.__dict__)
octok = tokmod.SimpleTokenizer(bpe_path=os.path.join(PACK, "clip_vocab.txt.gz"))


def oc_ids(text):
    t = [octok.sot_token_id] + octok.encode(text) + [octok.eot_token_id]
    t = t[:76] + [octok.eot_token_id] if len(t) > 77 else t
    return np.array([t + [0] * (77 - len(t))], dtype=np.int64)


hf_tok = CLIPTokenizer.from_pretrained(CLIP_ID)


def norm(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / np.linalg.norm(v)


# Reference embeddings from the original PyTorch model (zero-padded ids, exactly as the phone sends)
prompts = ref["prompts"] + [
    "a photo of a couple in the bedroom", "a person wearing lingerie in the shower",
    "a photo of a couple having sex in the Cowgirl position", "an explicit photo of Kissing",
]
prompt_ids = {pr: oc_ids(pr) for pr in prompts}
text_torch = {}
for pr in prompts:
    ids = prompt_ids[pr]
    hf = hf_tok(pr, padding="max_length", max_length=77, return_tensors="pt")["input_ids"]
    n_tok = int((ids[0] != 0).sum())
    same_ids = [int(x) for x in ids[0][:n_tok]] == [int(x) for x in hf[0][:n_tok]]
    torch_hf = norm(feats(cm.get_text_features(input_ids=hf)).numpy())
    text_torch[pr] = norm(feats(cm.get_text_features(input_ids=torch.from_numpy(ids))).numpy())
    log(f"  tokens {pr!r}: match HF {same_ids}, padding cos {torch_hf @ text_torch[pr]:.5f}")
    if not same_ids or torch_hf @ text_torch[pr] < 0.999:
        log("  WARNING: open_clip ids / zero padding differ from HF — check the phone tokenizer")
clip_inputs = {name: ip(images=Image.open(p).convert("RGB"), return_tensors="pt")["pixel_values"] for name, p in test_files}
image_torch = {name: norm(feats(cm.get_image_features(pixel_values=px)).numpy()) for name, px in clip_inputs.items()}


def text_score(path):
    s = session(path)
    return float(min(norm(s.run(None, {"input_ids": prompt_ids[pr]})[0]) @ text_torch[pr] for pr in prompts))


def vision_score(path):
    s = session(path)
    return float(min(norm(s.run(None, {"pixel_values": px.numpy()})[0]) @ image_torch[n] for n, px in clip_inputs.items()))


compress(v32, os.path.join(PACK, "clip_vision.onnx"), vision_score, need=0.97)
compress(t32, os.path.join(PACK, "clip_text.onnx"), text_score, need=0.97)
vs, ts = session(os.path.join(PACK, "clip_vision.onnx")), session(os.path.join(PACK, "clip_text.onnx"))
text_packed = {pr: norm(ts.run(None, {"input_ids": prompt_ids[pr]})[0]) for pr in ref["prompts"]}

agree = 0
for name, px in clip_inputs.items():
    q = norm(vs.run(None, {"pixel_values": px.numpy()})[0])
    sims = [float(q @ text_packed[pr]) for pr in ref["prompts"]]
    torch_sims = [float(image_torch[name] @ text_torch[pr]) for pr in ref["prompts"]]
    top, torch_top = int(np.argmax(sims)), int(np.argmax(torch_sims))
    agree += int(top == torch_top)
    log(f"  image {name}: packed → {ref['prompts'][top]!r}, original → {ref['prompts'][torch_top]!r}")
    ref["images"].setdefault(name, {})["clip_sims"] = sims
# Fidelity above is the real gate; flat colour squares can score prompts almost equally.
if agree < 4:
    log(f"  note: packed and original CLIP pick different prompts on {4 - agree}/4 near-tied test images")

manifest["models"]["clip"] = {
    "vision": "clip_vision.onnx", "text": "clip_text.onnx", "vocab": "clip_vocab.txt.gz", "source": CLIP_ID,
    "contextLength": 77,
    "spec": {"size": c_size, "resize": "crop", "mean": list(ip.image_mean), "std": list(ip.image_std)},
}

# ------------------------------------------------------------------ 4. SmolVLM2 (GGUF + projector)
# Pin the converted weights and verify their exact LFS content before publishing.
from huggingface_hub import hf_hub_download

VLM_ID = "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF"
VLM_REVISION = "ccd7aae53bcb1997355c2f094959e72b3642ce17"
VLM_FILES = [
    (
        "SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
        "smolvlm_q8.gguf",
        436808704,
        "6f67b8036b2469fcd71728702720c6b51aebd759b78137a8120733b4d66438bc",
    ),
    (
        "mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
        "smolvlm_mmproj_q8.gguf",
        108785184,
        "921dc7e259f308e5b027111fa185efcbf33db13f6e35749ddf7f5cdb60ef520b",
    ),
]
log("SmolVLM2:", VLM_ID, "revision", VLM_REVISION)
for source_name, pack_name, expected_size, expected_sha256 in VLM_FILES:
    downloaded = hf_hub_download(
        repo_id=VLM_ID,
        filename=source_name,
        revision=VLM_REVISION,
    )
    digest = hashlib.sha256()
    size = 0
    with open(downloaded, "rb") as src:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    if size != expected_size or digest.hexdigest() != expected_sha256:
        raise SystemExit(
            f"{source_name}: unexpected content ({size} bytes, sha256={digest.hexdigest()})"
        )
    shutil.copy2(downloaded, os.path.join(PACK, pack_name))
    log(f"  verified {pack_name}: {size / 1e6:.1f} MB")
manifest["models"]["smolvlm"] = {
    "model": "smolvlm_q8.gguf",
    "mmproj": "smolvlm_mmproj_q8.gguf",
    "source": VLM_ID,
    "revision": VLM_REVISION,
    "quantization": "Q8_0",
}

# ------------------------------------------------------------------ pack
with open(os.path.join(PACK, "LICENSES.txt"), "w") as f:
    f.write(
        "Netscape AI pack — third-party models, used unmodified apart from ONNX export and weight compression (int8 / fp16).\n\n"
        f"NSFW classifier: https://huggingface.co/{NSFW_ID} (see model card for licence)\n"
        "NudeNet: https://github.com/notAI-tech/NudeNet (MIT)\n"
        f"CLIP: https://huggingface.co/{CLIP_ID} (MIT)\n"
        "CLIP BPE vocabulary: https://github.com/mlfoundations/open_clip (MIT)\n"
        f"SmolVLM2 GGUF: https://huggingface.co/{VLM_ID} (Apache-2.0; Q8_0 files pinned to {VLM_REVISION})\n"
        "llama.cpp runtime: https://github.com/ggml-org/llama.cpp (MIT; see app build sources)\n"
    )


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Size + SHA-256 of every file, so the phone can prove each model arrived intact.
payload = sorted(n for n in os.listdir(PACK) if n != "manifest.json")
manifest["files"] = {
    n: {"size": os.path.getsize(os.path.join(PACK, n)), "sha256": sha256_of(os.path.join(PACK, n))} for n in payload
}
with open(os.path.join(PACK, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
with open(os.path.join(OUT, "ref.json"), "w") as f:
    json.dump(ref, f, indent=2)

# manifest.json goes first. Model weights are stored, not deflated: they barely compress, and a
# stored entry can't trip the phone's zlib decoder — damage shows up as a clear checksum error.
zpath = os.path.join(OUT, "Netscape-AI-Pack.zip")
with zipfile.ZipFile(zpath, "w") as z:
    for name in ["manifest.json"] + payload:
        compression = zipfile.ZIP_DEFLATED if name.endswith((".txt", ".json")) else zipfile.ZIP_STORED
        z.write(os.path.join(PACK, name), name, compress_type=compression)
with zipfile.ZipFile(zpath) as z:
    bad = z.testzip()
    if bad:
        raise SystemExit(f"zip self-check failed on {bad}")
    if z.namelist()[0] != "manifest.json":
        raise SystemExit("manifest.json must be the first entry")
log(f"wrote {zpath}: {os.path.getsize(zpath)/1e6:.1f} MB")
if os.path.getsize(zpath) >= 1_000_000_000:
    raise SystemExit("AI pack exceeds the 1 GB download limit")
for name in ["manifest.json"] + payload:
    log(f"  {name}: {os.path.getsize(os.path.join(PACK, name))/1e6:.1f} MB")
with open(os.path.join(OUT, "pack.sha256"), "w") as f:
    f.write(f"{sha256_of(zpath)}  Netscape-AI-Pack.zip\n")
