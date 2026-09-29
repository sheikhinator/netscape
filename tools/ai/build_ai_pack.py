"""
Builds Netscape-AI-Pack.zip — the offline models the app loads from Settings → AI Models.

  nsfw.onnx          Falconsai/nsfw_image_detection (ViT, normal / nsfw), int8
  nudenet.onnx       NudeNet 3 detector (320n, 18 body-part classes), as shipped on PyPI
  clip_vision.onnx   openai/clip-vit-base-patch32 image encoder, int8
  clip_text.onnx     openai/clip-vit-base-patch32 text encoder, int8
  clip_vocab.txt.gz  CLIP BPE merges (from open_clip)
  manifest.json      preprocessing + labels, so the phone never guesses

Every model is checked against the original PyTorch / Python implementation before packing,
and reference outputs are written to <out>/ref.json for the Kotlin harness to compare against.

usage: python build_ai_pack.py <out_dir>
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
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


def quantize(src, dst):
    # Only MatMul/Gemm: quantising the ViT patch Conv produces ConvInteger, which ONNX Runtime's
    # CPU provider (desktop and Android) can't run. Nearly all the weights are in MatMuls anyway.
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    log(f"  {os.path.basename(dst)}: {os.path.getsize(src)/1e6:.1f} MB -> {os.path.getsize(dst)/1e6:.1f} MB")


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

manifest = {"version": "1", "models": {}}

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
quantize(f32, os.path.join(PACK, "nsfw.onnx"))
ns = session(os.path.join(PACK, "nsfw.onnx"))
for name, p in test_files:
    px = npp(images=Image.open(p).convert("RGB"), return_tensors="pt")["pixel_values"]
    torch_p = softmax(nm(pixel_values=px).logits[0].numpy())
    onnx_p = softmax(ns.run(None, {"pixel_values": px.numpy()})[0][0])
    log(f"  {name}: torch {np.round(torch_p, 3)} int8 {np.round(onnx_p, 3)}")
    assert np.abs(torch_p - onnx_p).max() < 0.15, "int8 NSFW model drifted too far from torch"
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
quantize(v32, os.path.join(PACK, "clip_vision.onnx"))
quantize(t32, os.path.join(PACK, "clip_text.onnx"))

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
vs, ts = session(os.path.join(PACK, "clip_vision.onnx")), session(os.path.join(PACK, "clip_text.onnx"))


def norm(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / np.linalg.norm(v)


prompts = ref["prompts"] + ["a photo of a couple in the bedroom", "a person wearing lingerie in the shower"]
text_int8 = {}
for pr in prompts:
    ids = oc_ids(pr)
    hf = hf_tok(pr, padding="max_length", max_length=77, return_tensors="pt")["input_ids"]
    n_tok = int((ids[0] != 0).sum())
    same_ids = [int(x) for x in ids[0][:n_tok]] == [int(x) for x in hf[0][:n_tok]]
    torch_hf = norm(feats(cm.get_text_features(input_ids=hf)).numpy())
    torch_oc = norm(feats(cm.get_text_features(input_ids=torch.from_numpy(ids))).numpy())
    q = norm(ts.run(None, {"input_ids": ids})[0])
    log(f"  text {pr!r}: ids match HF {same_ids}, hf-vs-zero-pad cos {torch_hf @ torch_oc:.5f}, int8 cos {q @ torch_oc:.4f}")
    if not same_ids or torch_hf @ torch_oc < 0.999:
        log("  WARNING: open_clip ids / zero padding differ from HF — check the phone tokenizer")
    assert q @ torch_oc > 0.9, "int8 text encoder drifted"
    text_int8[pr] = q

correct = 0
for name, p in test_files:
    px = ip(images=Image.open(p).convert("RGB"), return_tensors="pt")["pixel_values"]
    torch_v = norm(feats(cm.get_image_features(pixel_values=px)).numpy())
    q = norm(vs.run(None, {"pixel_values": px.numpy()})[0])
    sims = [float(q @ text_int8[pr]) for pr in ref["prompts"]]
    probs = softmax(sims, 100)
    top = ref["prompts"][int(np.argmax(probs))]
    correct += int(name in top)
    log(f"  image {name}: int8 cos {q @ torch_v:.4f}; zero-shot → {top!r} ({probs.max():.2f})")
    assert q @ torch_v > 0.9, "int8 vision encoder drifted"
    ref["images"].setdefault(name, {})["clip_sims"] = sims
assert correct >= 3, f"CLIP zero-shot sanity check failed ({correct}/4 colours right)"

manifest["models"]["clip"] = {
    "vision": "clip_vision.onnx", "text": "clip_text.onnx", "vocab": "clip_vocab.txt.gz", "source": CLIP_ID,
    "contextLength": 77,
    "spec": {"size": c_size, "resize": "crop", "mean": list(ip.image_mean), "std": list(ip.image_std)},
}

# ------------------------------------------------------------------ pack
with open(os.path.join(PACK, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
with open(os.path.join(PACK, "LICENSES.txt"), "w") as f:
    f.write(
        "Netscape AI pack — third-party models, used unmodified apart from ONNX export and int8 quantisation.\n\n"
        f"NSFW classifier: https://huggingface.co/{NSFW_ID} (see model card for licence)\n"
        "NudeNet: https://github.com/notAI-tech/NudeNet (MIT)\n"
        f"CLIP: https://huggingface.co/{CLIP_ID} (MIT)\n"
        "CLIP BPE vocabulary: https://github.com/mlfoundations/open_clip (MIT)\n"
    )
with open(os.path.join(OUT, "ref.json"), "w") as f:
    json.dump(ref, f, indent=2)

zpath = os.path.join(OUT, "Netscape-AI-Pack.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for name in sorted(os.listdir(PACK)):
        z.write(os.path.join(PACK, name), name)
log(f"wrote {zpath}: {os.path.getsize(zpath)/1e6:.1f} MB")
for name in sorted(os.listdir(PACK)):
    log(f"  {name}: {os.path.getsize(os.path.join(PACK, name))/1e6:.1f} MB")
