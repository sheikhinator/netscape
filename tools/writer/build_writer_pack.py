"""
Builds Netscape-Writer-Pack.zip — the optional offline writer the app loads from
Settings → AI writer.

  writer.gguf     Qwen2.5-1.5B-Instruct (abliterated), Q4_K_M, converted with the same llama.cpp
                  commit the app is built against
  manifest.json   source, revision, quantisation, and size + SHA-256 of every file
  LICENSES.txt

usage: python build_writer_pack.py <llama.cpp dir> <llama-quantize binary> <out dir>
"""
import hashlib
import json
import os
import subprocess
import sys
import zipfile

from huggingface_hub import HfApi, snapshot_download

LLAMA, QUANTIZE, OUT = (os.path.abspath(a) for a in sys.argv[1:4])
PACK = os.path.join(OUT, "pack")
TMP = os.path.join(OUT, "tmp")
os.makedirs(PACK, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

# Uncensored ("abliterated") Qwen2.5-1.5B-Instruct — Apache-2.0, like the base model.
CANDIDATES = ["huihui-ai/Qwen2.5-1.5B-Instruct-abliterated"]
QUANT = "Q4_K_M"


def log(*a):
    print("[writer]", *a, flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


api = HfApi()
repo = revision = None
for cand in CANDIDATES:
    try:
        info = api.model_info(cand)
        repo, revision = cand, info.sha
        break
    except Exception as e:  # not found / gated
        log(f"{cand}: unavailable ({type(e).__name__})")
if not repo:
    raise SystemExit("No writer model could be downloaded")
log(f"model {repo} @ {revision}")

src = snapshot_download(
    repo_id=repo, revision=revision,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab.json"],
)
f16 = os.path.join(TMP, "writer-f16.gguf")
subprocess.check_call([sys.executable, os.path.join(LLAMA, "convert_hf_to_gguf.py"), src, "--outtype", "f16", "--outfile", f16])
out_model = os.path.join(PACK, "writer.gguf")
subprocess.check_call([QUANTIZE, f16, out_model, QUANT])
os.remove(f16)
log(f"writer.gguf: {os.path.getsize(out_model) / 1e6:.1f} MB ({QUANT})")

with open(os.path.join(PACK, "LICENSES.txt"), "w") as f:
    f.write(
        "Netscape Writer pack\n\n"
        f"Writer model: https://huggingface.co/{repo} (revision {revision}) — derived from Qwen/Qwen2.5-1.5B-Instruct, Apache-2.0.\n"
        f"Converted to GGUF and quantised to {QUANT} with llama.cpp (MIT).\n"
    )
payload = sorted(n for n in os.listdir(PACK) if n != "manifest.json")
manifest = {
    "version": "1",
    "kind": "writer",
    "models": {"writer": {"file": "writer.gguf", "source": repo, "revision": revision, "quantization": QUANT,
                          "template": "chatml"}},
    "files": {n: {"size": os.path.getsize(os.path.join(PACK, n)), "sha256": sha256_of(os.path.join(PACK, n))} for n in payload},
}
with open(os.path.join(PACK, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

zpath = os.path.join(OUT, "Netscape-Writer-Pack.zip")
with zipfile.ZipFile(zpath, "w") as z:
    for name in ["manifest.json"] + payload:
        ct = zipfile.ZIP_DEFLATED if name.endswith((".txt", ".json")) else zipfile.ZIP_STORED
        z.write(os.path.join(PACK, name), name, compress_type=ct)
with zipfile.ZipFile(zpath) as z:
    if z.testzip() or z.namelist()[0] != "manifest.json":
        raise SystemExit("zip self-check failed")
with open(os.path.join(OUT, "pack.sha256"), "w") as f:
    f.write(f"{sha256_of(zpath)}  Netscape-Writer-Pack.zip\n")
log(f"wrote {zpath}: {os.path.getsize(zpath) / 1e6:.1f} MB")
