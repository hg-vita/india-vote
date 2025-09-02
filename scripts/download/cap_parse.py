#!/usr/bin/env python3
# cap_parse.py — OCR via Ollama REST with light preprocessing (no deskew), Windows-friendly
# Usage:
#   python cap_parse.py "<image_path>" --model llava:7b --outdir ocr_out --timeout 180
#
# Tips:
#   1) First call can be slow (model load). Warm up once: `ollama run llava:7b "ready?"`
#   2) On CPU-only machines, prefer llava:7b over 13b.
#   3) If slow, use --max-side 512 and --timeout 240 for the first run.

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llava:7b"  # swap to llava:13b if you have a strong GPU
DEFAULT_PROMPT = (
    "Extract ALL visible text from this image. "
    "Return ONLY the transcribed text in reading order. "
    "If any part is unclear, include your best guess marked with [uncertain]. "
    "Do not describe the image; just transcribe."
)

def check_ollama(host: str) -> str:
    """Return '' if healthy, else error message."""
    url = f"{host.rstrip('/')}/api/tags"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return ""
        return f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return f"Could not reach Ollama at {url}: {e}"

def preprocess(img_path: Path, max_side: int = 640, contrast_boost: float = 1.8, quality: int = 70) -> bytes:
    """
    Fast, robust preprocessing (no deskew/rotation):
      - grayscale + autocontrast
      - gentle sharpen + contrast boost
      - resize so longest side <= max_side
      - JPEG compress (quality 70) for speed over REST
    Returns JPEG bytes.
    """
    img = Image.open(img_path).convert("RGB")
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    gray = ImageEnhance.Contrast(gray).enhance(contrast_boost)

    w, h = gray.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        gray = gray.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    buf = io.BytesIO()
    gray.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def vlm_ocr(image_path: Path, model: str, host: str, prompt: str, timeout_s: int,
            max_side: int, num_ctx: int, num_predict: int) -> str:
    """Call Ollama /api/generate with base64 image and return the text (or an error/timeout message)."""
    img_bytes = preprocess(image_path, max_side=max_side, contrast_boost=1.8, quality=70)
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64(img_bytes)],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
            "top_p": 0.9,
        },
    }
    url = f"{host.rstrip('/')}/api/generate"

    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except requests.exceptions.ReadTimeout:
        return "[timeout] Model took too long. Try: warm-up (`ollama run llava:7b \"ready?\"`), reduce --max-side, or increase --timeout."
    except Exception as e:
        return f"[error] Could not reach Ollama: {e}"

    if resp.status_code != 200:
        return f"[error] HTTP {resp.status_code}: {resp.text[:400]}"

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return f"[error] Non-JSON response: {resp.text[:400]}"

    out = (data.get("response") or "").strip()
    dt = time.time() - t0
    return out if out else f"[empty] (completed in {dt:.1f}s)"

def main():
    ap = argparse.ArgumentParser(description="OCR with Ollama Vision LLM over REST (no image flags needed).")
    ap.add_argument("image", help="Path to the image file (e.g., img_check.jpg)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model (e.g., llava:7b, llava:13b)")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama host URL (default: http://127.0.0.1:11434)")
    ap.add_argument("--outdir", default="ocr_out", help="Directory to save the transcript (default: ocr_out)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text for OCR behavior")
    ap.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds (first call may need more)")
    ap.add_argument("--max-side", type=int, default=640, help="Resize longest side to this many pixels (speeds up)")
    ap.add_argument("--num-ctx", type=int, default=512, help="Model context window (smaller is faster)")
    ap.add_argument("--num-predict", type=int, default=200, help="Max tokens to generate (limits runtime)")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Check Ollama health
    err = check_ollama(args.host)
    if err:
        print(f"[!] Ollama health check failed: {err}\n"
              f"    Make sure Ollama is running. If you see 'port in use', it's already running.", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {args.model}")
    text = vlm_ocr(
        image_path=image_path,
        model=args.model,
        host=args.host,
        prompt=args.prompt,
        timeout_s=args.timeout,
        max_side=args.max_side,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )

    print("\n--- OCR ---\n")
    print(text)

    # Save transcript next to processed outputs
    out_path = outdir / (image_path.stem + ".txt")
    try:
        out_path.write_text(text, encoding="utf-8")
        print(f"\nSaved transcript: {out_path}")
    except Exception as e:
        print(f"[warn] Could not write transcript: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
