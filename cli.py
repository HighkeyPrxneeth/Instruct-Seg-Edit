import argparse
import base64
import requests
import sys
from pathlib import Path

API_BASE = "http://127.0.0.1:8000/api"

def save_data_uri(uri: str, out_path: Path):
    header, b64 = uri.split(",", 1)
    data = base64.b64decode(b64)
    out_path.write_bytes(data)
    return out_path

def segment(image_path: Path, prompt: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"image": open(image_path, "rb")}
    data = {"prompt": prompt}
    r = requests.post(f"{API_BASE}/segment", files=files, data=data)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("success"):
        print("Error:", payload.get("error"))
        return 1
    masked_uri = payload["data"]["masked_image"]
    mask_uri = payload["data"]["mask"]
    masked_file = save_data_uri(masked_uri, out_dir / "masked.png")
    mask_file = save_data_uri(mask_uri, out_dir / "mask.png")
    print("Saved:", masked_file, mask_file)
    return 0

def inpaint(image_path: Path, mask_path: Path, prompt: str, steps: int, seed: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "image": open(image_path, "rb"),
        "mask": open(mask_path, "rb"),
    }
    data = {"prompt": prompt, "inference_steps": str(steps)}
    if seed is not None:
        data["seed"] = str(seed)
    r = requests.post(f"{API_BASE}/inpaint", files=files, data=data)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("success"):
        print("Error:", payload.get("error"))
        return 1
    result_uri = payload["data"]["result_image"]
    result_file = save_data_uri(result_uri, out_dir / "inpainted.png")
    print("Saved:", result_file)
    return 0

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    seg = sub.add_parser("segment")
    seg.add_argument("image", type=Path)
    seg.add_argument("prompt", type=str)
    seg.add_argument("--out", type=Path, default=Path("out/segment"))

    inp = sub.add_parser("inpaint")
    inp.add_argument("image", type=Path)
    inp.add_argument("mask", type=Path)
    inp.add_argument("prompt", type=str)
    inp.add_argument("--steps", type=int, default=25)
    inp.add_argument("--seed", type=int, default=None)
    inp.add_argument("--out", type=Path, default=Path("out/inpaint"))

    args = p.parse_args()
    if args.cmd == "segment":
        sys.exit(segment(args.image, args.prompt, args.out))
    elif args.cmd == "inpaint":
        sys.exit(inpaint(args.image, args.mask, args.prompt, args.steps, args.seed, args.out))

if __name__ == "__main__":
    main()