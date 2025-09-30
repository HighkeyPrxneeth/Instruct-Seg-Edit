import simdjson as json
import os
from tqdm import tqdm

data = json.load(open("data/train.json", "r", encoding="utf-8"))
for image in tqdm(os.listdir("data/train/")):
    image_id = image.split(".")[0]
    for entry in data:
        if entry["image_id"] == image_id:
            with open(f"data/train/{image_id}.txt", "w", encoding="utf-8") as f:
                f.write(entry["prompt"])