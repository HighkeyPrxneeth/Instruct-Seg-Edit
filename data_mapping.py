from glob import glob
import simdjson as json

original_images = sorted(glob('data\\val\\*.jpg'))
edited_images = sorted(glob('results\\*.jpg'))
masks = sorted(glob('masks\\*.jpg'))
prompts = sorted(glob('data\\val\\*.txt'))

with open("data\\val.json", "r", encoding="utf-8") as f:
    data = json.load(f)

mappings = []
for item in data:
    id = item["image_id"]
    if id + ".jpg" in [img.split('\\')[-1] for img in original_images]:
        idx = [img.split('\\')[-1] for img in original_images].index(id + ".jpg")
        mappings.append({
            'original_image': original_images[idx],
            'edited_image': edited_images[idx],
            'mask': masks[idx],
            'prompt': prompts[idx],
            'captions': item["caption_str"],
        })


with open('data_mappings.json', 'w') as f:
    json.dump(mappings, f, indent=4)