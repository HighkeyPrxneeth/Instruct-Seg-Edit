import simdjson as json
from time import sleep
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading


rate_limit_event = threading.Event()
rate_limit_lock = threading.Lock()


def download_image(entry):
    img_path = f"data/train/{entry['image_id']}.jpg"
    while rate_limit_event.is_set():
        sleep(1)
    try:
        response = requests.get(entry["flickr_300k_url"])
        if response.status_code == 200:
            img_data = response.content
            with open(img_path, "wb") as img_f:
                img_f.write(img_data)
        elif response.status_code == 429:
            with rate_limit_lock:
                if not rate_limit_event.is_set():
                    print(f"Rate limited on {entry['image_id']}, pausing all downloads...")
                    rate_limit_event.set()
                    sleep(5)
                    rate_limit_event.clear()
            download_image(entry)
        else:
            print(f"Failed to download {entry['image_id']}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Failed to download {entry['image_id']}: {e}")

with open("data/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(download_image, data), total=len(data)))
