import io
import json
import os
import uuid

import pandas as pd
import requests
import websocket
from PIL import Image
from tqdm.notebook import tqdm

# ComfyUI server configuration
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

# Load the ComfyUI workflow from flux.json
with open("flux.json", "r") as f:
    workflow = json.load(f)


def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = requests.post(f"http://{server_address}/prompt", data=data)
    return json.loads(req.text)


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = "&".join([f"{key}={value}" for key, value in data.items()])

    with requests.get(f"http://{server_address}/view?{url_values}") as response:
        return Image.open(io.BytesIO(response.content)).resize((512, 512))


def get_history(prompt_id):
    with requests.get(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.text)


def upload_image_to_comfyui(image_path):
    """Upload an image to ComfyUI and return the filename"""
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(f"http://{server_address}/upload/image", files=files)
        return response.json()["name"]


def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]
    output_images = {}

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break  # Execution is done
        else:
            continue  # previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        if "images" in node_output:
            images_output = []
            for image in node_output["images"]:
                images_output.append(
                    get_image(image["filename"], image["subfolder"], image["type"])
                )
            output_images[node_id] = images_output

    return output_images


# Connect to ComfyUI WebSocket
ws = websocket.WebSocket()
ws.connect(f"ws://{server_address}/ws?clientId={client_id}")

print("Connected to ComfyUI server")


images = os.listdir("data\\images")
prompts = os.listdir("data\\prompts")

data = []
for image in images:
    for prompt in prompts:
        if image.split(".")[0] == prompt.split(".")[0]:
            with open(
                os.path.join("data\\prompts", prompt), "r", encoding="utf-8"
            ) as f:
                edit_instruction = f.read().strip()
            data.append(
                {
                    "image": os.path.join("data\\images", image),
                    "edit_instruction": edit_instruction,
                }
            )

print(len(images), len(prompts), len(data))


df = pd.DataFrame(data)


# Process images one by one using ComfyUI
for i in tqdm(range(len(df))):
    row = df.iloc[i]
    original_image_path = row["image"]
    edit_instruction = row["edit_instruction"]

    # Upload image to ComfyUI
    uploaded_filename = upload_image_to_comfyui(original_image_path)

    # Create a copy of the workflow for this iteration
    current_workflow = workflow.copy()

    # Update the workflow with the current image and prompt
    current_workflow["5"]["inputs"]["image"] = uploaded_filename  # LoadImage node
    current_workflow["6"]["inputs"]["text"] = edit_instruction  # CLIP Text Encode node

    # Generate a random seed for each image
    import random

    current_workflow["20"]["inputs"]["noise_seed"] = random.randint(0, 2**32 - 1)

    # Execute the workflow
    try:
        output_images = get_images(ws, current_workflow)

        # Get the generated image from the PreviewImage node (node 21)
        if "21" in output_images and len(output_images["21"]) > 0:
            generated_image = output_images["21"][0]

            # Get original image dimensions
            original_image = Image.open(original_image_path)
            original_size = original_image.size

            # Resize the generated image to match original dimensions
            resized_image = generated_image.resize(
                original_size, Image.Resampling.LANCZOS
            )

            # Save the result
            original_name = os.path.basename(original_image_path)
            output_path = os.path.join(".\\data\\results", original_name)
            resized_image.save(output_path)

            print(f"Processed: {original_name}")
        else:
            print(f"No output generated for {original_image_path}")

    except Exception as e:
        print(f"Error processing {original_image_path}: {str(e)}")
        continue

print("Batch processing completed   !")
