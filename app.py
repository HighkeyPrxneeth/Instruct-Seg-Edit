from fastapi import FastAPI
from fastapi import (
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import torch
import time
import numpy as np

from src.model import SegmentationModel, InpaintingModel
from src.data_loader import DataLoader

segmentor = SegmentationModel()
inpainter = InpaintingModel()

dataloader = DataLoader()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/segment")
async def segment_image(image: UploadFile = File(...), prompt: str = Form(...)):
    try:
        print(f"Processing segmentation for prompt: {prompt}")
        img = Image.open(image.file).convert("RGB")
        print(f"Image loaded: {img.size}")
        
        seg_result = segmentor.segment_image(img, text_prompt=[prompt], return_result=True)
        result_img, mask_img = seg_result
        print(f"Segmentation complete. Result size: {result_img.size}, Mask size: {mask_img.size}")

        result_path = dataloader.save_image(result_img)
        mask_path = dataloader.save_image(mask_img)

        result_blob = dataloader.convert_to_blob(result_img, format="PNG")
        mask_blob = dataloader.convert_to_blob(mask_img, format="PNG")
        
        result_base64 = base64.b64encode(result_blob).decode('utf-8')
        mask_base64 = base64.b64encode(mask_blob).decode('utf-8')
        
        print(f"Base64 encoding complete. Result length: {len(result_base64)}, Mask length: {len(mask_base64)}")
        
        response = {
            "success": True,
            "data": {
                "masked_image": f"data:image/png;base64,{result_base64}",
                "mask": f"data:image/png;base64,{mask_base64}",
                "width": result_img.width,
                "height": result_img.height
            }
        }
        
        print("Response prepared successfully")
        return response
    except Exception as e:
        print(f"Error in segment_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/inpaint")
async def inpaint_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    inference_steps: int = Form(...),
    seed: int = Form(None)
):
    img = Image.open(image.file).convert("RGB")
    mask_img = Image.open(mask.file).convert("L")

    # Ensure mask and image have the same dimensions (use NEAREST to avoid anti-aliasing)
    if img.size != mask_img.size:
        mask_img = mask_img.resize(img.size, Image.NEAREST)

    # Binarize mask to ensure a clean inpaint region: white(255)=edit region, black(0)=keep
    mask_img = mask_img.point(lambda p: 255 if p > 127 else 0)

    # Invert mask: pipeline expects white (255) to be the area to KEEP and black (0) to be inpainted.
    # Our mask has white on the object, so we invert it.
    mask_img = Image.fromarray(255 - np.array(mask_img))

    if seed is None:
        seed = int(time.time())
    generator = torch.manual_seed(seed)
    inpainted_img = inpainter.inpaint(
        prompt=prompt,
        image=img,
        mask=mask_img,
        num_inference_steps=inference_steps,
        generator=generator
    )

    inpainted_path = dataloader.save_image(inpainted_img)
    inpainted_blob = dataloader.convert_to_blob(inpainted_img, format="PNG")
    
    inpainted_base64 = base64.b64encode(inpainted_blob).decode('utf-8')
    
    return {
        "success": True,
        "data": {
            "result_image": f"data:image/png;base64,{inpainted_base64}"
        }
    }