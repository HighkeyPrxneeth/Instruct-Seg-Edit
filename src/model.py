import numpy as np
from ultralytics import FastSAM
from PIL import Image
import torch
import cv2
from diffusers import FluxFillPipeline
from nunchaku import NunchakuFluxTransformer2dModel

class SegmentationModel:
    def __init__(self, model_path: str = 'models/instruct-seg-edit/best.pt', device: str = 'cuda:0'):
        """
        Initialize the segmentation model.

        Args:
            model_path (str): Path to the pre-trained model weights.
            device (str): Device to run the model on ('cuda:0' for GPU, 'cpu' for CPU).
        """
        self.model = FastSAM(model_path).cuda(device=torch.device(device))
        print("Segmentation model loaded.")
        self.overlay = np.array([255, 105, 180])
        self.alpha = 0.6

    def segment_image(self, image: Image.Image, text_prompt: list[str], return_result: bool = False):
        """
        Perform segmentation on the input image.

        Args:
            image (Image.Image): The input image.
            text_prompt (list[str]): Text prompt for segmentation.
            return_result (bool): If True, returns the image with overlay and mask; otherwise, returns only the mask.

        Returns:
            mask_image (Tuple[PIL.Image]): The segmentation mask image.
            If return_result is True, also returns the image with overlay.
        """
        print("Processing segmentation...")
        print("Text prompt:", ", ".join(text_prompt))
        results = self.model.predict(image, texts=text_prompt)
        mask = results[0].masks.data
        mask_np = mask.squeeze(0).cpu().numpy().astype(bool)
        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
        
        if return_result:
            image = image.resize((mask.shape[2], mask.shape[1]))
            image_np = np.array(image.convert("RGB"))

            image_np[mask_np] = (image_np[mask_np] * (1 - self.alpha) + self.overlay * self.alpha).astype(np.uint8)

            res_image = Image.fromarray(image_np)
            return (res_image, mask_image)

        return (mask_image,)

class InpaintingModel:
    def __init__(self, model_name: str = "black-forest-labs/FLUX.1-Fill-dev", torch_dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        """
        Initialize the inpainting model with Flux Fill using Nunchaku int4 quantization.

        Args:
            model_name (str): Name of the pre-trained Flux Fill model.
            torch_dtype (torch.dtype): Data type for the model tensors.
            device (str): Device to run the model on ('cuda' for GPU, 'cpu' for CPU).
        """
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            "models/nunchaku-flux.1-fill-dev/svdq-int4_r32-flux.1-fill-dev.safetensors",
            local_files_only=True
        )
        print("Nunchaku int4 transformer loaded.")
        
        self.pipe = FluxFillPipeline.from_pretrained(
            model_name,
            transformer=transformer,
            torch_dtype=torch_dtype,
            use_safetensors=True
        )
        print("Flux Fill pipeline initialized with Nunchaku int4 optimization.")
        self.pipe.to(device)
        self.device = device

    def inpaint(self, prompt: str, 
                image: Image.Image, 
                mask: Image.Image,
                num_inference_steps: int = 50, 
                generator: torch.Generator = torch.manual_seed(42), 
                visualize_steps: bool = False,
                guidance_scale: float = 30,
                height: int = 1024,
                width: int = 1024):
        """
        Perform inpainting on the input image using the provided mask.

        Args:
            prompt (str): Text prompt for inpainting.
            image (PIL.Image): The input image to be inpainted.
            mask (PIL.Image): The mask image indicating areas to be inpainted.
            num_inference_steps (int): Number of inference steps for the diffusion process.
            generator (torch.Generator): Random generator for reproducibility.
            visualize_steps (bool): If True, yields intermediate steps during inference.
            guidance_scale (float): Guidance scale for the diffusion process.
            height (int): Output image height.
            width (int): Output image width.
        """
        print("Processing inpainting...")
        print("Text prompt:", prompt)

        if visualize_steps:
            # For Flux Fill, we'll use a callback for intermediate steps
            def run_generator():
                edited_image = self.pipe(
                    prompt=prompt,
                    image=image,
                    mask_image=mask,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                ).images[0]
                yield edited_image

            return run_generator()
    
        else:
            edited_image = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            ).images[0]
            return edited_image