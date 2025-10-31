import numpy as np
from ultralytics import FastSAM
from PIL import Image
import torch
import cv2
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

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
    def __init__(self, model_name: str = "runwayml/stable-diffusion-inpainting", control_net: str = "lllyasviel/sd-controlnet-canny", torch_dtype: torch.dtype = torch.float16, device: str = "cuda"):
        """
        Initialize the inpainting model with ControlNet.

        Args:
            model_name (str): Name of the pre-trained Stable Diffusion inpainting model.
            control_net (str): Name of the pre-trained ControlNet model.
            torch_dtype (torch.dtype): Data type for the model tensors.
            device (str): Device to run the model on ('cuda' for GPU, 'cpu' for CPU).
        """
        self.control_net = ControlNetModel.from_pretrained(control_net, torch_dtype=torch_dtype, use_safetensors=True, cache_dir="models/controlnet")
        print("ControlNet model loaded.")
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            model_name,
            controlnet=self.control_net,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant="fp16",
            cache_dir="models/stable-diffusion-inpainting"
        )
        print("Inpainting pipeline initialized.")
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.pipe.vae.to(device)

        self.sigma = 0.33

    def _get_canny_edges(self, image_np: np.ndarray):
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        v = np.median(gray_image)
        lower_threshold = int(max(0, (1.0 - self.sigma) * v))
        upper_threshold = int(min(255, (1.0 + self.sigma) * v))

        canny_edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)
        return canny_edges
    
    def _callback_intermediate_diffusion_step(self, step: int, timestep: int, latents: torch.FloatTensor):
        with torch.no_grad():
            latents_for_decode = 1 / self.pipe.vae.config.scaling_factor * latents 
            image_tensor = self.pipe.vae.decode(latents_for_decode).sample

        image_step = self.pipe.image_processor.postprocess(image_tensor, output_type='pil')[0]
        yield image_step

    def inpaint(self, prompt: str, 
                image: Image.Image, 
                mask: Image.Image,
                num_inference_steps: int = 25, 
                generator: torch.Generator = torch.manual_seed(42), 
                visualize_steps: bool = False):
        """
        Perform inpainting on the input image using the provided mask and control image.

        Args:
            prompt (str): Text prompt for inpainting.
            image (PIL.Image): The input image to be inpainted.
            mask (PIL.Image): The mask image indicating areas to be inpainted.
            control (PIL.Image): The control image for ControlNet.
            num_inference_steps (int): Number of inference steps for the diffusion process.
            generator (torch.Generator): Random generator for reproducibility.
        """
        image_np = np.array(image)
        canny_edges = self._get_canny_edges(image_np)
        control_image = Image.fromarray(canny_edges).convert("RGB")
        print("Processing inpainting...")
        print("Text prompt:", prompt)

        if visualize_steps:
            def run_generator():
                edited_image = self.pipe(
                    prompt=prompt,
                    image=image,
                    mask_image=mask,
                    control_image=control_image,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    output_type="latent",
                    callback_on_step_end=self._callback_intermediate_diffusion_step,
                )

                final_image = self.pipe.decode_latents(edited_image.images)
                final_pil_image = self.pipe.image_processor.postprocess(final_image, output_type='pil')[0]
                yield final_pil_image

            return run_generator()
    
        else:
            edited_image = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                control_image=control_image,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
            return edited_image