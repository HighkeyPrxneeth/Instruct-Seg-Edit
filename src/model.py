import numpy as np
from ultralytics import FastSAM
from PIL import Image
import torch
import os
import cv2
from diffusers import (
    AutoPipelineForInpainting,
    ControlNetModel, 
    FluxFillPipeline, 
    StableDiffusionControlNetInpaintPipeline, 
    UniPCMultistepScheduler,
    StableDiffusionInpaintPipeline,
)
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel
from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from openai import OpenAI

from src.data_loader import DataLoader

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
    """Selectable inpainting backend supporting Flux Fill, SD3, or SD1.5 ControlNet."""

    _DEFAULT_BACKEND = "flux"
    _SUPPORTED_BACKENDS = {"flux", "sd3", "sd15", "sd2", "sdxl", "api"}
    _DEFAULT_MODELS = {
        "flux": "black-forest-labs/FLUX.1-Fill-dev",
        "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
        "sd15": "runwayml/stable-diffusion-inpainting",
        "sd2": "stabilityai/stable-diffusion-2-inpainting",
        "sdxl": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "api": "recraft"
    }
    _DEFAULT_CONTROLNET = "lllyasviel/sd-controlnet-canny"
    _SD3_NEGATIVE_PROMPT = (
        "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, "
        "extra limb, missing limb, floating limbs, mutated hands and fingers, "
        "disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW"
    )

    def __init__(
        self,
        model_name: str | None = None,
        torch_dtype: torch.dtype | None = None,
        device: str = "cuda",
        backend: str | None = None,
        control_net: str | None = None,
    ):
        """Initialize the inpainting model with the requested diffusion backend."""

        resolved_backend = (backend or os.environ.get("INPAINT_BACKEND") or self._DEFAULT_BACKEND).lower()
        if resolved_backend not in self._SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported inpainting backend '{resolved_backend}'. Pick one of: {sorted(self._SUPPORTED_BACKENDS)}."
            )

        self.backend = resolved_backend
        self.device = device
        self._sd15_sigma = 0.33
        self.dataloader = DataLoader()

        effective_model = model_name or self._DEFAULT_MODELS[self.backend]

        if self.backend == "flux":
            self._init_flux(model_name=effective_model, torch_dtype=torch_dtype)
        elif self.backend == "sd3":
            self._init_sd3()
        elif self.backend == "sd2":
            self._init_sd2()
        elif self.backend == "sdxl":
            self._init_sdxl(model_name=effective_model, torch_dtype=torch_dtype)
        elif self.backend == "api":
            self._init_api()
        else:
            self._init_sd15(model_name=effective_model, torch_dtype=torch_dtype, control_net=control_net)

        print(f"Inpainting backend '{self.backend}' initialized on device '{self.device}'.")

    def _init_flux(self, model_name: str, torch_dtype: torch.dtype | None) -> None:
        """Load the Flux Fill pipeline with the Nunchaku quantized transformer."""

        dtype = torch_dtype or torch.float16
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            "models/nunchaku-flux.1-fill-dev/svdq-int4_r32-flux.1-fill-dev.safetensors",
            local_files_only=True,
        )
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        print("Transformer model loaded for Nunchaku int4 optimization.")

        self.pipe = FluxFillPipeline.from_pretrained(
            model_name,
            transformer=transformer,
            use_safetensors=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir="models/flux-fill",
        )

        # # Reduce VRAM pressure by slicing attention and keeping text encoder on CPU.
        self.pipe.enable_attention_slicing()
        # If accelerate is installed / supported, enable CPU offload to greatly reduce VRAM peak.
        if hasattr(self.pipe, "enable_model_cpu_offload"):
            try:
                self.pipe.enable_model_cpu_offload()
                print("Enabled model CPU offload to reduce VRAM usage.")
            except Exception as e:
                print("Model CPU offload available but failed to enable:", e)
        # if self.device.startswith("cuda"):
        #     self.pipe.unet.to(self.device)
        #     if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
        #         self.pipe.vae.to(self.device)
        #     # Set execution device to keep scheduler latents on the GPU.
        #     self.pipe._execution_device = torch.device(self.device)
        # else:
        print("Flux Fill pipeline initialized with Nunchaku int4 optimization.")
        self.pipe.to(self.device)

    def _init_sd3(self) -> None:
        """Load the Stable Diffusion 3 ControlNet inpainting pipeline."""

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        controlnet = SD3ControlNetModel.from_pretrained(
            "alimama-creative/SD3-Controlnet-Inpainting",
            use_safetensors=True,
            extra_conditioning_channels=1,
            torch_dtype=dtype,
            cache_dir="models/sd3-controlnet-inpainting",
        )
        self.pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            controlnet=controlnet,
            torch_dtype=dtype,
            cache_dir="models/sd3-controlnet-inpainting",
        )

        if self.device.startswith("cuda"):
            self.pipe.text_encoder.to(dtype)
            self.pipe.controlnet.to(dtype)
        self.pipe.to(self.device)

    def _init_sd15(self, model_name: str, torch_dtype: torch.dtype | None, control_net: str | None) -> None:
        """Load the Stable Diffusion 1.5 ControlNet inpainting pipeline."""

        dtype = torch_dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        control_model_id = control_net or self._DEFAULT_CONTROLNET
        self.control_net = ControlNetModel.from_pretrained(
            control_model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir="models/controlnet",
        )
        pipe_kwargs = {
            "controlnet": self.control_net,
            "torch_dtype": dtype,
            "use_safetensors": True,
            "cache_dir": "models/stable-diffusion-inpainting",
        }
        if dtype == torch.float16:
            pipe_kwargs["variant"] = "fp16"
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            model_name,
            **pipe_kwargs,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            self.pipe.vae.to(self.device)
    
    def _init_sd2(self) -> None:
        """Load the Stable Diffusion 2 inpainting pipeline."""
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir="models/stable-diffusion-2-inpainting",
        )

        self.pipe.to(self.device)

    def _init_sdxl(self, model_name: str, torch_dtype: torch.dtype | None) -> None:
        """Load the Stable Diffusion XL inpainting pipeline."""

        dtype = torch_dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        pipe_kwargs: dict[str, object] = {
            "torch_dtype": dtype,
            "cache_dir": "models/stable-diffusion-xl-inpainting",
        }
        if dtype == torch.float16:
            pipe_kwargs["variant"] = "fp16"

        self.pipe = AutoPipelineForInpainting.from_pretrained(
            model_name,
            **pipe_kwargs,
        )
        self.pipe.to(self.device)

    def _init_api(self) -> None:
        client = OpenAI(
            base_url='https://external.api.recraft.ai/v1',
            api_key=os.environ["RECRAFT_API_KEY"],
        )
        self.client = client

    def _sd15_get_canny_edges(self, image_np: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        v = np.median(gray_image)
        lower_threshold = int(max(0, (1.0 - self._sd15_sigma) * v))
        upper_threshold = int(min(255, (1.0 + self._sd15_sigma) * v))
        return cv2.Canny(gray_image, lower_threshold, upper_threshold)

    def _sd15_control_image(self, image: Image.Image) -> Image.Image:
        image_np = np.array(image.convert("RGB"))
        canny_edges = self._sd15_get_canny_edges(image_np)
        return Image.fromarray(canny_edges).convert("RGB")

    def inpaint(self, prompt: str, 
                image: Image.Image, 
                mask: Image.Image,
                num_inference_steps: int = 50, 
                generator: torch.Generator = torch.manual_seed(42), 
                visualize_steps: bool = False,
                guidance_scale: float = 30,
                height: int = 1024,
                width: int = 1024,
                negative_prompt: str | None = None,
                controlnet_conditioning_scale: float | None = None,
                strength: float | None = None):
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

        if self.backend == "flux":
            if visualize_steps:
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

        if self.backend == "sdxl":
            def run_pipe() -> Image.Image:
                pipe_kwargs = {
                    "prompt": prompt,
                    "image": image,
                    "mask_image": mask,
                    "num_inference_steps": num_inference_steps,
                    "generator": generator,
                    "guidance_scale": guidance_scale,
                    "height": height,
                    "width": width,
                }
                if strength is not None:
                    pipe_kwargs["strength"] = strength
                return self.pipe(**pipe_kwargs).images[0]

            if visualize_steps:
                def run_generator():
                    yield run_pipe()

                return run_generator()

            return run_pipe()

        if self.backend == "sd15":
            control_image = self._sd15_control_image(image)
            if visualize_steps:
                def run_generator():
                    edited_image = self.pipe(
                        prompt=prompt,
                        image=image,
                        mask_image=mask,
                        control_image=control_image,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                    ).images[0]
                    yield edited_image

                return run_generator()

            edited_image = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                control_image=control_image,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
            return edited_image

        if self.backend == "sd2":
            if visualize_steps:
                def run_generator():
                    edited_image = self.pipe(
                        prompt=prompt,
                        image=image,
                        mask_image=mask,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                    ).images[0]
                    yield edited_image

                return run_generator()
            
            edited_image = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
            return edited_image
        
        if self.backend == "api":
            response = self.client.post(
                path="/images/inpaint",
                cast_to=object,
                options={'headers': {'Content-Type': 'multipart/form-data'}},
                files={
                    'image': open(self.dataloader.save_image(image), 'rb'),
                    'mask': open(self.dataloader.save_image(mask), 'rb'),
                },
                body={
                    'prompt': prompt,
                },
            )
            url = response['data'][0]['url']
            edited_image = self.dataloader.load_image_from_url(url)
            return edited_image

        scale = controlnet_conditioning_scale if controlnet_conditioning_scale is not None else 0.95
        neg_prompt = negative_prompt if negative_prompt is not None else self._SD3_NEGATIVE_PROMPT
        edited_image = self.pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=height,
            width=width,
            control_image=image,
            control_mask=mask.convert("RGB"),
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=scale,
            guidance_scale=guidance_scale,
        ).images[0]
        return edited_image