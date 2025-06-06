"""
stable_diffusion.py
-------------------
Minimal wrapper exposing a `Difix` class that performs a single‑call
Stable Diffusion image2image cleanup.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms as _T
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline

# PIL helpers
_TO_PIL = _T.ToPILImage()
_TO_TENSOR = _T.ToTensor()


class Difix(nn.Module):
    """
    Single‑step diffusion cleaner used inside train.py.

    Example
    -------
    >>> cleaner = Difix("runwayml/stable-diffusion-v1-5").to("cuda").eval()
    >>> cleaned  = cleaner(img_tensor)          # img_tensor shape (3,H,W) in [0,1]
    """

    def __init__(self,
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)

        # Load Stable Diffusion Img2Img pipeline
        self.model_id = model_id
        self.pipe = None

        # Default parameters (from user’s prompt)
        self.prompt           = "A sharp, high-resolution photo of a road scene, no blur, no motion artifacts"
        self.negative_prompt  = "blurry, warped, noisy, distorted, painterly"
        self.strength         = 0.30
        self.guidance_scale   = 7.5
        self.steps            = 2

    
    def _get_pipe(self):
        if self.pipe is None:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device.type=="cuda" else torch.float32
            ).to(self.device)
            self.pipe.set_progress_bar_config(disable=False)
            self.pipe.safety_checker = None
            # self.pipe.enable_attention_slicing()
            # self.pipe.enable_vae_slicing()
            # try:
            #     self.pipe.enable_xformers_memory_efficient_attention()
            # except Exception:
            #     pass
            # # comment out if latency matters more than VRAM
            # self.pipe.enable_model_cpu_offload()
        return self.pipe

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self,
                imgs: torch.Tensor,
                prompt: str | None = None,
                negative_prompt: str | None = None,
                strength: float | None = None,
                guidance_scale: float | None = None,
                num_steps: int | None = None) -> torch.Tensor:
        """
        imgs : (3,H,W) or (B,3,H,W) float32/16 in [0,1].
        Returns tensor with identical shape on the same device.
        """
        pipe = self._get_pipe()
        single = imgs.dim() == 3
        if single:
            imgs = imgs.unsqueeze(0)

        prompt          = prompt          or self.prompt
        negative_prompt = negative_prompt or self.negative_prompt
        strength        = strength        if strength is not None else self.strength
        guidance_scale  = guidance_scale  if guidance_scale is not None else self.guidance_scale
        num_steps       = num_steps       if num_steps is not None else self.steps

        outs = []
        for img in imgs:
            # Ensure dimensions divisible by 8 (SD‑v1 constraint)
            C, H, W = img.shape
            pad_h = (8 - (H % 8)) % 8
            pad_w = (8 - (W % 8)) % 8
            if pad_h or pad_w:
                img_proc = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            else:
                img_proc = img
            pil_img = _TO_PIL(img_proc.clamp(0, 1).cpu())
            out_pil = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil_img,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                height=pil_img.height,
                width=pil_img.width,
            ).images[0]
            out_tensor = _TO_TENSOR(out_pil).to(imgs.device)
            if pad_h or pad_w:
                out_tensor = out_tensor[:, :H, :W]
            outs.append(out_tensor)

        result = torch.stack(outs, 0)
        return result[0] if single else result

    # alias
    __call__ = forward


# ---------------------------------------------------------------------
# Quick self‑test:  python stable_diffusion.py --img path/to/input.jpg
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from PIL import Image
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Self‑test for Difix wrapper")
    parser.add_argument("--img", required=True, type=str,
                        help="Path to an RGB image to clean")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5",
                        help="Diffusers model id or local folder")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--out", default=None,
                        help="Optional path to save the cleaned image")
    args = parser.parse_args()

    # Load image as tensor in [0,1]
    pil_in  = Image.open(args.img).convert("RGB")
    to_tensor = _T.ToTensor()
    img_tensor = to_tensor(pil_in).to(args.device)

    # Instantiate Difix and clean
    cleaner = Difix(model_id=args.model, device=args.device).eval()
    cleaned = cleaner(img_tensor)          # keep original spatial resolution
    print(f"output image tensor: {img_tensor.shape}")

    print(f"in {img_tensor.shape} vs out {cleaned.shape}")

    # Save or display result
    to_pil = _T.ToPILImage()
    pil_out = to_pil(cleaned.cpu().clamp(0, 1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pil_out.save(args.out)
        print("Cleaned image written to:", args.out)
    else:
        pil_out.show(title="Difix cleaned")