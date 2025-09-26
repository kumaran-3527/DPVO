import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import cv2
import numpy as np


class DinoV2Encoder(nn.Module):
    def __init__(self, folder_path: str = "dinov2-models/dinov2-base",
                 model_name: str = "facebook/dinov2-base",
                 accepted_image_size: int = 224,
                 trainable: bool = False):
        """
        Wrap a DINOv2 ViT backbone and return patch features as a 2D feature map.

        - Input: images (B, N, 3, H, W) in [0,1]
        - Output: features (B, N, C, H_p, W_p)
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, cache_dir=folder_path)

        for p in self.model.parameters():
            p.requires_grad = bool(trainable)
        self.model.train(mode=trainable)

        self.accepted_image_size = accepted_image_size

        # normalization constants (ImageNet stats)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, N, 3, H, W) in [0,1]
        returns: (B, N, C, H_p, W_p) feature map
        """
        B, N, C_in, H_in, W_in = images.shape
        flat = images.view(B * N, C_in, H_in, W_in)

        # resize to model input size
        flat = F.interpolate(
            flat, size=(self.accepted_image_size, self.accepted_image_size),
            mode="bilinear", align_corners=False
        )

        # normalize
        flat = (flat - self.mean) / self.std

        no_grad = not any(p.requires_grad for p in self.model.parameters())
        with torch.no_grad() if no_grad else torch.enable_grad():
            outputs = self.model(pixel_values=flat)

        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B*N, n_patches, hidden_dim)

        # reshape to feature map
        b_flat, n_patches, hidden_dim = patch_tokens.shape
        side = int(n_patches ** 0.5)
        if side * side != n_patches:
            raise ValueError(f"Patch count {n_patches} is not square")
        feat = patch_tokens.transpose(1, 2).contiguous().view(b_flat, hidden_dim, side, side)

        return feat.view(B, N, hidden_dim, side, side)






if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encoder
    encoder = DinoV2Encoder().to(device)

    image_path = "datasets/TartanAir/AbandonedFactory/P001/image_left/000000_left.png"
    img = np.array(cv2.imread(image_path))
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0  # (1,1,3,H,W)

    B, N, C, H, W = x.shape
    # Warmup/Timing
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    with torch.no_grad():
        feats = encoder(x)  # (B, N, C, Hf, Wf)

    if device.type == "cuda":
        end.record()
        torch.cuda.synchronize()
        print(f"Encoder output: {tuple(feats.shape)} | time: {start.elapsed_time(end):.2f} ms")
    else:
        print(f"Encoder output: {tuple(feats.shape)}")
