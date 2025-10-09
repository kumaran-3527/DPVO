import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation



MODEL_PATH = 'seg_models/facebook/mask2former-swin-base-coco-panoptic'
SEG_FP_16 = True

class Mask2Former(nn.Module):
    """Wrapper around Mask2Former panoptic model that also exposes encoder (backbone) features.

    Expected input shape: (B, N, 3, H, W) with pixel values either in [0,1] (float) or [0,255] (uint / float).
    """
    def __init__(self, model_path: str = MODEL_PATH, device: str = 'cuda', eval: bool = True , capture_encoder: bool = True, fp_16 : bool = True ) :
        super().__init__()

        self.device = device
        self.capture_encoder = capture_encoder
        self.fp_16 = fp_16

        # Load processor and model from local path or HuggingFace
        self.seg_processor = AutoImageProcessor.from_pretrained(model_path)
        self.seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(torch.device(self.device))

        if eval:
            self.seg_model.eval()
        
        if fp_16 : 
            self.seg_model.half()

        # Ensure encoder returns hidden states (saves needing extra forward pass)
        enc_cfg = self.seg_model.model.pixel_level_module.encoder.config
        enc_cfg.output_hidden_states = True
        enc_cfg.return_dict = True

        # Container for last captured encoder output & hook handle
        self._encoder_out = None
        self._enc_handle = None
        if self.capture_encoder:
            self._register_encoder_hook()

   
    def _register_encoder_hook(self):
        if self._enc_handle is not None:
            return  
        def _enc_hook(_m, _inp, out):
            self._encoder_out = out

        self._enc_handle = self.seg_model.model.pixel_level_module.encoder.register_forward_hook(_enc_hook)


    def enable_encoder_capture(self):
        """Enable persistent encoder feature capture (idempotent)."""
        self.capture_encoder = True
        self._register_encoder_hook()


    def disable_encoder_capture(self, remove_hook: bool = True):
        """Disable encoder capture. Optionally remove hook to save tiny overhead.
        """
        self.capture_encoder = False
        if remove_hook and self._enc_handle is not None:
            self._enc_handle.remove()
            self._enc_handle = None
        self._encoder_out = None  # free reference


    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """Run panoptic segmentation and return (segmentation_maps, encoder_features).

        segmentation_maps: (B, N, H, W) integer map of segment ids
        encoder_features:  Raw encoder output captured by forward hook (implementation dependent)
        """
        # assert x.dim() == 5, f"Expected (B,N,C,H,W) got {x.shape}"
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

    
        # Processor expects list or batch tensor; it handles resizing internally
        inputs = self.seg_processor(images=list(x), return_tensors="pt", do_rescale=False).to(self.device)
        
        if self.fp_16:
            for k in inputs:
                if inputs[k].dtype == torch.float32:
                    inputs[k] = inputs[k].half()
    
        # Clear previous encoder reference (if any) to avoid holding onto old tensors
        self._encoder_out = None

        seg_outputs = self.seg_model(**inputs)

        panoptic_results = self.seg_processor.post_process_panoptic_segmentation(
            seg_outputs, target_sizes=[(H, W)] * (B * N)
        )

        seg_maps = torch.stack([r["segmentation"] for r in panoptic_results], dim=0)  # (B*N, H, W)
        seg_maps = seg_maps.view(B, N, H, W)

        encoder_feature_maps = None
        if self.capture_encoder and self._encoder_out is not None:
            enc_obj = self._encoder_out
            encoder_feature_maps = enc_obj.feature_maps

        return seg_maps, encoder_feature_maps
    



# class FeatureFusion(nn.Module):

#     def __init__(self, in_channels , out_channels , ):
#         super(FeatureFusion).__init__()



if __name__ == "__main__" : 

    from PIL import Image
    import numpy as np
    import time

    model = Mask2Former(model_path=MODEL_PATH, device='cuda', eval=True, capture_encoder=True, fp_16=SEG_FP_16)
    model.eval()
  
    image = Image.open("datasets/mono/ME002/000002.png").convert("RGB")
    image = torch.from_numpy( np.array(image).astype('float32').transpose(2,0,1) / 255.0 )  # (3,H,W) in [0,1]
    with torch.no_grad():
        seg_maps, enc_feats = model(image.unsqueeze(0).unsqueeze(0).to(model.device))  # (B,N,H,W), list of (B*N,C,H',W')

    print("Segmentation maps shape:", seg_maps.shape)  # (B,N,H,W)

    print("Encoder feature map 1 shape:", enc_feats[0].shape)  # (B*N,C,H',W')
    print("Encoder feature map 2 shape:", enc_feats[1].shape)  # (B*N,C,H',W')
    print("Encoder feature map 3 shape:", enc_feats[2].shape)  # (B*N,C,H',W')
    print("Encoder feature map 4 shape:", enc_feats[3].shape)  # (B*N,C,H',W')
