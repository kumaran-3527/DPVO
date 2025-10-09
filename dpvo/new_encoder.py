import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from typing import Tuple
import warnings
import os
warnings.filterwarnings("ignore")



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
        inputs = self.seg_processor(images=list(x), return_tensors="pt", do_rescale=False, use_fast = True).to(self.device)
        
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


class TwoLevelEncoder(nn.Module):
    """Minimal encoder-decoder with 2 conv (down) + 2 deconv (up) layers and skip connection.

    Design:
        - conv1 : stride=2  (H,W) -> (H/2, W/2), C
        - conv2 : stride=2  (H/2, W/2) -> (H/4, W/4), 2C
        - deconv1 : stride=2 upsample to (H/2, W/2), reduce to C and fuse (concat) with skip from conv1
        - deconv2 : stride=1 (no spatial change) produce final out_dim at (H/2, W/2)

    Only one final resolution: 1/2 of input as requested.
    Input shape: (B, N, 3, H, W)
    Output shape: (B, N, out_dim, H/2, W/2)
    """
    def __init__(self, in_channels = 3 , inter_channels = 64, out_channels =64, norm='instance', activation='relu'):
        super().__init__()
        self.out_dim = out_channels

        Act = nn.ReLU if activation == 'relu' else nn.GELU
        self.act = Act(inplace=True) if activation != 'gelu' else Act()

        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels * 2, kernel_size=3, stride=2, padding=1)

        self.deconv1 = nn.ConvTranspose2d(inter_channels * 2, inter_channels, kernel_size=2, stride=2)
        self.fuse1 = nn.Conv2d(inter_channels * 2, inter_channels, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Norm layers helper
        def make_norm(ch):
            if norm == 'batch':
                return nn.BatchNorm2d(ch)
            if norm == 'instance':
                return nn.InstanceNorm2d(ch)
            return nn.Identity()

        self.n1 = make_norm(inter_channels)
        self.n2 = make_norm(inter_channels * 2)
        self.n3 = make_norm(inter_channels)
        self.n4 = make_norm(out_channels)
        self.nf = make_norm(inter_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept shape (B,N,C,H,W)
        assert x.dim() == 5, f"Expected (B,N,C,H,W), got {x.shape}"
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        # Encoder
        e1 = self.act(self.n1(self.conv1(x)))        # (B*N, inter_channels, H/2, W/2)
        e2 = self.act(self.n2(self.conv2(e1)))       # (B*N, 2*inter_channels, H/4, W/4)

        d1 = self.act(self.n3(self.deconv1(e2)))     # (B*N, inter_channels, H/2, W/2)
        # Skip fuse
        f1 = torch.cat([d1, e1], dim=1)
        f1 = self.act(self.nf(self.fuse1(f1)))       # (B*N, inter_channels, H/2, W/2)
        out = self.n4(self.deconv2(f1))              # (B*N, out_channels, H/2, W/2)

        out = out.view(B, N, self.out_dim, H // 2, W // 2)
        return out


class Mask2FormerTwoLevel(nn.Module):
    """Run Mask2Former (segmentation + backbone features) and TwoLevelEncoder on the same input.

    Features:
        - Shared input shape: (B, N, 3, H, W)
        - Output dict keys:
            segmentation: (B, N, H, W) panoptic ids
            mask2former_encoder: list/tuple of backbone feature maps (depends on HF model)
            twolevel_features: (B, N, out_dim, H/2, W/2)
        - Optional parallel execution using two CUDA streams to overlap heavy Mask2Former forward
          with the lightweight TwoLevelEncoder (only if parallel=True and CUDA available).
        - Optionally run Mask2Former under torch.no_grad (default) while allowing gradients
          through the small TwoLevelEncoder if you are training something on its features.

    Args:
        device: torch.device or str, target device for both models
        parallel: bool, try overlapping with CUDA streams.
        no_grad_mask2former: bool, wrap Mask2Former forward in no_grad (saves memory).
    """
    def __init__(self,
                 device: str | torch.device = 'cuda',
                 parallel: bool = False,
                 no_grad_mask2former: bool = True,
            ):
        super().__init__()
        
        self.device = torch.device(device) if isinstance(device, str) else device
        self.parallel = parallel and torch.cuda.is_available()
        self.no_grad_mask2former = no_grad_mask2former
    
        m2f_kwargs = {
            'model_path': MODEL_PATH,
            'device': self.device,
            'eval': True,
            'capture_encoder': True,
            'fp_16': SEG_FP_16,
        }
        self.mask2former = Mask2Former(**m2f_kwargs)
        
        tl_kwargs = {
            'in_channels': 3,
            'inter_channels': 64,
            'out_channels': 128,
            'norm': 'instance',
            'activation': 'relu',
        }
        self.twolevel = TwoLevelEncoder(**tl_kwargs).to(self.device)
        
        if self.parallel:
            self._stream1 = torch.cuda.Stream(device=self.device)
            self._stream2 = torch.cuda.Stream(device=self.device)
        else:
            self._stream1 = self._stream2 = None

    def to(self, device):
        """Override to ensure both models move together."""
        super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.mask2former = self.mask2former.to(device)
        self.twolevel = self.twolevel.to(device)

        if self.parallel and torch.cuda.is_available():
            self._stream1 = torch.cuda.Stream(device=self.device)
            self._stream2 = torch.cuda.Stream(device=self.device)
        return self

    def train(self, mode=True):
        """Override to handle mixed training modes properly."""

        super().train(mode)
        self.twolevel.train(mode)
        self.mask2former.eval()
        return self

    def forward(self, x: torch.Tensor) :

        if x.device != self.device:
            x = x.to(self.device)
            
        if not self.parallel:
  
            if self.no_grad_mask2former:
                with torch.no_grad():
                    seg_maps, enc_feats = self.mask2former(x)
            else:
                seg_maps, enc_feats = self.mask2former(x)
            
            twolevel_feats = self.twolevel(x)
            
        else:
            seg_maps = enc_feats = twolevel_feats = None
            if self.no_grad_mask2former:
                with torch.cuda.stream(self._stream1), torch.no_grad():
                    seg_maps, enc_feats = self.mask2former(x)
            else:
                with torch.cuda.stream(self._stream1):
                    seg_maps, enc_feats = self.mask2former(x)
     
            with torch.cuda.stream(self._stream2):
                twolevel_feats = self.twolevel(x)
            
            self._stream1.synchronize()
            self._stream2.synchronize()

        return seg_maps, enc_feats, twolevel_feats
    

    # def get_memory_usage(self):
    #     """Helper to monitor GPU memory usage."""
    #     if torch.cuda.is_available():
    #         return {
    #             'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
    #             'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,    # GB
    #             'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3,  # GB
    #         }
    #     return None


def test_combined_model(device='cuda', parallel=True, test_training=False):

    print(f"Testing Mask2FormerTwoLevel (parallel={parallel}, device={device})")
    
    try:
       
        model = Mask2FormerTwoLevel(device=device, parallel=parallel)
        print(f"Model initialized on {model.device}")
        
        x = torch.rand(1, 1, 3, 480, 640, device=device, dtype=torch.float32)

        print("Warming up...")
        for i in range(3):
            with torch.no_grad():
                seg_maps, enc_feats, twolevel_feats = model(x)
        
        torch.cuda.synchronize()
        import time
        start = time.time()
        with torch.no_grad():
            seg_maps, enc_feats, twolevel_feats = model(x)
        torch.cuda.synchronize()
        inference_time = 1000 * (time.time() - start)
        print(seg_maps.shape , enc_feats.shape, twolevel_feats.shape)
        print(f"Inference time: {inference_time:.2f} ms")



        if test_training:
            print("\nTesting training mode...")
            model.train()
            model.no_grad_mask2former = True 

            seg_maps, enc_feats, twolevel_feats = model(x)
            loss = twolevel_feats.mean()
            loss.backward()
            print(f"Training forward+backward successful. Loss: {loss.item():.6f}")
            
      
            has_grads = sum(1 for p in model.twolevel.parameters() if p.grad is not None)
            total_params = sum(1 for p in model.twolevel.parameters())
            print(f"TwoLevel gradients: {has_grads}/{total_params} parameters")
            
            m2f_grads = sum(1 for p in model.mask2former.parameters() if p.grad is not None)
            print(f"Mask2Former gradients: {m2f_grads} (should be 0)")
        
        return model , seg_maps, enc_feats, twolevel_feats
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM Error: {e}")
        print("Try reducing batch size or input resolution")
        torch.cuda.empty_cache()
        return None, None
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 50)
    # model_seq, out_seq = test_combined_model(device=device, parallel=False, test_training=True)
    
    print("\n" + "=" * 50)
    model_par , seg_maps, enc_feats, twolevel_feats = test_combined_model(device=device, parallel=True, test_training=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nTesting complete!")



         

