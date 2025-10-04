import torch
import torch.nn.functional as F

def normalize_feat(Fx, eps=1e-6):
    return Fx / (Fx.norm(dim=1, keepdim=True) + eps)

def local_corr(F0, F1, R):  # cosine similarity in a (2R+1)^2 window
    # F0, F1: (B,C,h,w)
    B, C, h, w = F0.shape
    F0n = normalize_feat(F0)
    F1n = normalize_feat(F1)

    # Unfold F1 into local windows around each position of F0
    pad = R
    F1p = F.pad(F1n, (pad, pad, pad, pad))
    # Extract all (2R+1)^2 neighbors as a (B, C*(K), h*w) then reshape
    K = (2*R+1)**2
    patches = F.unfold(F1p, kernel_size=2*R+1, padding=0, stride=1)  # (B, C*(K), h*w)
    patches = patches.view(B, C, K, h, w)                            # (B, C, K, h, w)

    # Cosine sim: dot(F0, each neighbor)
    F0e = F0n.unsqueeze(2)                                           # (B, C, 1, h, w)
    sims = (F0e * patches).sum(dim=1)                                # (B, K, h, w)

    # Get best and second-best
    s1, idx = sims.max(dim=1)                                        # (B, h, w)
    top2 = sims.topk(2, dim=1).values                                # (B, 2, h, w)
    s2 = top2[:,1]                                                   # (B, h, w)

    # Decode displacement from idx
    # Map linear idx -> (dy, dx) in [-R, R]
    ky = idx // (2*R+1)
    kx = idx %  (2*R+1)
    dy = (ky - R).float()
    dx = (kx - R).float()
    flow = torch.stack([dx, dy], dim=1)                              # (B, 2, h, w)

    return s1, s2, flow  # sims in [-1,1], flow in feature pixels

def mutual_nn_mask(F0, F1, R):
    s1_f, _, flow_f = local_corr(F0, F1, R)  # 0->1
    s1_b, _, flow_b = local_corr(F1, F0, R)  # 1->0

    B, _, h, w = F0.shape
    # coords in 0->1
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid = torch.stack([xx, yy], dim=0).float().to(F0.device)  # (2,h,w)

    xy1 = grid + flow_f[0]  # assume B=1; extend to batch if needed
    # Sample backward flow at xy1
    # Build normalized grid for grid_sample
    def to_norm(xy, W, H):
        x = (xy[0] / (W-1))*2 - 1
        y = (xy[1] / (H-1))*2 - 1
        return torch.stack([x, y], dim=-1)  # (H,W,2)

    norm_grid = to_norm(xy1, w, h).unsqueeze(0)  # (1,h,w,2)
    flow_b_at_xy1 = F.grid_sample(flow_b, norm_grid, mode='bilinear', align_corners=True)
    # MNN if flow_f + flow_b(x+flow_f) ≈ 0
    fb_res = (flow_f + flow_b_at_xy1).norm(dim=1)  # (1,h,w)
    mnn = (fb_res < 0.5).float()  # threshold in feature pixels

    return mnn.clamp(0,1), flow_f, flow_b

def warp_image(I, flow_img):  # warp I by flow (from ref to I), both at image res
    B, C, H, W = I.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    base = torch.stack([xx, yy], dim=0).float().to(I.device).unsqueeze(0)  # (1,2,H,W)
    tgt = base + flow_img  # (B,2,H,W)
    x = (tgt[:,0] / (W-1))*2 - 1
    y = (tgt[:,1] / (H-1))*2 - 1
    grid = torch.stack([x, y], dim=-1)
    return F.grid_sample(I, grid, mode='bilinear', padding_mode='border', align_corners=True)

def compute_confidence(I0, I1, F0, F1, R=4, feat_stride=8,
                       sigma_photo=0.15, tau_fb=1.5, eps=1e-6):
    # 1) Match & scores
    s1, s2, flow_f = local_corr(F0, F1, R)           # (1,h,w)
    c_sim = ((s1 + 1.0) * 0.5).clamp(0,1)
    c_ratio = (s1 - s2).clamp(0,1)

    # 2) MNN + forward-backward residual
    mnn, flow_fwd, flow_bwd = mutual_nn_mask(F0, F1, R)
    B, _, h, w = F0.shape

    # 3) Photometric check at image res
    # Upsample flow to image res
    flow_img = F.interpolate(flow_fwd, size=I0.shape[-2:], mode='bilinear', align_corners=True) * feat_stride
    I1w = warp_image(I1, flow_img)
    e_photo = (I0 - I1w).abs().mean(dim=1, keepdim=True)           # (1,1,H,W)
    c_photo = torch.exp(-(e_photo / sigma_photo)).clamp(0,1)       # (1,1,H,W)
    # Downsample c_photo to (h,w) to combine at feature res
    c_photo_ds = F.interpolate(c_photo, size=(h,w), mode='area').squeeze(1)

    # 4) FB consistency at feature res (already computed inside MNN if you want softer version)
    # Recompute soft residual for confidence:
    # sample flow_bwd at x+flow_fwd
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid = torch.stack([xx, yy], dim=0).float().to(F0.device).unsqueeze(0)  # (1,2,h,w)
    xy1 = grid + flow_fwd
    x = (xy1[:,0] / (w-1))*2 - 1
    y = (xy1[:,1] / (h-1))*2 - 1
    g = torch.stack([x, y], dim=-1)
    flow_b_at_xy1 = F.grid_sample(flow_bwd, g, mode='bilinear', align_corners=True)
    fb_err = (flow_fwd + flow_b_at_xy1).norm(dim=1)                # (1,h,w)
    c_fb = torch.exp(-(fb_err / tau_fb)).clamp(0,1)

    # 5) Combine (multiplicative)
    c = (c_sim.clamp_min(eps) *
         c_ratio.clamp_min(eps) *
         torch.where(mnn>0, mnn, torch.full_like(mnn, 0.2)) *  # soft-penalize non-MNN
         c_photo_ds.clamp_min(eps) *
         c_fb.clamp_min(eps))

    # Optional: light edge-aware smoothing (fast guidance via joint bilateral approximation)
    # Here: simple 3x3 median as placeholder (replace with guided filtering if needed)
    c = F.avg_pool2d(c.unsqueeze(1), 3, 1, 1).squeeze(1).clamp(0,1)

    return c, flow_fwd  # confidence at (h,w), flow at feature res (pixels)



if __name__ == "__main__":
    # Test with dummy data

    from PIL import Image
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    from pathlib import Path


    image1 = Image.open("datasets/mono/MH001/000001.png")
    image2 = Image.open("datasets/mono/MH001/000003.png")

    model = 'facebook/mask2former-swin-base-coco-panoptic' ## Pretty decent
    seg_processor = AutoImageProcessor.from_pretrained(model)
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(model)
    seg_model = seg_model.eval().to(torch.device('cuda'))

    seg_inputs = seg_processor(images=[image1,image2], return_tensors="pt")
    inputs = {k: (v.to(seg_model.device) if isinstance(v, torch.Tensor) else v) for k, v in seg_inputs.items()}

    # with torch.no_grad():
    # # Run the whole model while capturing the backbone output via a forward hook
    # _captured = {}
    # def _enc_hook(_m, _in, out):
    #     _captured["bb_out"] = out
    # _h = seg_model.model.pixel_level_module.encoder.register_forward_hook(_enc_hook)
    # seg_outputs = seg_model(**inputs)
    # _h.remove()

    with torch.no_grad():

        captured = {}
        def enc_hook(m, _in, out):
            captured["bb_out"] = out
        h = seg_model.model.pixel_level_module.encoder.register_forward_hook(enc_hook)
        seg_outputs = seg_model(**inputs)
        h.remove()
    
    bb_out = captured["bb_out"]
    c4_map    = bb_out.feature_maps[0]  # (2, C, H/8, W/8) C=128 for swin-base

    I0 = seg_inputs['pixel_values'][0:1].to(torch.device('cuda'))  # (1,3,H,W)
    I1 = seg_inputs['pixel_values'][1:2].to(torch.device('cuda'))  # (1,3,H,W)
    F0 = c4_map[0:1]  # (1,C,h,w)
    F1 = c4_map[1:2]  # (1,C,h,w)   

    c, flow = compute_confidence(I0, I1, F0, F1, R=4, feat_stride=4,
                                sigma_photo=0.15, tau_fb=1.5, eps=1e-6)

    # ================= Visualization =================
    import numpy as np, matplotlib.pyplot as plt

    feat_stride = 4  # must match the value passed above

    # Confidence at feature resolution
    conf_feat = c[0].detach().cpu().numpy()  # (h,w)
    # Upsample confidence map to image resolution
    conf_up = F.interpolate(c.unsqueeze(1), size=I0.shape[-2:], mode='bilinear', align_corners=False)[0,0].cpu().numpy()
    ## If pixels in conf_up are abouve a threshold, consider them as 1 and others 0
    conf_up = (conf_up > 0.05).astype(np.float32)

    # Flow at feature resolution (dx, dy) in feature pixels
    flow_feat = flow[0].detach().cpu().numpy()  # (2,h,w)
    # Upsample flow to image resolution (multiply by stride after interpolation)
    flow_up = F.interpolate(flow, size=I0.shape[-2:], mode='bilinear', align_corners=True)[0].detach().cpu().numpy() * feat_stride  # (2,H,W)

    # Convert normalized processor pixel_values (already normalized) back to displayable RGB approx.
    # pixel_values are usually normalized; just min-max for quick visualization (not exact original colors)
    def tensor_to_disp(img_t):
        x = img_t.detach().cpu().float()
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return np.transpose(x.numpy(), (1,2,0))

    img0_disp = tensor_to_disp(I0[0])
    img1_disp = tensor_to_disp(I1[0])

    # Simple flow -> color (HSV) utility
    def flow_to_color(fx, fy):
        mag = np.sqrt(fx**2 + fy**2)
        ang = np.arctan2(fy, fx)  # [-pi, pi]
        ang_norm = (ang + np.pi) / (2*np.pi)  # [0,1]
        mag_norm = mag / (mag.max() + 1e-8)
        hsv = np.stack([ang_norm, np.ones_like(ang_norm), mag_norm], axis=-1)  # (H,W,3)
        import matplotlib.colors as mcolors
        rgb = mcolors.hsv_to_rgb(hsv)
        return (rgb*255).astype(np.uint8)

    flow_color = flow_to_color(flow_up[0], flow_up[1])  # (H,W,3)

    # Quiver sampling (to avoid clutter)
    Hf, Wf = conf_feat.shape
    step = max(1, min(Hf, Wf)//25)  # heuristic for density
    yy, xx = np.mgrid[0:Hf:step, 0:Wf:step]
    u = flow_feat[0, yy, xx]
    v = flow_feat[1, yy, xx]
    # Scale feature coords to image coords for arrow placement
    xx_img = (xx + 0.5) * feat_stride
    yy_img = (yy + 0.5) * feat_stride

    plt.figure(figsize=(18,6))
    plt.subplot(1,5,1); plt.title('Image 1'); plt.imshow(img0_disp); plt.axis('off')
    plt.subplot(1,5,2); plt.title('Image 2'); plt.imshow(img1_disp); plt.axis('off')
    plt.subplot(1,5,3); plt.title('Conf (feat res)'); im0 = plt.imshow(conf_feat, vmin=0, vmax=1, cmap='viridis'); plt.axis('off'); plt.colorbar(im0, fraction=0.046, pad=0.04)
    plt.subplot(1,5,4); plt.title('Conf (image res)'); im1 = plt.imshow(conf_up, vmin=0, vmax=1, cmap='viridis'); plt.axis('off'); plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.subplot(1,5,5); plt.title('Flow (quiver+color)'); plt.imshow(flow_color); plt.quiver(xx_img, yy_img, u, v, color='white', angles='xy', scale_units='xy', scale=1, width=0.002, headwidth=3); plt.axis('off')
    plt.tight_layout(); plt.show()

    # Print stats for quick inspection
    print('Confidence stats (feat): min %.3f max %.3f mean %.3f' % (conf_feat.min(), conf_feat.max(), conf_feat.mean()))
    print('Confidence stats (img ): min %.3f max %.3f mean %.3f' % (conf_up.min(), conf_up.max(), conf_up.mean()))
    print('Flow magnitude (px) min %.2f max %.2f mean %.2f' % (np.sqrt(flow_up[0]**2 + flow_up[1]**2).min(),
                                                               np.sqrt(flow_up[0]**2 + flow_up[1]**2).max(),
                                                               np.sqrt(flow_up[0]**2 + flow_up[1]**2).mean()))
    






