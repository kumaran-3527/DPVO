import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def boundary_sensitive_sample_coords(
    seg_maps: torch.Tensor,
    grad_map: torch.Tensor,
    feature_hw: tuple,
    patches_per_image: int,
    batch_index: int = 0,
    dilate: int = 2,
    device: torch.device | None = None,
):
    """
    Sample patch centroids using gradient confidence while suppressing segmentation boundaries.

    Args:
        seg_maps: (B, N, H, W) integer tensor of semantic ids (from Mask2Former)
        grad_map: gradient magnitude map from images, shape (B, N, h', w')
        feature_hw: target feature resolution (h, w) for sampling
        patches_per_image: number of patches per frame (N)
        batch_index: which batch index to sample from (matches downstream fmap[0], imap[0])
        dilate: boundary mask dilation radius (pixels)
        device: output device; defaults to grad_map.device

    Returns:
        x, y: each LongTensor of shape (N, patches_per_image), device=device
    """

    if device is None:
        device = grad_map.device

    B, N = seg_maps.shape[:2]
    h, w = feature_hw

    # 1) Boundary mask at feature scale (h, w)
    bndry_masks = seg_to_boundary_mask(seg_maps, dilate=dilate, target_size=(h, w))  # numpy (B*N, h, w)
    bndry_masks = torch.from_numpy(bndry_masks).to(torch.bool).to(device)  # (B*N, h, w)

    # 2) Confidence map from image gradients, resized to (h, w) and normalized per frame
    g = grad_map  # (B, N, ?, ?)
    g = g.view(B * N, 1, g.shape[-2], g.shape[-1])
    g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=False).squeeze(1)  # (B*N, h, w)

    g_flat = g.view(B * N, -1)
    g_min = g_flat.min(dim=1).values.view(B * N, 1, 1)
    g_max = g_flat.max(dim=1).values.view(B * N, 1, 1)
    conf_map = (g - g_min) / (g_max - g_min + 1e-8)  # (B*N, h, w)

    final_conf_map = conf_map * (~bndry_masks).float()  # zero-out boundaries, (B*N, h, w)

    # Use batch_index to match downstream usage of fmap[0], imap[0], images[0]
    final_conf_map = final_conf_map.view(B, N, h, w)[batch_index]  # (N, h, w)
    bndry_masks_b = bndry_masks.view(B, N, h, w)[batch_index]      # (N, h, w)

    # Grid params
    G = max(1, int(np.sqrt(patches_per_image)))
    cells = G * G
    top_k_per_cell = max(1, patches_per_image // cells)
    cand_per_cell = 2 * top_k_per_cell

    x = torch.empty(N, patches_per_image, device=device, dtype=torch.long)
    y = torch.empty(N, patches_per_image, device=device, dtype=torch.long)

    cell_h = max(1, h // G)
    cell_w = max(1, w // G)

    for i in range(N):
        conf_i = final_conf_map[i]            # (h, w)
        bndry_i = bndry_masks_b[i]            # (h, w)

        # valid mask excludes boundaries and 1-pixel border for patch context
        valid_i = (~bndry_i).clone()
        if valid_i.numel() > 0:
            valid_i[0, :] = False
            valid_i[-1, :] = False
            valid_i[:, 0] = False
            valid_i[:, -1] = False

        sel_x, sel_y, sel_s = [], [], []

        # sample within each grid cell
        for gy in range(G):
            for gx in range(G):
                y0 = gy * cell_h
                y1 = h if gy == G - 1 else (gy + 1) * cell_h
                x0 = gx * cell_w
                x1 = w if gx == G - 1 else (gx + 1) * cell_w

                # keep inside [1, h-1) and [1, w-1)
                y_low = max(1, y0)
                y_high = min(h - 1, y1)
                x_low = max(1, x0)
                x_high = min(w - 1, x1)

                if y_high <= y_low or x_high <= x_low:
                    continue

                cy = torch.randint(y_low, y_high, (cand_per_cell,), device=device)
                cx = torch.randint(x_low, x_high, (cand_per_cell,), device=device)

                # score and rank inside cell (final_conf_map already zeros boundaries)
                scores = conf_i[cy, cx]
                if cand_per_cell > 0:
                    k = min(top_k_per_cell, cand_per_cell)
                    topk = torch.topk(scores, k=k, largest=True).indices
                    sel_y.append(cy[topk])
                    sel_x.append(cx[topk])
                    sel_s.append(scores[topk])

        if len(sel_x) > 0:
            sel_x = torch.cat(sel_x)
            sel_y = torch.cat(sel_y)
            sel_s = torch.cat(sel_s)
        else:
            sel_x = torch.empty(0, dtype=torch.long, device=device)
            sel_y = torch.empty(0, dtype=torch.long, device=device)
            sel_s = torch.empty(0, dtype=conf_i.dtype, device=device)

        # fill if needed using global top on valid_i
        if sel_x.numel() < patches_per_image:
            need = patches_per_image - sel_x.numel()
            conf_fill = conf_i.clone()
            conf_fill[~valid_i] = -1e9
            flat = conf_fill.view(-1)
            k = min(need, flat.numel())
            if k > 0:
                top_idx = torch.topk(flat, k=k, largest=True).indices
                fy = top_idx // w
                fx = top_idx % w
                sel_x = torch.cat([sel_x, fx])
                sel_y = torch.cat([sel_y, fy])
                sel_s = torch.cat([sel_s, conf_i[fy, fx]])

        # trim to exact count by highest scores
        if sel_x.numel() > patches_per_image:
            k = patches_per_image
            top = torch.topk(sel_s, k=k, largest=True).indices
            sel_x = sel_x[top]
            sel_y = sel_y[top]

        # final fallback: if still short, pad randomly within valid area
        if sel_x.numel() < patches_per_image:
            need = patches_per_image - sel_x.numel()
            vy, vx = torch.where(valid_i)
            if vy.numel() > 0:
                ridx = torch.randint(0, vy.numel(), (need,), device=device)
                sel_y = torch.cat([sel_y, vy[ridx]])
                sel_x = torch.cat([sel_x, vx[ridx]])
            else:
                # last resort: random anywhere inside safe margin
                ry = torch.randint(1, h - 1, (need,), device=device)
                rx = torch.randint(1, w - 1, (need,), device=device)
                sel_y = torch.cat([sel_y, ry])
                sel_x = torch.cat([sel_x, rx])

        x[i] = sel_x[:patches_per_image]
        y[i] = sel_y[:patches_per_image]

    return x, y



@torch.no_grad()
def seg_to_boundary_mask(seg_maps, dilate=2, target_size =None):
        """
        Compute boundary mask from segmentation map.
        Args:
            seg_map: BxNxHxW numpy array of integer segment IDs
            dilate: dilation radius in pixels (default=2)
        Returns:
            boundary_masks: B x N x H x W numpy array, 1 on boundaries, 0 elsewhere
        """
        
        b,n,h,w = seg_maps.shape
        seg_maps = seg_maps.view(b*n, h, w).cpu().numpy()

        boundary_masks = np.zeros((b*n, h, w), dtype=np.uint8)

  
        boundary_masks[:, :, :-1] |= (seg_maps[:, :, :-1] != seg_maps[:, :, 1:])  # right neighbor
        boundary_masks[:, :-1, :] |= (seg_maps[:, :-1, :] != seg_maps[:, 1:, :])  # bottom neighbor
        boundary_masks[:, :, 1:] |= (seg_maps[:, :, 1:] != seg_maps[:, :, :-1])  # left neighbor
        boundary_masks[:, 1:, :] |= (seg_maps[:, 1:, :] != seg_maps[:, :-1, :])  # top neighbor

        if dilate > 0:
            kernel = torch.ones((2 * dilate + 1, 2 * dilate + 1), dtype=torch.float32)
            boundary_tensor = torch.from_numpy(boundary_masks).float().unsqueeze(0).unsqueeze(0)
            boundary_tensor = torch.nn.functional.conv2d(boundary_tensor, kernel.unsqueeze(0).unsqueeze(0), padding=dilate)
            boundary_masks = (boundary_tensor.squeeze() > 0).numpy().astype(np.uint8)

        if target_size is not None and (target_size[0] != h or target_size[1] != w):
            boundary_masks = torch.nn.functional.interpolate(torch.from_numpy(boundary_masks).unsqueeze(0).unsqueeze(0).int(), size=target_size, mode='nearest').squeeze().numpy().astype(np.uint8)
        
        
        return boundary_masks