import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt

DIM = 384

class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)


class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self, images, patches_per_image=80, disps=None, centroid_sel_strat='RANDOM', return_color=False):
        """ extract patches from input images """
        fmap = self.fnet(images) / 4.0
        imap = self.inet(images) / 4.0

        b, n, c, h, w = fmap.shape
        P = self.patch_size

        # bias patch selection towards regions with high gradient
        if centroid_sel_strat == 'GRADIENT_BIAS':
            g = self.__image_gradient(images)
            x = torch.randint(1, w-1, size=[n, 3*patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, 3*patches_per_image], device="cuda")

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0,:,None], coords, 0).view(n, 3 * patches_per_image)
            
            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        elif centroid_sel_strat == 'RANDOM':
            x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")

        else:
            raise NotImplementedError(f"Patch centroid selection not implemented: {centroid_sel_strat}")

        coords = torch.stack([x, y], dim=-1).float()
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, 128, P, P)

        if return_color:
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P)

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)

        self.DIM = DIM
        self.RES = 4


    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        # images = 2 * (images / 255.0) - 0.5 ## Maps to [-0.5 , 1.5]
        images  = images / 225.0 ## Maps to [0 , 1]
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"), indexing='ij')
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"), indexing='ij')
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"), indexing='ij')

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj


    # @autocast(enabled=False)
    # def forward_deferred_ba(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False, final_ep=20):
    #     """Like forward(), but defers bundle adjustment until the end.

    #     Steps:
    #     - Run STEPS of correlation/update to produce per-edge target and weight.
    #     - Accumulate constraints from each step without calling BA inside the loop.
    #     - Run a single BA over all accumulated constraints.
    #     - Return a trajectory list similar to forward(), plus a final post-BA entry.

    #     Returns:
    #         traj: list of tuples (valid, coords, coords_gt, Gs[:, :n], Ps[:, :n], kl)
    #               One tuple per step (pre-BA), and one final post-BA tuple appended at the end.
    #     """

    #     # Preprocess inputs (same as forward)
    #     images = 2 * (images / 255.0) - 0.5
    #     intrinsics = intrinsics / 4.0
    #     disps = disps[:, :, 1::4, 1::4].float()

    #     fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)
    #     corr_fn = CorrBlock(fmap, gmap)

    #     b, N, c, h, w = fmap.shape
    #     p = self.P

    #     patches_gt = patches.clone()
    #     Ps = poses

    #     # Initialize patch depths with random values at center pixel
    #     d = patches[..., 2, p//2, p//2]
    #     patches = set_depth(patches, torch.rand_like(d))

    #     # Build initial edge set over first 8 frames
    #     kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0, 8, device="cuda"), indexing='ij')
    #     ii = ix[kk]

    #     # Feature/hidden states for update
    #     imap = imap.view(b, -1, DIM)
    #     net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)

    #     # Initialize poses
    #     Gs = SE3.IdentityLike(poses)
    #     if structure_only:
    #         Gs.data[:] = poses.data[:]

    #     # Accumulators for a single BA at the end
    #     ii_list, jj_list, kk_list = [], [], []
    #     target_list, weight_list = [], []

    #     traj = []
    #     bounds = [-64, -64, w + 64, h + 64]

    #     # Main iterative loop without BA
    #     while len(traj) < STEPS:
    #         # Detach to avoid building huge graphs
    #         Gs = Gs.detach()
    #         patches = patches.detach()

    #         # Possibly introduce a new frame (same logic as forward)
    #         n = ii.max() + 1
    #         if len(traj) >= 8 and n < images.shape[1]:
    #             if not structure_only:
    #                 Gs.data[:, n] = Gs.data[:, n - 1]
    #             kk1, jj1 = flatmeshgrid(torch.where(ix < n)[0], torch.arange(n, n + 1, device="cuda"), indexing='ij')
    #             kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n + 1, device="cuda"), indexing='ij')

    #             ii = torch.cat([ix[kk1], ix[kk2], ii])
    #             jj = torch.cat([jj1, jj2, jj])
    #             kk = torch.cat([kk1, kk2, kk])

    #             net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
    #             net = torch.cat([net1, net], dim=1)

    #             if np.random.rand() < 0.1:
    #                 k = (ii != (n - 4)) & (jj != (n - 4))
    #                 ii = ii[k]
    #                 jj = jj[k]
    #                 kk = kk[k]
    #                 net = net[:, k]

    #             # Initialize depths for patches in the new frame from median of previous frames
    #             patches[:, ix == n, 2] = torch.median(patches[:, (ix == n - 1) | (ix == n - 2), 2])
    #             n = ii.max() + 1

    #         # Predict correspondences and residual corrections
    #         coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
    #         coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

    #         corr = corr_fn(kk, jj, coords1)
    #         net, (delta, weight, _) = self.update(net, imap[:, kk], corr, None, ii, jj, kk)

    #         # Accumulate constraints for final BA
    #         lmbda = 1e-4
    #         target = coords[..., p // 2, p // 2, :] + delta

    #         ii_list.append(ii.clone())
    #         jj_list.append(jj.clone())
    #         kk_list.append(kk.clone())
    #         target_list.append(target)
    #         weight_list.append(weight)

    #         # For logging/compatibility, compute the same per-step traj tuple (pre-BA)
    #         kl = torch.as_tensor(0)
    #         dij = (ii - jj).abs()
    #         k = (dij > 0) & (dij <= 2)

    #         coords_log = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
    #         coords_gt_log, valid_log, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

    #         traj.append((valid_log, coords_log, coords_gt_log, Gs[:, :n], Ps[:, :n], kl))

    #     # Run a single BA at the end over all accumulated constraints
    #     ii_cat = torch.cat(ii_list, dim=0) if len(ii_list) > 0 else ii
    #     jj_cat = torch.cat(jj_list, dim=0) if len(jj_list) > 0 else jj
    #     kk_cat = torch.cat(kk_list, dim=0) if len(kk_list) > 0 else kk
    #     target_cat = torch.cat(target_list, dim=1) if len(target_list) > 0 else target
    #     weight_cat = torch.cat(weight_list, dim=1) if len(weight_list) > 0 else weight

    #     # Do one bigger BA to refine Gs and structure
    #     Gs, patches = BA(Gs, patches, intrinsics, target_cat, weight_cat, lmbda,
    #                      ii_cat, jj_cat, kk_cat, bounds, ep=final_ep, fixedp=1,
    #                      structure_only=structure_only)

    #     # Append a final post-BA trajectory entry for convenience
    #     final_n = int(ii_cat.max().item()) + 1 if ii_cat.numel() > 0 else int(ii.max().item()) + 1
    #     dij = (ii_cat - jj_cat).abs()
    #     k = (dij > 0) & (dij <= 2)

    #     coords_final = pops.transform(Gs, patches, intrinsics, ii_cat[k], jj_cat[k], kk_cat[k])
    #     coords_gt_final, valid_final, _ = pops.transform(Ps, patches_gt, intrinsics, ii_cat[k], jj_cat[k], kk_cat[k], jacobian=True)

    #     traj.append((valid_final, coords_final, coords_gt_final, Gs[:, :final_n], Ps[:, :final_n], torch.as_tensor(0)))

    #     return traj

