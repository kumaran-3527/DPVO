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

from .new_encoders_1 import BackBone, Mask2Former, SimpleFeatureFusion

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt

DIM = 384
MODEL_PATH = "seg_models/facebook/mask2former-swin-small-coco-panoptic"
SEG_FP_16 = True

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
        # self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        # self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')

        # self.fnet = BackBone(model_path= MODEL_PATH, device='cuda', eval=True, capture_encoder=True, fp_16 = SEG_FP_16 , out_dim=128)
        # self.inet = BackBone(model_path = MODEL_PATH,device='cuda', eval=True, capture_encoder=True, fp_16 = SEG_FP_16 , out_dim=386)

        self.mask2former = Mask2Former(model_path= MODEL_PATH, device='cuda', eval=True, capture_encoder=True, fp_16 = SEG_FP_16 ).eval()
        self.fnet = SimpleFeatureFusion(out_dim=128)
        self.inet  = SimpleFeatureFusion(out_dim=DIM)


    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g
    

    def forward(self, images, patches_per_image=80, disps=None, centroid_sel_strat='RANDOM', return_color=False):
        """ extract patches from input images """
        
        B, N, C, H, W = images.shape

        with torch.no_grad():
            seg_maps , enc_feats = self.mask2former(images)

        fmap = self.fnet(enc_feats[0], enc_feats[1], enc_feats[2], out_size=(H//8, W//8), interp_mode='bilinear')  # (B*N,128,H/8,W/8)
        fmap = fmap.view(B, N, fmap.shape[1], fmap.shape[2], fmap.shape[3]).to(torch.float32)

        imap = self.inet(enc_feats[0], enc_feats[1], enc_feats[2], out_size=(H//8, W//8), interp_mode='bilinear') 
        imap = imap.view(B, N, imap.shape[1], imap.shape[2], imap.shape[3]).to(torch.float32)  # (B*N,D,H/8,W/8)

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
 
        
        # elif centroid_sel_strat == 'BOUNDARY_SENSITIVE'

        # elif centroid_sel_strat == 'SEMANTIC_BIAS':



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

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl, net))

        return traj


if __name__ == "__main__":


    import torch

    model = VONet()  # your model
    model.train()    # enable training mode globally

    # 1) Keep Mask2Former frozen and in eval mode
    model.patchify.mask2former.eval()
    for p in model.patchify.mask2former.parameters():
        p.requires_grad_(False)

    # 2) Ensure fusion modules are trainable
    model.patchify.fnet.train()
    model.patchify.inet.train()
    for p in model.patchify.fnet.parameters():
        p.requires_grad_(True)
    for p in model.patchify.inet.parameters():
        p.requires_grad_(True)

    # 3) Build optimizer over only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

    # (Optional) separate LR for fusion vs rest:
    fusion_params = list(model.patchify.fnet.parameters()) + list(model.patchify.inet.parameters())
    rest_params = [p for p in model.parameters() if p.requires_grad and p not in fusion_params]
    optimizer = torch.optim.Adam(
        [
            {"params": fusion_params, "lr": 1e-4},
            {"params": rest_params, "lr": 1e-4},  # or a different LR if you like
        ]
    )

    # 4) Safety: if you ever call model.train() again (e.g., each epoch), re-pin mask2former to eval:
    # model.train()
    # model.patchify.mask2former.eval()
    # for p in model.patchify.mask2former.parameters():
    #     p.requires_grad_(False)

    # 5) Gradient sanity check (optional)
    dummy_images = torch.randn(1, 8, 3, 256, 320, device="cuda")
    dummy_poses = SE3.IdentityLike(torch.zeros(1,8, device="cuda"))
    dummy_disps = torch.ones(1, 8, 256, 320, device="cuda")
    dummy_intr = torch.tensor([[[320., 0., 160.], [0., 320., 128.], [0., 0., 1.]]], device="cuda")

    model.cuda()
    out = model(dummy_images, dummy_poses, dummy_disps, dummy_intr)  # your normal forward & loss path
    loss = out[0][0].shape[0] * 0.001  # dummy loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


    # assert any grad in fusion, none in mask2former
    assert any(p.grad is not None for p in model.patchify.fnet.parameters()), "fnet not receiving grads"
    assert any(p.grad is not None for p in model.patchify.inet.parameters()), "inet not receiving grads"
    assert all(p.grad is None for p in model.patchify.mask2former.parameters()), "mask2former should be frozen"
