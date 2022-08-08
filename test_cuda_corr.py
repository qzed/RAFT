#!/usr/bin/env python3

from __future__ import print_function, division
import sys
sys.path.append('core')

import torch
import torch.nn.functional as F

import corr
from utils.utils import coords_grid, bilinear_sampler


def compute_corr_vol(fmap1, fmap2, level):
    cr = corr.CorrBlock.corr(fmap1, fmap2)                  # b, h1, w1, 1, h1, w1

    batch, h1, w1, dim, h2, w2 = cr.shape
    cr = cr.reshape(batch*h1*w1, dim, h2, w2)               # (b*h1*w1, 1, h1, w1)

    for _ in range(level):
        cr = F.avg_pool2d(cr, 2, stride=2)                  # (b*h1*w1, 1, h2, w2)

    return cr


def sample(fmap1, fmap2, coords, radius, level):
    b, _, hc, wc = coords.shape

    cr = compute_corr_vol(fmap1, fmap2, level)              # (b*h1*w1, 1, h2, w2)

    coords = coords.permute(0, 2, 3, 1)                     # (b, hc, wc, 2)
    coords = coords.reshape(b, 1, hc, wc, 2)                # (b, 1, hc, wc, 2)
    coords = coords / 2**level

    dx = torch.linspace(-radius, radius, 2*radius+1, device=coords.device)
    dy = torch.linspace(-radius, radius, 2*radius+1, device=coords.device)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)    # (2r+1, 2r+1, 2)

    centroid_lvl = coords.reshape(b*hc*wc, 1, 1, 2)         # (b*hc*wc,    1,    1, 2)
    delta_lvl = delta.view(1, 2*radius+1, 2*radius+1, 2)    # (      1, 2r+1, 2r+1, 2)
    coords_lvl = centroid_lvl + delta_lvl                   # (b*hc*wc, 2r+1, 2r+1, 2)

    cr = bilinear_sampler(cr, coords_lvl)                   # (b*hc*wc, 1, 2r+1, 2r+1)
    cr = cr.reshape(b, hc, wc, 2*radius + 1, 2*radius + 1)  # (b, hc, wc, 2r+1, 2r+1)

    return cr.permute(0, 1, 2, 4, 3)


def alt_corr(fmap1, fmap2, coords, radius, level):
    b, _, hc, wc = coords.shape
    _, c, h1, w1 = fmap1.shape

    for _ in range(level):
        fmap2 = F.avg_pool2d(fmap2, 2, stride=2)            # (b, c, h2, w2)

    fmap1 = fmap1.permute(0, 2, 3, 1).contiguous()          # b, h1, w1, c
    fmap2 = fmap2.permute(0, 2, 3, 1).contiguous()          # b, h1, w1, c

    coords = coords.permute(0, 2, 3, 1)                     # (b, hc, wc, 2)
    coords = coords.reshape(b, 1, hc, wc, 2)                # (b, 1, hc, wc, 2)
    coords = coords / 2**level
    coords = coords.contiguous()

    cr = corr.alt_corr(fmap1, fmap2, coords, radius)        # (b, 1, (2r+1)**2, hc, wc)
    cr = cr / torch.sqrt(torch.tensor(c).float())
    cr = cr.reshape(b, 2*radius + 1, 2*radius + 1, hc, wc)  # (b, 2r+1, 2r+1, hc, wc)
    cr = cr.permute(0, 3, 4, 1, 2)                          # (b, hc, wc, 2r+1, 2r+1)

    return cr.permute(0, 1, 2, 4, 3)


def main():
    b = 1
    h1, w1 = 16, 16
    c = 32
    level = 0
    radius = 3
    dr = 0

    fmap1 = torch.randn(b, c, h1, w1).cuda()                # b, c, h1, w1
    fmap2 = torch.randn(b, c, h1, w1).cuda()                # b, c, h1, w1

    coords = coords_grid(b, h1, w1, device=fmap1.device)    # (b, 2, h1, w1)
    coords += torch.randn_like(coords) * 3

    # print("fmap1:", fmap1.shape)
    # print(fmap1)
    # print()
    # print("fmap2:", fmap2.shape)
    # print(fmap2)
    # print()

    # print("Coordinates:", coords.shape)
    # print(coords)
    # print()

    fmap1_o = fmap1.clone().detach()
    fmap2_o = fmap2.clone().detach()
    coords_o = coords.clone().detach()

    fmap1_o.requires_grad = True
    fmap2_o.requires_grad = True
    coords_o.requires_grad = True

    cr_orig = sample(fmap1_o, fmap2_o, coords_o, radius, level)
    # print("Python:", cr_orig.shape)
    # print(cr_orig)
    # print()

    fmap1_c = fmap1.clone().detach()
    fmap2_c = fmap2.clone().detach()
    coords_c = coords.clone().detach()

    fmap1_c.requires_grad = True
    fmap2_c.requires_grad = True
    coords_c.requires_grad = True

    cr_alt = alt_corr(fmap1_c, fmap2_c, coords_c, radius, level)
    # print("CUDA:", cr_alt.shape)
    # print(cr_alt)
    # print()

    print("cr_orig.absmean:", cr_orig.abs().mean().item())
    print("cr_alt.absmean:", cr_alt.abs().mean().item())

    err_fwd = torch.sum(((cr_alt - cr_orig)**2)).item() / (h1 * w1 * (2 * radius + 1)**2)
    print("Forward pass mse:", err_fwd)
    print()

    # print()
    # print("Backward pass:")
    # print()

    d_fmap1_o, d_fmap2_o, d_coords_o = torch.autograd.grad(cr_orig.sum(), (fmap1_o, fmap2_o, coords_o))
    # print("original:")
    # print(d_fmap1_o)
    # print(d_fmap2_o)
    # print(d_coords_o)
    # print()

    d_fmap1_c, d_fmap2_c, d_coords_c = torch.autograd.grad(cr_alt.sum(), (fmap1_c, fmap2_c, coords_c))
    # print("cuda:")
    # print(d_fmap1_c)
    # print(d_fmap2_c)
    # print(d_coords_c)
    # print()

    print("d_fmap1_c.absmean:", d_fmap1_c.abs().mean().item())
    print("d_fmap2_c.absmean:", d_fmap2_c.abs().mean().item())
    print("d_coords_c.absmean:", d_coords_c.abs().mean().item())
    print("d_fmap1_o.absmean:", d_fmap1_o.abs().mean().item())
    print("d_fmap2_o.absmean:", d_fmap2_o.abs().mean().item())
    print("d_coords_o.absmean:", d_coords_o.abs().mean().item())

    err_bwd_f1 = torch.sum(((d_fmap1_c - d_fmap1_o)**2)).item() / (torch.prod(torch.tensor(d_fmap1_c.shape)))
    err_bwd_f2 = torch.sum(((d_fmap2_c - d_fmap2_o)**2)).item() / (torch.prod(torch.tensor(d_fmap2_c.shape)))
    err_bwd_crd = torch.sum(((d_coords_c - d_coords_o)**2)).item() / (torch.prod(torch.tensor(d_coords_c.shape)))

    print("Backward pass mse: fmap1", err_bwd_f1.item())
    print("Backward pass mse: fmap2", err_bwd_f2.item())
    print("Backward pass mse: coords", err_bwd_crd.item())
    print()

    def sample_wrapped(*inp):
        return sample(*inp, radius=radius, level=level)

    def alt_wrapped(*inp):
        return alt_corr(*inp, radius=radius, level=level)

    inputs = (fmap1_o, fmap2_o, coords_o)
    j_fmap1_o, j_fmap2_o, j_coords_o = torch.autograd.functional.jacobian(sample_wrapped, inputs)

    inputs = (fmap1_c, fmap2_c, coords_c)
    j_fmap1_c, j_fmap2_c, j_coords_c = torch.autograd.functional.jacobian(alt_wrapped, inputs)

    print("j_fmap1_o.absmean:", j_fmap1_o.abs().mean().item())
    print("j_fmap2_o.absmean:", j_fmap2_o.abs().mean().item())
    print("j_coords_o.absmean:", j_coords_o.abs().mean().item())
    print("j_fmap1_c.absmean:", j_fmap1_c.abs().mean().item())
    print("j_fmap2_c.absmean:", j_fmap2_c.abs().mean().item())
    print("j_coords_c.absmean:", j_coords_c.abs().mean().item())

    err_j_f1 = torch.sum(((j_fmap1_c - j_fmap1_o)**2)).item() / (torch.prod(torch.tensor(j_fmap1_c.shape)))
    err_j_f2 = torch.sum(((j_fmap2_c - j_fmap2_o)**2)).item() / (torch.prod(torch.tensor(j_fmap2_c.shape)))
    err_j_crd = torch.sum(((j_coords_c - j_coords_o)**2)).item() / (torch.prod(torch.tensor(j_coords_c.shape)))

    print("Jacobian mse: fmap1", err_j_f1.item())
    print("Jacobian mse: fmap2", err_j_f2.item())
    print("Jacobian mse: coords", err_j_crd.item())
    print()

#    print()
#    print("gradcheck:")
#
#    coords_gc = coords_c.double().detach()
#    fmap1_gc = fmap1_c.double().detach()
#    fmap2_gc = fmap2_c.double().detach()
#
#    fmap1_gc.requires_grad = True
#    fmap2_gc.requires_grad = True
#    coords_gc.requires_grad = False
#
#    inputs = (fmap1_gc, fmap2_gc, coords_gc, radius, level)
#    torch.autograd.gradcheck(alt_corr, inputs, atol=0.3)


if __name__ == '__main__':
    main()
