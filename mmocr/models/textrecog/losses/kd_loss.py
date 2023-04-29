import torch
from torch.nn import functional as F
lambda_f_base = 1
lambda_c_base = 1
def KD_Loss(old_features, features):
    B = features.shape[0]
    flat_loss = (
            F.cosine_embedding_loss(
                features.view(B,-1),
                old_features.detach().view(B,-1),
                torch.ones(features.shape[0]).to(features.device),
            )
            * lambda_f_base
    )
    spatial_loss = (
            pod_spatial_lossv2([old_features],[features]) * lambda_c_base
    )
    loss = spatial_loss + flat_loss
    return loss

def pod_spatial_loss(old_fmaps, fmaps, normalize=True):
    """
    a, b: list of [bs, c, w, h]
    """
    loss = torch.tensor(0.0).to(fmaps[0].device)
    for i, (a, b) in enumerate(zip(old_fmaps, fmaps)):
        assert a.shape == b.shape, "Shape error"

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        a_h = a.sum(dim=3).view(a.shape[0], -1)  # [bs, c*w]
        b_h = b.sum(dim=3).view(b.shape[0], -1)  # [bs, c*w]
        a_w = a.sum(dim=2).view(a.shape[0], -1)  # [bs, c*h]
        b_w = b.sum(dim=2).view(b.shape[0], -1)  # [bs, c*h]

        a = torch.cat([a_h, a_w], dim=-1)
        b = torch.cat([b_h, b_w], dim=-1)

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(fmaps)

def pod_spatial_lossv2(old_fmaps, fmaps, normalize=True):
    """
    a, b: list of [bs, c, w, h]
    """
    # b, c, w, h = fmaps.shape
    # pool_12 = torch.nn.AdaptiveMaxPool2d(w // 2, h)

    loss = torch.tensor(0.0).to(fmaps[0].device)
    for i, (a, b) in enumerate(zip(old_fmaps, fmaps)):
        B, c, w, h = a.shape
        pool_22 = torch.nn.AdaptiveAvgPool2d((w // 2, h // 2))
        assert a.shape == b.shape, "Shape error"

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        a_wh = a.sum(dim=1).view(a.shape[0], -1)  # [bs, h*w]
        b_wh = b.sum(dim=1).view(b.shape[0], -1)  # [bs, h*w]

        # # at_wh = pool_22(a).sum(dim=1).view(a.shape[0], -1)  # [bs, c*h]
        # # bt_wh = pool_22(b).sum(dim=1).view(b.shape[0], -1)  # [bs, c*h]
        #
        # a = torch.cat([a_wh, at_wh], dim=-1)
        # b = torch.cat([b_wh, bt_wh], dim=-1)
        a = a_wh
        b = b_wh

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(fmaps)