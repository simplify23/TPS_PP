import cv2
import numpy as np
from torch import nn
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageOps
from mmdet.datasets.builder import PIPELINES
DEVICE = torch.device('cuda:0')
# '''
#     PIL resize (W,H)
#     Torch resize is (H,W)
# '''
def norm(points_int, width, height,DEVICE):
    """
        将像素点坐标归一化至 -1 ~ 1
    """
    points_int_clone = torch.from_numpy(points_int).detach().float().to(DEVICE)
    x = ((points_int_clone * 2)[..., 0] / (width - 1) - 1)
    y = ((points_int_clone * 2)[..., 1] / (height - 1) - 1)
    return torch.stack([x, y], dim=-1).contiguous().view(-1, 2)

class TPS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y, w, h, device):

        """ 计算grid"""
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        grid = grid.view(-1, h * w, 2)

        """ 计算W, A"""
        n, k = X.shape[:2]
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)

        eps = 1e-9
        D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        K = D2 * torch.log(D2 + eps)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        Q = torch.solve(Z, L)[0]
        W, A = Q[:, :k], Q[:, k:]

        """ 计算U """
        eps = 1e-9
        D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        U = D2 * torch.log(D2 + eps)

        """ 计算P """
        n, k = grid.shape[:2]
        device = grid.device
        P = torch.ones(n, k, 3, device=device)
        P[:, :, 1:] = grid

        # grid = P @ A + U @ W
        grid = torch.matmul(P, A) + torch.matmul(U, W)
        return grid.view(-1, h, w, 2)



@PIPELINES.register_module()
class Stretch:
    def __init__(self):
        # self.tps = tps
        self.tps = None
        # self.tps = cv2.createThinPlateSplineShapeTransformer()

    def __call__(self, result, mag=-1, prob=1.):
        # print("stretch\n")
        # result['img_origin'] = result['img']
        if np.random.uniform(0, 1) > prob:
            return result
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        img = result['img'][..., ::-1]
        H, W = img.shape[:2]
        # W, H = img.size
        img = np.array(img)
        srcpt = list()
        dstpt = list()

        W_33 = 0.33 * W
        W_50 = 0.50 * W
        W_66 = 0.66 * W

        H_50 = 0.50 * H

        P = 0
        # frac = 0.4

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = len(b) - 1
        else:
            index = mag
        frac = b[index]

        # left-most
        srcpt.append([P, P])
        srcpt.append([P, H - P])
        srcpt.append([P, H_50])
        x = np.random.uniform(0, frac) * W_33  # if np.random.uniform(0,1) > 0.5 else 0
        dstpt.append([P + x, P])
        dstpt.append([P + x, H - P])
        dstpt.append([P + x, H_50])

        # 2nd left-most
        srcpt.append([P + W_33, P])
        srcpt.append([P + W_33, H - P])
        x = np.random.uniform(-frac, frac) * W_33
        dstpt.append([P + W_33 + x, P])
        dstpt.append([P + W_33 + x, H - P])

        # 3rd left-most
        srcpt.append([P + W_66, P])
        srcpt.append([P + W_66, H - P])
        x = np.random.uniform(-frac, frac) * W_33
        dstpt.append([P + W_66 + x, P])
        dstpt.append([P + W_66 + x, H - P])

        # right-most
        srcpt.append([W - P, P])
        srcpt.append([W - P, H - P])
        srcpt.append([W - P, H_50])
        x = np.random.uniform(-frac, 0) * W_33  # if np.random.uniform(0,1) > 0.5 else 0
        dstpt.append([W - P + x, P])
        dstpt.append([W - P + x, H - P])
        dstpt.append([W - P + x, H_50])

        N = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(N)]
        dst_shape = np.array(dstpt).reshape((-1, N, 2))
        src_shape = np.array(srcpt).reshape((-1, N, 2))

        # ten_img = ToTensor()(img).to(DEVICE)
        # h, w = ten_img.shape[1], ten_img.shape[2]
        # ten_source = norm(dst_shape, w, h,DEVICE)
        # ten_target = norm(src_shape, w, h,DEVICE)
        #
        # # tps = TPS()
        # warped_grid = self.tps(ten_target[None, ...], ten_source[None, ...], w, h, DEVICE)  # 这个输入的位置需要归一化，所以用norm
        # ten_wrp = torch.grid_sampler_2d(ten_img[None, ...], warped_grid, 0, 0,align_corners=True)
        # img = np.array(ToPILImage()(ten_wrp[0].cpu()))

        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)[..., ::-1]
        # img = Image.fromarray(img)
        result['img'] = img
        # print(img)
        # print(img.size)
        result['img_shape'] = img.shape
        return result

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class Distort:
    def __init__(self):
        self.tps = None
        # self.tps = TPS()
        # self.tps = cv2.createThinPlateSplineShapeTransformer()

    def __call__(self, result, mag=-1, prob=1.):
        # print("distort\n")
        if np.random.uniform(0, 1) > prob:
            return result
        # result['img_origin'] = result['img']
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        img = result['img'][..., ::-1]
        H, W = img.shape[:2]
        # W, H = img.size
        img = np.array(img)
        srcpt = list()
        dstpt = list()

        W_33 = 0.33 * W
        W_50 = 0.50 * W
        W_66 = 0.66 * W

        H_50 = 0.50 * H

        P = 0
        # frac = 0.4

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = len(b) - 1
        else:
            index = mag
        frac = b[index]

        # top pts
        srcpt.append([P, P])
        x = np.random.uniform(0, frac) * W_33
        y = np.random.uniform(0, frac) * H_50
        dstpt.append([P + x, P + y])

        srcpt.append([P + W_33, P])
        x = np.random.uniform(-frac, frac) * W_33
        y = np.random.uniform(0, frac) * H_50
        dstpt.append([P + W_33 + x, P + y])

        srcpt.append([P + W_66, P])
        x = np.random.uniform(-frac, frac) * W_33
        y = np.random.uniform(0, frac) * H_50
        dstpt.append([P + W_66 + x, P + y])

        srcpt.append([W - P, P])
        x = np.random.uniform(-frac, 0) * W_33
        y = np.random.uniform(0, frac) * H_50
        dstpt.append([W - P + x, P + y])

        # bottom pts
        srcpt.append([P, H - P])
        x = np.random.uniform(0, frac) * W_33
        y = np.random.uniform(-frac, 0) * H_50
        dstpt.append([P + x, H - P + y])

        srcpt.append([P + W_33, H - P])
        x = np.random.uniform(-frac, frac) * W_33
        y = np.random.uniform(-frac, 0) * H_50
        dstpt.append([P + W_33 + x, H - P + y])

        srcpt.append([P + W_66, H - P])
        x = np.random.uniform(-frac, frac) * W_33
        y = np.random.uniform(-frac, 0) * H_50
        dstpt.append([P + W_66 + x, H - P + y])

        srcpt.append([W - P, H - P])
        x = np.random.uniform(-frac, 0) * W_33
        y = np.random.uniform(-frac, 0) * H_50
        dstpt.append([W - P + x, H - P + y])

        N = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(N)]
        dst_shape = np.array(dstpt).reshape((-1, N, 2))
        src_shape = np.array(srcpt).reshape((-1, N, 2))

        # ten_img = ToTensor()(img).to(DEVICE)
        # h, w = ten_img.shape[1], ten_img.shape[2]
        # ten_source = norm(dst_shape, w, h,DEVICE)
        # ten_target = norm(src_shape, w, h,DEVICE)
        #
        # # tps = TPS()
        # warped_grid = self.tps(ten_target[None, ...], ten_source[None, ...], w, h, DEVICE)  # 这个输入的位置需要归一化，所以用norm
        # ten_wrp = torch.grid_sampler_2d(ten_img[None, ...], warped_grid, 0, 0,align_corners=True)
        # img = np.array(ToPILImage()(ten_wrp[0].cpu()))

        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        # img = Image.fromarray(img)
        result['img'] = img[..., ::-1]
        # print(img)
        # print(img.size)
        result['img_shape'] = img.shape
        return result

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class Curve:
    def __init__(self, square_side=224):
        self.tps = None
        # self.tps = cv2.createThinPlateSplineShapeTransformer()
        self.side = square_side
        # self.w_side = 128

    def __call__(self, result, mag=-1, prob=1.):
        # print("curve\n")
        # result['img_origin'] = result['img']
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        if np.random.uniform(0, 1) > prob:
            return result
        img = result['img'][..., ::-1]
        H, W = img.shape[:2]
        # W, H = img.size

        if H != self.side or W != self.side:
            img = cv2.resize(img, (self.side, self.side), Image.BICUBIC)

        isflip = np.random.uniform(0, 1) > 0.5
        # isflip = False
        if isflip:
            img = cv2.flip(img, 0)
            # img = ImageOps.flip(img)
            # img = TF.vflip(img)

        img = np.array(img)
        w = self.side
        h = self.side
        w_25 = 0.25 * w
        w_50 = 0.50 * w
        w_75 = 0.75 * w

        b = [1.1, .95, .8]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        rmin = b[index]

        r = np.random.uniform(rmin, rmin + .1) * h
        x1 = (r ** 2 - w_50 ** 2) ** 0.5
        h1 = r - x1

        t = np.random.uniform(0.4, 0.5) * h

        w2 = w_50 * t / r
        hi = x1 * t / r
        h2 = h1 + hi

        sinb_2 = ((1 - x1 / r) / 2) ** 0.5
        cosb_2 = ((1 + x1 / r) / 2) ** 0.5
        w3 = w_50 - r * sinb_2
        h3 = r - r * cosb_2

        w4 = w_50 - (r - t) * sinb_2
        h4 = r - (r - t) * cosb_2

        w5 = 0.5 * w2
        h5 = h1 + 0.5 * hi
        h_50 = 0.50 * h

        srcpt = [(0, 0), (w, 0), (w_50, 0), (0, h), (w, h), (w_25, 0), (w_75, 0), (w_50, h), (w_25, h), (w_75, h),
                 (0, h_50), (w, h_50)]
        dstpt = [(0, h1), (w, h1), (w_50, 0), (w2, h2), (w - w2, h2), (w3, h3), (w - w3, h3), (w_50, t), (w4, h4),
                 (w - w4, h4), (w5, h5), (w - w5, h5)]

        N = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(N)]
        dst_shape = np.array(dstpt).reshape((-1, N, 2))
        src_shape = np.array(srcpt).reshape((-1, N, 2))
        # #
        # ten_img = ToTensor()(img).to(DEVICE)
        # h, w = ten_img.shape[1], ten_img.shape[2]
        # ten_source = norm(dst_shape, w, h,DEVICE)
        # ten_target = norm(src_shape, w, h,DEVICE)
        #
        # # tps = TPS()
        # warped_grid = self.tps(ten_target[None, ...], ten_source[None, ...], w, h, DEVICE)  # 这个输入的位置需要归一化，所以用norm
        # ten_wrp = torch.grid_sampler_2d(ten_img[None, ...], warped_grid, 0, 0,align_corners=True)
        # img = np.array(ToPILImage()(ten_wrp[0].cpu()))

        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)

        if isflip:
            # img = TF.vflip(img)
            img = cv2.flip(img, 0)
            # img = ImageOps.flip(img)
            rect = (0, self.side // 2, self.side, self.side)
        else:
            rect = (0, 0, self.side, self.side // 2)
        img = Image.fromarray(img)
        img = img.crop(rect)
        # print("6666")
        img = img.resize((W, H), Image.BICUBIC)
        img = np.array(img).astype(np.uint8)
        # print("6666")
        result['img'] = img[..., ::-1]

        # print(img)
        # print(img.size)
        result['img_shape'] = img.shape
        return result

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
