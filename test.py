import scipy.io as scio
import torch
from option import opt
# from utils import *
import os
import numpy as np
from ssim_torch import ssim

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

def LoadTest_pre(path_test):
    data = scio.loadmat(path_test)
    truth = np.asarray(data["truth"], dtype=np.float32)
    pred = np.asarray(data["pred"], dtype=np.float32)

    if truth.ndim != 4 or truth.shape != pred.shape:
        raise ValueError("preds 和 truths 必须是相同尺寸的 [N,H,W,C] 数组")

    test_data = torch.from_numpy(truth.transpose(0, 3, 1, 2).copy())
    test_pre = torch.from_numpy(pred.transpose(0, 3, 1, 2).copy())
    return test_data, test_pre

FLT_MIN = torch.finfo(torch.float32).eps

def torch_sam(im_test, im_true, eps=1e-12):
    """输入 [C,H,W]；返回平均 SAM，单位：度。"""
    pred = im_test.detach().double()
    truth = im_true.detach().double()

    dot = torch.sum(pred * truth, dim=0)
    norm_pred = torch.sqrt(torch.sum(pred ** 2, dim=0))
    norm_truth = torch.sqrt(torch.sum(truth ** 2, dim=0))

    # 真实零光谱没有定义明确的光谱方向
    valid = norm_truth > eps
    if not valid.any().item():
        raise ValueError("真实图像中没有可计算 SAM 的非零光谱")

    denominator = (
        norm_pred.clamp_min(eps)
        * norm_truth.clamp_min(eps)
    )
    cosine = (dot / denominator).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)

    return angle[valid].mean() * (180.0 / np.pi)

def torch_ergas(im_test, im_true, scale=1.0, eps=1e-12):
    """
    输入 [C,H,W]。
    scale 为公式中的乘法系数；同分辨率评测使用 1。
    """
    pred = im_test.detach().double()
    truth = im_true.detach().double()

    mse_per_band = torch.mean((pred - truth) ** 2, dim=(1, 2))
    mean_truth = torch.mean(truth, dim=(1, 2))

    if torch.any(mean_truth.abs() <= eps).item():
        raise ValueError("真实图像存在均值接近零的波段，ERGAS 分母退化")

    return 100.0 * scale * torch.sqrt(
        torch.mean(mse_per_band / mean_truth.square())
    )

# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def _as_floats(im1, im2):
    im1 = im1.cpu().detach().numpy()
    im2 = im2.cpu().detach().numpy()
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    return np.asarray(im1, dtype=float_type), np.asarray(im2, dtype=float_type)

def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return torch.tensor(np.mean(np.square(im1 - im2), dtype=np.float64))

test_data, test_pre = LoadTest_pre(
    '/home/dwx/dwx/code/6 CASSI 双/simulation/train_code/exp/rgbloss_0.001 diffloss 0.01 stage7 1019/result/Test_82_49.2646_0.9968'
)
test_pre = test_pre.cuda().float()

def test():
    psnr_list, ssim_list, sam_list = [], [], []
    mse_list,  cc_list,   ergas_list = [], [], []
    test_gt = test_data.cuda().float()

    for k in range(test_gt.shape[0]):
        pred_k = test_pre[k]
        gt_k   = test_gt[k]

        psnr_val  = torch_psnr(pred_k, gt_k)
        ssim_val  = torch_ssim(pred_k, gt_k)
        sam_val   = torch_sam(pred_k, gt_k)
        ergas_val = torch_ergas(pred_k, gt_k)

        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        sam_list.append(sam_val.detach().cpu().numpy())
        ergas_list.append(ergas_val.detach().cpu().numpy())

    pred  = np.transpose(test_pre.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(),           (0, 2, 3, 1)).astype(np.float32)
    mask  = test_pre.cpu().numpy().astype(np.float32)

    psnr_mean  = np.mean(np.asarray(psnr_list))
    ssim_mean  = np.mean(np.asarray(ssim_list))
    sam_mean   = np.mean(np.asarray(sam_list))
    ergas_mean = np.mean(np.asarray(ergas_list))

    print('===> testing  psnr = {:.4f},  ssim = {:.4f},  sam = {:.4f}, ergas = {:.4f}'.format(psnr_mean, ssim_mean, sam_mean, ergas_mean))
    print("PSNR per sample:")
    for i, v in enumerate(psnr_list):
        print(f"{v.item():.4f}")
    print("SSIM per sample:")
    for i, v in enumerate(ssim_list):
        print(f"{v.item():.4f}")
    print("SAM per sample:")
    for i, v in enumerate(sam_list):
        print(f"{v.item():.4f}")

    return pred, truth, mask, psnr_list, ssim_list, psnr_mean, ssim_mean


def main():
    (pred, truth, mask, psnr_all, ssim_all, psnr_mean, ssim_mean) = test()

if __name__ == '__main__':
    main()