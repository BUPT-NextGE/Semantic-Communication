"""
ssim_psnr.py
@author Echo
@date 2022-05-05 15:22 
@description 计算数据集的SSIM和PSNR

"""

import glob
import os

from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2


def cal_psnr_ssim(test_dir):
    original_images = glob.glob(os.path.join('/Users/serendipity/Downloads/ImageNet/test_original', '**/*.JPEG'),
                                recursive=True)
    p = 0
    s = 0
    total = len(original_images)
    for img_dir in original_images:
        # 测扩展名是JPEG
        # test_img_name = img_dir.split('/')[-2] + '/' + img_dir.split('/')[-1]
        # 测扩展名是webp
        test_img_name = img_dir.split('/')[-2] + '/' + img_dir.split('/')[-1].split('.')[0] + '.webp'

        print(img_dir)
        print(test_dir + test_img_name)
        img1 = cv2.imread(img_dir)  # 正确图像地址
        img2 = cv2.imread(test_dir + test_img_name)  # 待比较图像地址

        psnr = compare_psnr(img1, img2)
        ssim = compare_ssim(img1, img2, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
        print(psnr)
        print(ssim)
        p = p + psnr
        s = s + ssim
        total = total - 1
        print(total)
        print("===============================")

    print(p)
    print(s)
    p = p / len(original_images)
    s = s / len(original_images)
    print('PSNR：{} dB，SSIM：{}'.format(p, s))


if __name__ == '__main__':
    cal_psnr_ssim('/Users/serendipity/Downloads/ImageNet/WebP/WebP_test_bpp_0.5' + '/')
