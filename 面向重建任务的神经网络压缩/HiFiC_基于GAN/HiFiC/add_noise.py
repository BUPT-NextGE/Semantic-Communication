import glob
import os

import numpy as np
from torchvision.transforms import ToPILImage
import torch
from PIL import Image


def img_to_bit(input_path, output_path):
    """
    将输入路径下的所有图片转为二进制比特流，并保存为txt
    :param input_path: 输入图片路径
    :param output_path: 输出txt路径
    :return: None
    """
    file = open(input_path, 'rb')  # 输入bpg压缩后的文件
    file_context = file.read()  # <class 'bytes'>字节流

    tmp_a = []
    bit_all = ''
    for i in file_context:
        tmp_a.append(i)  # int类型的数据
    tmp_b = np.array(tmp_a, dtype=np.uint8)
    for j in tmp_b:
        k = bin(j).replace('0b', '').rjust(8, '0')
        bit_all = bit_all + k
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(bit_all)
        f.close()


def cut_string(obj, sec):
    """
    切割字符串
    :param obj: 输入字符串
    :param sec: 切割的位数
    :return: 切割后的字符串
    """
    return [obj[i:i + sec] for i in range(0, len(obj), sec)]


def add_noise(input_path, target_snr=10):
    """
    给图片添加指定信噪比的噪声
    :param input_path: 图片对应比特流存储的txt路径
    :param target_snr: 目标信噪比
    :return: 返回添加噪声后的比特流字符串
    """
    with open(input_path, 'r') as f:
        f_context = f.read().strip()  # 读取字符串
        k_char = cut_string(f_context, 1)  # 字符串按8切割
        # int(a, 2)表示将二进制的字符串a表示为十进制的int
        k = [int(a, 2) for a in k_char]  # 字符串转换为int类型的数据
        bit_array = np.array(k)
        signal = bit_array

        SNR = target_snr

        noise = np.random.randn(signal.shape[0])  # 产生N(0,1)噪声数据
        noise = noise - np.mean(noise)  # 均值为0
        signal_power = np.linalg.norm(signal - signal.mean()) ** 2 / signal.size  # 信号方差

        noise_variance = signal_power / np.power(10, (SNR / 10))  # 噪声方差
        noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
        signal_with_noise = noise + signal

        Ps = (np.linalg.norm(signal - signal.mean())) ** 2  # 信号功率
        Pn = (np.linalg.norm(signal - signal_with_noise)) ** 2  # 噪声功率
        snr = 10 * np.log10(Ps / Pn)
        print(snr)

        bitstring_with_noise = ''
        num = 0
        for i in signal_with_noise:
            if i >= 0.5:
                i = 1
            else:
                i = 0
            bitstring_with_noise += str(i)
            num = num + 1
        # print(num)
        # print(len(bitstring_with_noise))
        return bitstring_with_noise


def random_noise(nc, width, height):
    """Generator a random noise image from tensor.

    If nc is 1, the Grayscale image will be created.
    If nc is 3, the RGB image will be generated.

    Args:
        nc (int): (1 or 3) number of channels.
        width (int): width of output image.
        height (int): height of output image.
    Returns:
        PIL Image.
    """
    img = torch.rand(nc, width, height)
    img = ToPILImage()(img)
    return img


def bit_to_img(string, img_dir, output_path):
    """
    将比特流字符串重新转换为图片，若转换后的图片无法打开则将之变为一副对应尺寸的全白的图
    :param img_dir: 输入图片路径
    :param string: 已经添加噪声后的图片比特流字符串
    :param output_path: 输出图片路径
    :return: None
    """
    split_char = cut_string(string, 8)  # 字符串按8切割
    # int(a, 2)表示将二进制的字符串a表示为十进制的int
    int_8 = [int(a, 2) for a in split_char]  # 字符串转换为int类型的数据
    out_stream = np.array(int_8, dtype=np.uint8)
    # print(out_stream)
    # print(out_stream.size)
    directory = output_path + '/' + os.path.basename(os.path.dirname(img_dir))
    if not os.path.exists(directory):
        os.makedirs(directory)
    output_path = directory + '/' + os.path.basename(img_dir)
    out_stream.tofile(output_path)

    try:
        Image.open(output_path).convert('RGB')
    except IOError:
        print('Error')
        width = Image.open(img_dir).width
        height = Image.open(img_dir).height
        random_noise(3, width, height).save(output_path)


if __name__ == '__main__':

    # ImageNet
    # for bpp in ['0.125', '0.25', '0.5']:
    #     input_images = glob.glob(
    #         os.path.join('/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/ImageNet/JPEG/without_noise/JPEG_test_bpp_' + bpp, '**/*.JPEG'),
    #         recursive=True)
    #     for snr in [1, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    #         total = 2780
    #         for img_dir in input_images:
    #             img_to_bit(img_dir, '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/ImageNet/JPEG/add_noise/tmp.txt')
    #             img_bitstring_with_noise = add_noise('/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/ImageNet/JPEG/add_noise/tmp.txt',
    #                                                  target_snr=snr)
    #             output_path = '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/ImageNet/JPEG/add_noise/bpp_' + bpp + '/noise_with_repair/snr_' + str(
    #                 snr) + '_with_repair'
    #             bit_to_img(img_bitstring_with_noise, img_dir, output_path=output_path)
    #             total = total - 1
    #             print(total)

    # STL-10
    input_images = glob.glob(
        os.path.join(
            '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/STL-10/WebP/testset_stl_bpp_0.25',
            '**/*.webp'),
        recursive=True)
    for snr in [16, 17, 18, 19]:
        total = len(input_images)
        for img_dir in input_images:
            img_to_bit(img_dir,
                       '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/ImageNet/JPEG/add_noise/tmp.txt')
            img_bitstring_with_noise = add_noise(
                '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/ImageNet/JPEG/add_noise/tmp.txt',
                target_snr=snr)
            output_path = '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/STL-10/WebP/testset_stl_bpp_0.25_with_noise/snr_' + str(
                snr) + '_with_repair'
            print(img_dir)
            bit_to_img(img_bitstring_with_noise, img_dir, output_path=output_path)
            total = total - 1
            print(total)
