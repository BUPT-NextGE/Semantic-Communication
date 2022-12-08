import torch
from torch.utils.data import DataLoader
import os
# import cv2
from torchvision import models, transforms, datasets
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm
# from PIL import Image
import torch.nn as nn
from src.helpers.my_dataset import Dataset_imagenet_test
from utils import save_imgs

# import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 多张图片测试，计算所有图片分类任务的准确度
# 输入十个文件夹的图片，输出这些图片完成分类任务的平均准确度
# 输入图片的路径，要求这个路径下要有文件夹，文件夹的名字是类别，文件夹有多少个不作要求，可以测试单类也可以测试多类，但是需要手动调文件夹
# 记录分类错误的图片路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))
# str = "/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/ImageNet/JPEG/add_noise/bpp_0.5/noise_with_repair/snr_13_with_repair"


def test_imagenet(test_dir):
    classes_num = 7
    # classes = ["tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead", "electric_ray", "stingray"]
    classes = ["electric_ray", "goldfish", "great_white_shark", "hammerhead", "stingray", "tench", "tiger_shark"]
    base_dir = test_dir
    # base_dir = "/Users/serendipity/Desktop/GAN/dataset/imagenet_original/reconstructions/original_model/hific_hi"
    # base_dir = "/Users/serendipity/Desktop/GAN/dataset/imagenet_original/testset_fish_imagenet"

    os.makedirs(f"{base_dir}/wrong", exist_ok=True)
    for i in range(len(classes)):
        os.makedirs(f"{base_dir}/wrong/pre_{classes[i]}", exist_ok=True)
    filename = f"{base_dir}/wrong.txt"
    BATCH_SIZE = 100  # batchsize必须被总数据集数整除，才能看错误的是哪些图片
    print(test_dir)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes_num)
    # model.load_state_dict(torch.load('/mnt/DataDrive164/sunqizheng/imagenet_original/fish_7.pth'))#164
    model.load_state_dict(torch.load(
        '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/models/fish_7.pth'))
    # model.load_state_dict(torch.load('/Users/serendipity/Desktop/GAN/dataset/imagenet_original/fish_7.pth', map_location='cpu'))#5
    test_data = Dataset_imagenet_test(data_dir=test_dir)  # 把transform定义在Dataset_imagenet_test里面了
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
    filename = f"{base_dir}/wrong.txt"
    with open(filename, 'w') as file_object:
        file_object.write(" " + '\n')
    criterion = nn.CrossEntropyLoss()  # 交叉熵
    model.to(device)
    correct_test = 0.
    total_test = 0.
    loss_test = 0.
    loss_test_img = list()
    correct_val = 0.
    total_val = 0.
    loss_val = 0.
    with torch.no_grad():
        model.eval()
        for j, data in enumerate(test_loader):
            # print(j)
            inputs, labels, path_input = data
            # inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # inputs(bs,3,256,256)
            outputs = model(inputs).data  # (batch_size,class_num),(10,7)
            loss = criterion(outputs, labels)
            loss_test_img.append(loss.squeeze().cpu().numpy())
            _, predicted = torch.max(outputs, 1)  # (batch_size)(10)
            total_test += labels.size(0)

            mask = torch.ne(predicted, labels)  # 比较两个tensor是否不等，返回一个tensor，识别正确（相等）为0，识别错误（不等）为1
            index = torch.arange(len(labels))  # [0,1,2,...,batchsize]
            # print(mask)
            # print(index[mask])
            index_wrong = index[mask]  # 一个batch中识别错误的索引
            # for i in range(len(index_wrong)):
            #     path_wrong = str(path_input[index_wrong[i]])
            #     # print(path_wrong)
            #     with open(filename, 'a') as file_object:
            #         file_object.write(path_wrong + '\n')
            correct_test += (predicted == labels).squeeze().cpu().sum().numpy()
            for i in range(len(labels)):
                if i in index_wrong:  # 如果识别错误，则记录
                    # path_wrong = str(path_input[i])
                    # print(i)
                    base_name_wrong = os.path.splitext(os.path.basename(path_input[i]))[
                        0]  # path_img_test[i]说明是batch里面的第i个图片
                    ##使用splitext分离文件名和扩展名;[0]是文件名，[1]是扩展名
                    # print(base_name_wrong)
                    # print(out_test[i].shape)
                    label_classi_pre = classes[predicted[i]]
                    label_classi_gt = classes[labels[i]]
                    # print(label_classi_pre)
                    with open(filename, 'a') as file_object:
                        file_object.write(
                            base_name_wrong + ' ground truth:' + label_classi_gt + ' predicted:' + label_classi_pre + '\n')
                    save_imgs(imgs=inputs[i].unsqueeze(0), to_size=(3, 256, 256),
                              name=f"{base_dir}/wrong/pre_{label_classi_pre}/{base_name_wrong}.JPEG")

            loss_test += loss.item()
        loss_test_mean = loss_test / len(test_loader)

    # print("Loss: {:.4f}, Accuracy: {:.2%}".format(loss_test_mean, correct_test / total_test))
    print("Accuracy: {:.2%}".format(correct_test / total_test))

    # print(loss_test_img)
    acc = correct_test / total_test
    return acc


if __name__ == '__main__':

    # base_dir = '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/ImageNet/JPEG/add_noise'
    #
    # for mode in ['JPEG', 'WebP']:
    #     for bpp in [0.125, 0.25, 0.5]:
    #         for snr in [1, 5, 11, 12, 13]:
    #             print('mode: {}, bpp: {}, snr: {}'.format(mode, bpp, snr))
    #             if mode == 'JPEG':
    #                 test_dir = base_dir + '/imagenet_' + mode.lower() + '_' + str(bpp) + '_snr_' + str(snr) + 'dB'
    #             elif mode == 'WebP':
    #                 test_dir = base_dir.split('J')[0] + mode + '/add_noise' + '/imagenet_' + mode.lower() + '_' + str(bpp) + '_snr_' + str(snr) + 'dB'
    #             acc = test_imagenet(test_dir)
    #             # print(acc)
    #             print("====================================================================")
    for bpp in ['0.125', '0.25']:
        acc = test_imagenet('/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/ImageNet/BPG/imagenet_bpg_' + bpp + '_bit2bit2img')
