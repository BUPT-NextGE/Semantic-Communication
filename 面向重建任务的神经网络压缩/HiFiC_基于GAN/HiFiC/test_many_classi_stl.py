import torch
from torch.utils.data import DataLoader
import os
import cv2
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from src.helpers.my_dataset import Dataset_stl_test
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#多张图片测试，计算所有图片分类任务的准确度
#输入十个文件夹的图片，输出这些图片完成分类任务的平均准确度
# 输入图片的路径，要求这个路径下要有文件夹，文件夹的名字是类别，文件夹有多少个不作要求，可以测试单类也可以测试多类，但是需要手动调文件夹
#记录分类错误的图片路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))
def test_stl(test_dir):
    classes = 10
    BATCH_SIZE = 100#batchsize必须被总数据集数整除，才能看错误的是哪些图片


    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)
    #加载分类resnet的pth
    # model.load_state_dict(torch.load('/home/ubuntu/users/sunqizheng/faster_rcnn/Lossy_Image_Compression_AE/cae-master/src/STL10_7_96.5.pth'))
    # model.load_state_dict(torch.load('/mnt/DataDrive164/sunqizheng/cls_liu.pth'))#164
    #model.load_state_dict(torch.load('/home/ubuntu/users/sunqizheng/faster_rcnn/classification_STL/STL10_1.pth'))
    model.load_state_dict(torch.load('/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/models/cls_liu.pth'))#5
    #print(model)

    #norm_mean = [0.4416781, 0.43651673, 0.40246016]
    #norm_std = [0.25671667, 0.25384226, 0.2686348]

    #BASEDIR = os.path.dirname(os.path.abspath(__file__))
    #test_dir = "F:\代码\classification_STL\data\STL10\cae_num_1000"



    print(test_dir)
    test_transform = transforms.Compose([
        # transforms.Resize(96),
        transforms.Resize(128),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    #    transforms.Normalize(norm_mean, norm_std),
    ])

    #test_data,_ = AntsDataset(data_dir=test_dir, transform=test_transform)
    # test_data = Dataset_stl_test(data_dir=test_dir, transform=test_transform)
    test_data = Dataset_stl_test(data_dir=test_dir)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)


    criterion = nn.CrossEntropyLoss()   #交叉熵
    model.to(device)
    model.eval()
    correct_test = 0.
    total_test = 0.
    loss_test = 0.
    loss_test_img = list()
    filename = 'wrong_stl.txt'
    with open(filename, 'w') as file_object:
        file_object.write(" " + '\n')
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            # print(j)
            inputs, labels, path_input = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_test_img.append(loss.squeeze().cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)

            mask = torch.ne(predicted, labels)#比较两个tensor是否不等，返回一个tensor，识别正确（相等）为0，识别错误（不等）为1
            # len = mask.shape[0]
            # index = torch.arange(len)
            index = torch.arange(len(labels))
            # print(mask)
            # print(index[mask])
            index_wrong = index[mask]
            for i in range(len(index_wrong)):
                path_wrong = str(path_input[index_wrong[i]])
                # print(path_wrong)
                with open(filename, 'a') as file_object:
                    file_object.write(path_wrong + '\n')
            correct_test += (predicted == labels).squeeze().cpu().sum().numpy()
            loss_test += loss.item()

        loss_test_mean = loss_test / len(test_loader)

    '''
    def deal():
    
        # list转dataframe
        df = pd.DataFrame(loss_test_img, columns=['loss'])
    
        # 保存到本地excel
        df.to_excel("loss_img.xlsx", index=False)
    '''

    acc = correct_test / total_test

    print("Accuracy: {:.2%}".format(correct_test / total_test))

    # print(loss_test_img)
    return acc, loss_test_mean

if __name__ == '__main__':

    # base_dir = '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/STL/JPEG/add_noise'
    #
    # for mode in ['JPEG', 'WebP']:
    #     for bpp in [0.125, 0.25, 0.5]:
    #         for snr in [1, 5, 10, 11, 12, 13, 14, 15, 20]:
    #             print('mode: {}, bpp: {}, snr: {}'.format(mode, bpp, snr))
    #             if mode == 'JPEG':
    #                 test_dir = base_dir + '/stl_' + mode.lower() + '_' + str(bpp) + '_snr_' + str(snr) + 'dB'
    #             elif mode == 'WebP':
    #                 test_dir = base_dir.split('J')[0] + mode + '/add_noise' + '/stl_' + mode.lower() + '_' + str(
    #                     bpp) + '_snr_' + str(snr) + 'dB'
    #             acc = test_stl(test_dir)
    #             # print(acc)
    #             print("====================================================================")

    for snr in [20]:
        acc = test_stl('/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/openimages/test_with_different_compression_mode/different_mode_ldpc/STL-10/BPG/stl_bpg_0.125_ldpc2img/stl_bpg_0.125_ldpc2img_' + str(snr) + 'dB')
