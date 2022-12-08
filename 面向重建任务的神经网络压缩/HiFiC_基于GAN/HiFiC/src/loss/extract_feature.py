import torch
# import torchvision
# import os
# import cv2
# from torchvision import models,transforms,datasets
from torchvision import models
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm
# from PIL import Image
import torch.nn as nn

#单张图片完成分类任务
'''
可以用来使用backbone提取特征
'''


#norm_mean = [0.485, 0.456, 0.406]
# #norm_std = [0.229, 0.224, 0.225]
# transform1 = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(96),
# ])
#
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
# #    transforms.Normalize(norm_mean, norm_std),
# ])

def extract_feature_stl(tensor_128, pth):
    '''
    直接把128维度输入分类任务
    输入batchsize*3*128*128的tensor
    输出分类任务最后一层的特征表示，是tensor,batchsize*512*4*4
    '''
    # 数据预处理有问题，96还是128。 目前使用128实现
    # print("extract_feature_stl")
    classes = 10
    # a = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # label = {0: "airplane", 1: "bird", 2: "car", 3: "cat", 4: "deer", 5: "dog", 6: "horse", 7: "monkey", 8: "ship",
    #          9: "truck"}
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)
    # model.load_state_dict(torch.load(
    #     '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/STL10_7_96.5.pth'))
    #服务器164
    # model.load_state_dict(torch.load(
    #     '/mnt/DataDrive164/sunqizheng/Lossy_Image_Compression_AE/dataset/stl_original/cls_liu.pth'))
    # 服务器5
    # model.load_state_dict(torch.load(
    #     '/mnt/DataDrive5/sunqizheng/Lossy_Image_Compression_AE/dataset/stl_original/cls_liu.pth'))
    model.load_state_dict(torch.load(pth))    # # print(model)
    model.cuda()
    batch_t = tensor_128
    model.eval()
    out = model(batch_t)
    before = nn.Sequential(*list(model.children())[:-2])
    # list(model.children())是一个包含网络结构的列表，*表示将多个参数放入元组
    feature = before(batch_t)
    feature_51244 = feature  # 512*4*4

    feature_51244 = torch.squeeze(feature_51244)
    # print(feature_51244.shape)
    after = nn.Sequential(list(model.children())[-2])
    feature = after(feature)
    feature = feature.view(batch_t.size(0), -1)  # pytorch的view相当于numpy的reshape
    my_out = model.fc(feature)
    assert (out.shape == my_out.shape)
    # print(feature_51244.shape)
    return feature_51244, my_out
def extract_feature_imagenet_fish(tensor_256, pth):
    '''
    直接把128维度输入分类任务
    输入batchsize*3*128*128的tensor
    输出分类任务最后一层的特征表示，是tensor,batchsize*512*4*4
    数据预处理有问题，224 还是256.目前使用256实现
    '''
    # 
    # print("extract_feature_imagenet_fish")
    classes = 7
    # a = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    # label = {"electric_ray": 0, "goldfish": 1, "great_white_shark": 2, "hammerhead": 3, "stingray": 4,
    #          "tench": 5, "tiger_shark": 6}
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)
    #服务器164
    # model.load_state_dict(torch.load(
    #     '/mnt/DataDrive164/sunqizheng/Lossy_Image_Compression_AE/dataset/imagenet_original/fish_7.pth'))
    #服务器5
    # model.load_state_dict(torch.load(
    #     '/mnt/DataDrive5/sunqizheng/Lossy_Image_Compression_AE/dataset/imagenet_original/fish_7.pth'))
    model.load_state_dict(torch.load(pth))
    # model.load_state_dict(torch.load(pth, map_location='cpu'))
    # print(model)
    # print("extract_feature_fish")
    model.cuda()
    batch_t = tensor_256
    model.eval()
    out = model(batch_t)
    before = nn.Sequential(*list(model.children())[:-2])
    # list(model.children())是一个包含网络结构的列表，*表示将多个参数放入元组
    feature = before(batch_t)
    feature_51244 = feature  # 512*4*4

    feature_51244 = torch.squeeze(feature_51244)
    # print(feature_51244.shape)
    after = nn.Sequential(list(model.children())[-2])
    feature = after(feature)
    feature = feature.view(batch_t.size(0), -1)  # pytorch的view相当于numpy的reshape
    my_out = model.fc(feature)
    assert (out.shape == my_out.shape)
    # print(feature_51244.shape)
    # parms_clss = model.state_dict()
    # print(model)
    # return feature_51244, my_out, model
    return feature_51244, my_out
# def extract_feature_pascal(tensor_256, pth):

# img0 = Image.open(r'/mnt/DataDrive164/sunqizheng/stl_train_no_class/0.png').convert("RGB")
# img1 = Image.open(r'/mnt/DataDrive164/sunqizheng/stl_train_no_class/1.png').convert("RGB")
# imgt0 = transform(img0).unsqueeze(0)
# imgt1 = transform(img1).unsqueeze(0)
# imgg = torch.cat((imgt0, imgt1),0)
# y = model (imgg)
# print(y)
# yy = model(imgt0)
# print(yy)
# print(0)
# def extract_feature_path(img_path):
#     '''
#     根据图片路径。输入多个路径，提取出这些图片的特征
#     输入tuple的图片路径，输出这些路径下图片的特征，是tensor,batchsize*512*7*7
#     '''
#     #img = Image.open(r'/mnt/DataDrive164/sunqizheng/stl_train_no_class/1.png').convert("RGB")
#     batch_t = torch.zeros(1,3,224,224)
#     for k in range (len(img_path)):
#         img = Image.open(img_path[k]).convert("RGB")
#         img_t = transform(img).unsqueeze(0)
#         batch_t = torch.cat((batch_t, img_t),0)
#     batch_t = batch_t[torch.arange(batch_t.size(0)) != 0]
#     model.eval()
#     out = model(batch_t)
#     before = nn.Sequential(*list(model.children())[:-2])
#     #list(model.children())是一个包含网络结构的列表，*表示将多个参数放入元组
#     feature = before(batch_t)
#     feature_51277 = feature#512*7*7
#     feature_51277 = torch.squeeze(feature_51277)
#     after = nn.Sequential(list(model.children())[-2])
#     feature = after(feature)
#     feature = feature.view(batch_t.size(0), -1)#pytorch的view相当于numpy的reshape
#     my_out = model.fc(feature)
#     assert (out.shape == my_out.shape)
#     return feature_51277, my_out
#
# def extract_feature_tran(tensor_128):
#     '''
#     经过transform
#     输入batchsize*3*128*128的tensor，
#     再恢复成原图batchsize*3*96*96，
#     然后按照分类任务预处理成batchsize*3*224*24
#     然后完成分类任务
#     然后输出分类任务最后一层的特征表示，是tensor,batchsize*512*7*7
#     '''
#     #img = Image.open(r'/mnt/DataDrive164/sunqizheng/stl_train_no_class/1.png').convert("RGB")
#     #batch_t = (batchsize,3,224,224)
#     PIL_96 = []
#     for k in range(tensor_128.shape[0]):
#         PIL_96_temp = transform1(tensor_128.cpu()[k])
#         PIL_96.append(PIL_96_temp)
#     tensor_224 = torch.zeros(1,3,224,224).cuda()
#     for k in range(tensor_128.shape[0]):
#         tensor_224_temp = transform(PIL_96[k]).cuda()
#         tensor_224_temp = tensor_224_temp.unsqueeze(0)
#         tensor_224 = torch.cat((tensor_224,tensor_224_temp),0)
#     batch_t = tensor_224
#     batch_t = batch_t[torch.arange(batch_t.size(0)) != 0]
#     assert(len(PIL_96) == batch_t.shape[0])
#     #print(len(PIL_96))
#     #print(batch_t.shape[0])
#     model.eval()
#     out = model(batch_t)
#     before = nn.Sequential(*list(model.children())[:-2])
#     #list(model.children())是一个包含网络结构的列表，*表示将多个参数放入元组
#     feature = before(batch_t)
#     feature_51277 = feature#512*7*7
#     feature_51277 = torch.squeeze(feature_51277)
#     after = nn.Sequential(list(model.children())[-2])
#     feature = after(feature)
#     feature = feature.view(batch_t.size(0), -1)#pytorch的view相当于numpy的reshape
#     my_out = model.fc(feature)
#     assert (out.shape == my_out.shape)
#     return feature_51277, my_out


# '''
# array_of_img = [] # this if for store all of the image data
# # this function is for read image,the input is directory name
# def read_directory(directory_name):
#     # this loop is for read each image in this foder,directory_name is the foder name with images.
#     for filename in os.listdir(r"/home/qoc099/wuqianwen/classification2/"+directory_name):
#         #print(filename) #just for test
#         #img is used to store the image data
#         img = cv2.imread(directory_name + "/" + filename)r
#         img_t=transform(img)
#         array_of_img.append(img_t)
#         #print(img)
#         print(array_of_img)

# img_t=read_directory("data")
# '''
# '''
# def imshow(img):
#  img = img / 2 + 0.5  # unnormalize
#  npimg = img.numpy()
#  plt.imshow(np.transpose(npimg, (1, 2, 0)))
#  plt.show()
# imshow(img_t1)
# '''

#img_path = ['/mnt/DataDrive164/sunqizheng/stl_train_no_class/0.png','/mnt/DataDrive164/sunqizheng/stl_train_no_class/1.png']
# img_path = ('/mnt/DataDrive164/sunqizheng/stl_train_no_class/0.png', '/mnt/DataDrive164/sunqizheng/stl_train_no_class/0.png','/mnt/DataDrive164/sunqizheng/stl_train_no_class/1.png', '/mnt/DataDrive164/sunqizheng/stl_train_no_class/2.png')
#
# print(img_path)
#my_feature, my_out = extract_feature_path(img_path)#直接读入tuple形式的path，
# my_feature是tensor,batchsize*512*7*7,my_out是tensor batchsize*10
#print(my_feature)
#print(my_feature[0])
#print(my_feature[1])

def result(my_out):
    percentage = torch.nn.functional.softmax(my_out, dim=1)[0] * 100
    _, indices = torch.sort(my_out, descending=True)
    result_all = [(label[idx], percentage[idx].item()) for idx in indices.cpu().numpy()[0][:5]]
    #print(result_all)
    return result_all

#result_all = result(my_out)
#print(result_all)