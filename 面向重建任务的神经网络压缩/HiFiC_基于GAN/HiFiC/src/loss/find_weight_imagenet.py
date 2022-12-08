#grad_cam方法得到权重文件json情况下，读取这10个json文件，进行归一化和预处理，并求出平均值，得到可以直接使用的权重。
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
import json
import torch as T
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# k = 4000#给0权重乘的系数，为了softmax结果好一些
# k = 3100 → bpp = 0.3
# k = 3500 → bpp = 0.45
# k = 500 → bpp = 0.14
# r = 30

# r = 75#给loss最终结果乘的系数，为了loss稍微大一些好计算
# r = 15
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
label_name = {"electric_ray": 0, "goldfish": 1, "great_white_shark": 2, "hammerhead": 3, "stingray": 4, "tench": 5,
              "tiger_shark": 6}
#先归一化等操作，最后平均
#grad_cam方法得到权重文件json情况下，读取这10个json文件，进行归一化和预处理，并求出平均值，得到可以直接使用的权重。
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def find_weights_electric_ray(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/electric_ray.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/electric_ray.json'#164服务器
    filename = '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/models/electric_ray.json'#5服务器
    # filename = '/Users/serendipity/Desktop/GAN/HiFiC/electric_ray.json'
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)

    weights_0_tensor = weights_0_tensor * k
    weights_norm = T.nn.functional.softmax(weights_0_tensor)
    # sum = T.sum(weights_norm)
    # print(sum)
    weights_final = T.sqrt(weights_norm)
    # print(weights_final)
    weights_final = weights_final * r
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print(filename)
    return weights

def find_weights_goldfish(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/goldfish.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/goldfish.json'#164
    filename = '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/models/goldfish.json'  # 5服务器
    # filename = '/Users/serendipity/Desktop/GAN/HiFiC/goldfish.json'
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)

    weights_0_tensor = weights_0_tensor * k
    weights_norm = T.nn.functional.softmax(weights_0_tensor)
    # sum = T.sum(weights_norm)
    # print(sum)
    weights_final = T.sqrt(weights_norm)
    weights_final = weights_final * r
    # print(weights_final)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print(filename)
    return weights

def find_weights_great_white_shark(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/great_white_shark.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/great_white_shark.json'#164
    filename = '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/models/great_white_shark.json'  # 5服务器
    # filename = '/Users/serendipity/Desktop/GAN/HiFiC/great_white_shark.json'
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)

    weights_0_tensor = weights_0_tensor * k
    weights_norm = T.nn.functional.softmax(weights_0_tensor)
    # sum = T.sum(weights_norm)
    # print(sum)
    weights_final = T.sqrt(weights_norm)
    weights_final = weights_final * r
    # print(weights_final)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print(filename)
    return weights

def find_weights_hammerhead(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/hammerhead.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/hammerhead.json'#164
    filename = '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/models/hammerhead.json'  # 5服务器
    # filename = '/Users/serendipity/Desktop/GAN/HiFiC/hammerhead.json'
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)

    weights_0_tensor = weights_0_tensor * k
    weights_norm = T.nn.functional.softmax(weights_0_tensor)
    # sum = T.sum(weights_norm)
    # print(sum)
    weights_final = T.sqrt(weights_norm)
    weights_final = weights_final * r
    # print(weights_final)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print(filename)
    return weights

def find_weights_stingray(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/stingray.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/stingray.json'#164
    filename = '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/models/stingray.json'  # 5服务器
    # filename = '/Users/serendipity/Desktop/GAN/HiFiC/stingray.json'
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)

    weights_0_tensor = weights_0_tensor * k
    weights_norm = T.nn.functional.softmax(weights_0_tensor)
    # sum = T.sum(weights_norm)
    # print(sum)
    weights_final = T.sqrt(weights_norm)
    weights_final = weights_final * r
    # print(weights_final)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print(filename)
    return weights

def find_weights_tench(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/tench.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/tench.json'#164
    filename = '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/models/tench.json'  # 5服务器
    # filename = '/Users/serendipity/Desktop/GAN/HiFiC/tench.json'
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)

    weights_0_tensor = weights_0_tensor * k
    weights_norm = T.nn.functional.softmax(weights_0_tensor)
    # sum = T.sum(weights_norm)
    # print(sum)
    weights_final = T.sqrt(weights_norm)
    weights_final = weights_final * r
    # print(weights_final)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print(filename)
    return weights

def find_weights_tiger_shark(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/tiger_shark.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/tiger_shark.json'#164
    filename = '/mnt/DataDrive5/tangrui/PostGraduate/Grade_1/GAN/HiFiC/high-fidelity-generative-compression/data/models/tiger_shark.json'  # 5服务器
    # filename = '/Users/serendipity/Desktop/GAN/HiFiC/tiger_shark.json'
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)

    weights_0_tensor = weights_0_tensor * k
    weights_norm = T.nn.functional.softmax(weights_0_tensor)
    # sum = T.sum(weights_norm)
    # print(sum)
    weights_final = T.sqrt(weights_norm)
    weights_final = weights_final * r
    # print(weights_final)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print(filename)
    return weights

def find_weights_avg_imagenet(k, r):

    weights_cat = find_weights_electric_ray(k, r)
    weights_airplane = find_weights_goldfish(k, r)
    weights_bird = find_weights_great_white_shark(k, r)
    weights_car = find_weights_hammerhead(k, r)
    weights_deer = find_weights_stingray(k, r)
    weights_dog = find_weights_tench(k, r)
    weights_horse = find_weights_tiger_shark(k, r)

    weights_sum = (weights_cat + weights_airplane + weights_bird + weights_car + weights_deer + weights_dog
                + weights_horse )
    weights_all_avg = (1/7) * weights_sum

    # a = T.eq(weights_airplane,weights_truck).sum()
    #
    # print(a)
    # print(r)
    return weights_all_avg
# weights = find_weights_all_avg()

def find_weights_electric_ray_minmax():

    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/tiger_shark.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/tiger_shark.json'#164
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/tiger_shark.json'  # 5服务器
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)
    # print("weights_0_tensor:")
    # print(weights_0_tensor)
    weight_max = T.max(weights_0_tensor)
    # print("weight_max")
    # print(weight_max)
    weight_min = T.min(weights_0_tensor)
    # print("weight_min")
    # print(weight_min)

    weights_norm = (weights_0_tensor - weight_min) / (weight_max - weight_min)
    weights_final = T.sqrt(weights_norm)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print("minmax")
    # print(weights)
    return weights
def find_weights_goldfish_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/goldfish.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/goldfish.json'#164
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/goldfish.json'  # 5服务器
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)
    # print("weights_0_tensor:")
    # print(weights_0_tensor)
    weight_max = T.max(weights_0_tensor)
    # print("weight_max")
    # print(weight_max)
    weight_min = T.min(weights_0_tensor)
    # print("weight_min")
    # print(weight_min)

    weights_norm = (weights_0_tensor - weight_min) / (weight_max - weight_min)
    weights_final = T.sqrt(weights_norm)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print("minmax")
    # print(weights)
    return weights
def find_weights_great_white_shark_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/great_white_shark.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/great_white_shark.json'#164
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/great_white_shark.json'  # 5服务器
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)
    # print("weights_0_tensor:")
    # print(weights_0_tensor)
    weight_max = T.max(weights_0_tensor)
    # print("weight_max")
    # print(weight_max)
    weight_min = T.min(weights_0_tensor)
    # print("weight_min")
    # print(weight_min)

    weights_norm = (weights_0_tensor - weight_min) / (weight_max - weight_min)
    weights_final = T.sqrt(weights_norm)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print("minmax")
    # print(weights)
    return weights
def find_weights_hammerhead_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/hammerhead.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/hammerhead.json'#164
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/hammerhead.json'  # 5服务器
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)
    # print("weights_0_tensor:")
    # print(weights_0_tensor)
    weight_max = T.max(weights_0_tensor)
    # print("weight_max")
    # print(weight_max)
    weight_min = T.min(weights_0_tensor)
    # print("weight_min")
    # print(weight_min)

    weights_norm = (weights_0_tensor - weight_min) / (weight_max - weight_min)
    weights_final = T.sqrt(weights_norm)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print("minmax")
    # print(weights)
    return weights
def find_weights_stingray_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/stingray.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/stingray.json'#164
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/stingray.json'  # 5服务器
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)
    # print("weights_0_tensor:")
    # print(weights_0_tensor)
    weight_max = T.max(weights_0_tensor)
    # print("weight_max")
    # print(weight_max)
    weight_min = T.min(weights_0_tensor)
    # print("weight_min")
    # print(weight_min)

    weights_norm = (weights_0_tensor - weight_min) / (weight_max - weight_min)
    weights_final = T.sqrt(weights_norm)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print("minmax")
    # print(weights)
    return weights
def find_weights_tench_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/tench.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/tench.json'#164
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/tench.json'  # 5服务器
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)
    # print("weights_0_tensor:")
    # print(weights_0_tensor)
    weight_max = T.max(weights_0_tensor)
    # print("weight_max")
    # print(weight_max)
    weight_min = T.min(weights_0_tensor)
    # print("weight_min")
    # print(weight_min)

    weights_norm = (weights_0_tensor - weight_min) / (weight_max - weight_min)
    weights_final = T.sqrt(weights_norm)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print("minmax")
    # print(weights)
    return weights
def find_weights_tiger_shark_minmax():

    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/tiger_shark.json'#0
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/tiger_shark.json'#164
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/tiger_shark.json'  # 5服务器
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    weights_0_tensor = T.Tensor(weights_0)
    # print("weights_0_tensor:")
    # print(weights_0_tensor)
    weight_max = T.max(weights_0_tensor)
    # print("weight_max")
    # print(weight_max)
    weight_min = T.min(weights_0_tensor)
    # print("weight_min")
    # print(weight_min)

    weights_norm = (weights_0_tensor - weight_min) / (weight_max - weight_min)
    weights_final = T.sqrt(weights_norm)
    #转换成tensor,且维度为1*512*1*1
    weights = T.Tensor(weights_final).cuda()#z
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(0)
    # print("minmax")
    # print(weights)
    return weights

def find_weights_avg_imagenet_minmax():
    
    weights_cat = find_weights_electric_ray_minmax()
    weights_airplane = find_weights_goldfish_minmax()
    weights_bird = find_weights_great_white_shark_minmax()
    weights_car = find_weights_hammerhead_minmax()
    weights_deer = find_weights_stingray_minmax()
    weights_dog = find_weights_tench_minmax()
    weights_horse = find_weights_tiger_shark_minmax()

    weights_sum = (weights_cat + weights_airplane + weights_bird + weights_car + weights_deer + weights_dog
                + weights_horse )
    weights_all_avg = (1/7) * weights_sum

    # a = T.eq(weights_airplane,weights_truck).sum()
    #
    # print(a)
    # print(r)
    
    return weights_all_avg

if __name__ == '__main__':
    # weight_minmax = find_weights_avg_stl_minmax()
    # print("weight_final_minmax")
    # print(weight_minmax)
    # weight_minmax_max = T.max(weight_minmax)
    # weight_minmax_min = T.min(weight_minmax)
    # print("weight_minmax_max")
    # print(weight_minmax_max)
    # print("weight_minmax_min")
    # print(weight_minmax_min)

    # weight_max = T.max(weight)
    # weight_min = T.min(weight)
    # print("weight_max")
    # print(weight_max)
    # print("weight_min")
    # print(weight_min)

    # weight = find_weights_avg_imagenet()
    # print("weight_final")
    # print(weight)
    # k = 10000
    # r = 3000
    # weights = find_weights_avg_imagenet(k, r)
    # weights_minmax = find_weights_avg_imagenet_minmax()
    # x_values = list(range(1,513))
    # y_values = list(weights.squeeze().cpu().numpy())
    # y_minmax_values = list(weights_minmax.squeeze().cpu().numpy())
    # plt.scatter(x_values, y_values, c="r", alpha=0.5, marker='o', label=f"softmax_k{k}_r{r}")
    # # plt.scatter(x_values, y_minmax_values, c="b",alpha=0.5, marker='x', label="minmax")
    # # plt.scatter(x_values, y_values, c="r", alpha=0.5, marker='o', label=f"softmax_k{k}_r{r}")
    # plt.legend(loc="best")
    # plt.title("weights")
    # plt.xlabel("feature_map_num")
    # plt.ylabel("weights")
    # os.chdir("/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/weight")
    # plt.savefig(f'k{k}_r{r}')
    # print(1)
    k1=2500
    r1=30
    k2=3255
    r2=30
    weights_1 = find_weights_avg_imagenet(k1, r1)
    weights_minmax = find_weights_avg_imagenet_minmax()
    weights_2 = find_weights_avg_imagenet(k2, r2)
    x_values = list(range(1,513))
    y_values1 = list(weights_1.squeeze().cpu().numpy())
    y_values2 = list(weights_2.squeeze().cpu().numpy())
    y_minmax_values = list(weights_minmax.squeeze().cpu().numpy())
    plt.ylim((0, 5))
    plt.scatter(x_values, y_values1, c="r", alpha=0.5, marker='o')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.scatter(x_values, y_minmax_values, c="b",alpha=0.5, marker='x', label="minmax")
    # plt.scatter(x_values, y_values2, c="r", alpha=0.5, marker='x', label=f"softmax_k{k2}_r{r2}")
    plt.legend(loc="best", fontsize=15)
    # plt.title("weights", fontsize=15)
    plt.xlabel("feature_map_num", fontsize=15)
    plt.ylabel("semantic weights", fontsize=15)
    os.chdir("/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/weight_imagenet")
    # plt.savefig(f'k{k1}_r{r1}_k{k2}_r{r2}')
    # plt.savefig(f'k3500')
    plt.savefig(f'k{k1}_r{r1}')
    print(1)