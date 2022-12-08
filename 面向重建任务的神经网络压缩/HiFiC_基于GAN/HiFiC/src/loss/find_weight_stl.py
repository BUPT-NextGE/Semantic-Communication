#grad_cam方法得到权重文件json情况下，读取这10个json文件，进行归一化和预处理，并求出平均值，得到可以直接使用的权重。
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
import json
import torch as T
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# k = 1500#给0权重乘的系数，为了softmax结果好一些
# r = 15#给loss最终结果乘的系数，为了loss稍微大一些好计算
def find_weights_cat(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = 'airplane.json'
    # filename = 'airplane_no_channel.json'
    # filename = 'cat_no_channel.json'
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/cat_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/cat_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/cat_no_channel.json'  # 5服务器
    with open(filename) as f_obj:
        weights_0 = json.load(f_obj)  # 512维的list
    # print(weights_0)
    # weights_0 = np.maximum(weights_0, 0)
    # weights_norm = [i * 6 * 512 for i in weights_0]  # 把权值归512化
    weights_0_tensor = T.Tensor(weights_0)
    # weights_norm = T.nn.functional.softmax(weights_0_tensor)  # 把权值softmax归一化，用传统提供的exp计算，后续可以改为用直接求和计算
    # print(weights_norm)
    # weight_norm = T.sum(weights_0_tensor)
    # weights_0_tensor = weights_0_tensor*10000
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # 1500是超参数，后期要调
    weights_0_tensor = weights_0_tensor * k
    weights_norm = T.nn.functional.softmax(weights_0_tensor)
    # sum = T.sum(weights_norm)
    # print(sum)
    # weights_final = [np.sqrt(i) for i in weights_norm]#最终的权值
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
# a = find_weights_airplane()
# filename = 'airplane.json'
# with open(filename) as f_obj:
#     weights_0 = json.load(f_obj)#512维的list
# # print(weights_0)
# weights_norm = [i * 6 *512 for i in weights_0]#把权值归512化
# # print(weights_norm)
# weights_final = [np.sqrt(i) for i in weights_norm]
# print(weights_final)
def find_weights_airplane(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/airplane_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/airplane_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/airplane_no_channel.json'#5服务器
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
def find_weights_bird(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = 'bird_no_channel.json'
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/bird_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/bird_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/bird_no_channel.json'  # 5服务器
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
def find_weights_car(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = 'car_no_channel.json'
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/car_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/car_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/car_no_channel.json'  # 5服务器
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
def find_weights_deer(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = 'deer_no_channel.json'
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/deer_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/deer_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/deer_no_channel.json'  # 5服务器
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
def find_weights_dog(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = 'dog_no_channel.json'
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/dog_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/dog_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/dog_no_channel.json'  # 5服务器
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
def find_weights_horse(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = 'horse_no_channel.json'
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/horse_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/horse_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/horse_no_channel.json'  # 5服务器
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
def find_weights_monkey(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = 'monkey_no_channel.json'
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/monkey_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/monkey_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/monkey_no_channel.json'  # 5服务器
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
def find_weights_ship(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = 'ship_no_channel.json'
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/ship_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/ship_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/ship_no_channel.json'  # 5服务器
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
def find_weights_truck(k, r):
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/truck_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/truck_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/truck_no_channel.json'  # 5服务器
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

def find_weights_avg_stl(k, r):
    weights_cat = find_weights_cat(k, r)
    weights_airplane = find_weights_airplane(k, r)
    weights_bird = find_weights_bird(k, r)
    weights_car = find_weights_car(k, r)
    weights_deer = find_weights_deer(k, r)
    weights_dog = find_weights_dog(k, r)
    weights_horse = find_weights_horse(k, r)
    weights_monkey = find_weights_monkey(k, r)
    weights_ship = find_weights_ship(k, r)
    weights_truck = find_weights_truck(k, r)
    weights_sum = (weights_cat + weights_airplane + weights_bird + weights_car + weights_deer + weights_dog
                + weights_horse + weights_monkey + weights_ship + weights_truck )
    weights_all_avg = (1/10) * weights_sum

    # a = T.eq(weights_airplane,weights_truck).sum()
    #
    # print(a)
    # print(r)
    return weights_all_avg
# weights = find_weights_all_avg()

def find_weights_cat_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/cat_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/cat_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/cat_no_channel.json'  # 5服务器
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
    # print(filename)
    return weights
def find_weights_airplane_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/airplane_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/airplane_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/airplane_no_channel.json'  # 5服务器
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
    # print(filename)
    return weights
def find_weights_car_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/car_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/car_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/car_no_channel.json'  # 5服务器
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
    # print(filename)
    return weights
def find_weights_bird_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/bird_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/bird_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/bird_no_channel.json'  # 5服务器
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
    # print(filename)
    return weights
def find_weights_deer_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/deer_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/deer_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/deer_no_channel.json'  # 5服务器
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
    # print(filename)
    return weights
def find_weights_dog_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/dog_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/dog_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/dog_no_channel.json'  # 5服务器
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
    # print(filename)
    return weights
def find_weights_horse_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/horse_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/horse_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/horse_no_channel.json'  # 5服务器
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
    # print(filename)
    return weights
def find_weights_monkey_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/monkey_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/monkey_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/monkey_no_channel.json'  # 5服务器
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
    # print(filename)
    return weights
def find_weights_ship_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/ship_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/ship_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/ship_no_channel.json'  # 5服务器
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
    # print(filename)
    return weights
def find_weights_truck_minmax():
    #cam的weight是使用原图得到的
    # 修改读取的json文件
    # filename = '/home/ubuntu/users/sunqizheng/code_codec/Lossy_Image_Compression_AE/cae-master/src/truck_no_channel.json'# 164服务器
    # filename = '/home/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/truck_no_channel.json'# 0服务器
    filename = '/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/src/truck_no_channel.json'  # 5服务器
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
    # print(filename)
    return weights

def find_weights_avg_stl_minmax():
    weights_cat = find_weights_cat_minmax()
    weights_airplane = find_weights_airplane_minmax()
    weights_bird = find_weights_bird_minmax()
    weights_car = find_weights_car_minmax()
    weights_deer = find_weights_deer_minmax()
    weights_dog = find_weights_dog_minmax()
    weights_horse = find_weights_horse_minmax()
    weights_monkey = find_weights_monkey_minmax()
    weights_ship = find_weights_ship_minmax()
    weights_truck = find_weights_truck_minmax()
    weights_sum = (weights_cat + weights_airplane + weights_bird + weights_car + weights_deer + weights_dog
                + weights_horse + weights_monkey + weights_ship + weights_truck )
    weights_all_avg = (1/10) * weights_sum

    # a = T.eq(weights_airplane,weights_truck).sum()
    #
    # print(a)
    # print(r)
    return weights_all_avg
# weights = find_weights_all_avg()

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
    # weight = find_weights_avg_stl()
    # print("weight_final")
    # print(weight)
    # weight_max = T.max(weight)
    # weight_min = T.min(weight)
    # print("weight_max")
    # print(weight_max)
    # print("weight_min")
    # print(weight_min)
    k1=5500
    r1=30
    k2=3255
    r2=30
    weights_1 = find_weights_avg_stl(k1, r1)
    weights_minmax = find_weights_avg_stl_minmax()
    weights_2 = find_weights_avg_stl(k2, r2)
    x_values = list(range(1,513))
    y_values1 = list(weights_1.squeeze().cpu().numpy())
    y_values2 = list(weights_2.squeeze().cpu().numpy())
    y_minmax_values = list(weights_minmax.squeeze().cpu().numpy())
    plt.scatter(x_values, y_values1, c="r", alpha=0.5, marker='o', label=f"softmax_k{k1}_r{r1}")
    plt.scatter(x_values, y_minmax_values, c="b",alpha=0.5, marker='x', label="minmax")
    # plt.scatter(x_values, y_values2, c="r", alpha=0.5, marker='x', label=f"softmax_k{k2}_r{r2}")
    plt.legend(loc="best")
    plt.title("weights_stl")
    plt.xlabel("feature_map_num")
    plt.ylabel("weights")
    os.chdir("/home/lab239-5/users/sunqizheng/Lossy_Image_Compression_AE/cae-master/weight_stl")
    # plt.savefig(f'k{k1}_r{r1}_k{k2}_r{r2}')
    # plt.savefig(f'k3500')
    # plt.savefig(f'k{k1}_r{r1}')
    plt.savefig(f'k{k1}_r{r1}'+'+minmax')
    # plt.savefig(f'minmax')
    print(1)