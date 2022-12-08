import os
# from PIL import Image
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import glob

import numpy as np
import torch

from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
import random

# # 遇到破损的图像时，程序就会跳过去，读取另一张图片
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from add_noise import random_noise

'''
    测试集loader
    输入下设class_num个子文件夹的主文件夹，输出img,label,img_path
'''
class Dataset_stl_test():
    def __init__(self, data_dir):

        self.label_name = {"airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
    

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        try:
            img = Image.open(path_img).convert('RGB')  # 0~255
        except IOError:
            print('Error')
            # width = Image.open(img_dir).width
            # height = Image.open(img_dir).height
            random_noise(3, 96, 96).save(path_img)
            img = Image.open(path_img).convert('RGB')  # 0~255
        transform = transforms.Compose([
        # transforms.Resize(96),
        # transforms.Resize(128),
        transforms.Resize((128,128)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    #    transforms.Normalize(norm_mean, norm_std),
    ])
       
        img = transform(img)   # 在这里做transform，转为tensor等等

        return img, label, path_img

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self,data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                #img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # STL-10
                img_names = list(filter(lambda x: x.endswith('.webp') or x.endswith('.JPEG') or x.endswith('.png'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        if len(data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return data_info

class Dataset_imagenet_test():
    def __init__(self, data_dir):

        self.label_name = {"electric_ray": 0, "goldfish": 1, "great_white_shark": 2, "hammerhead": 3, "stingray": 4, "tench": 5,
              "tiger_shark": 6}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        # print(path_img)
        try:
            img = Image.open(path_img).convert('RGB')  # 0~255
        except IOError:
            print('Error')
            # width = Image.open(img_dir).width
            # height = Image.open(img_dir).height
            random_noise(3, 256, 256).save(path_img)
            img = Image.open(path_img).convert('RGB')  # 0~255

        transform = transforms.Compose([
        # transforms.Resize(96),
        # transforms.Resize(128),
        transforms.Resize((256,256)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    #    transforms.Normalize(norm_mean, norm_std),
    ])

        img = transform(img)   # 在这里做transform，转为tensor等等

        return img, label, path_img

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self,data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            """
             os.walk() 中对应三个参数：
                root 所指的是当前正在遍历的这个文件夹的本身的地址
                dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
                files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
            """
            # 遍历类别
            for sub_dir in dirs:
                # sub_dir就是子类别图片的文件夹名
                img_names = os.listdir(os.path.join(root, sub_dir))
                #img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # img_names = list(filter(lambda x: x.endswith('.png'), img_names))

                # 将img_names作为可迭代对象放入filter中筛选出格式为JPEG的文件，得到新的列表
                # ImageNet
                img_names = list(filter(lambda x: x.endswith('.webp') or x.endswith('.JPEG') or x.endswith('.png'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        if len(data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return data_info

class Dataset_stl_train():
    def __init__(self, data_dir):
        self.label_name = {"airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本 
    def __getitem__(self, index):
        path_img, _ = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255
        transform = transforms.Compose([
        # transforms.Resize(96),
        # transforms.Resize(128),
        transforms.Resize((128,128)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    #    transforms.Normalize(norm_mean, norm_std),
    ])
       
        img = transform(img)   # 在这里做transform，转为tensor等等
        img = np.array(img)
        patches = np.reshape(img, (3, 1, 128, 1, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))
        img = torch.from_numpy(img).float()  # 不改变类型，
        patches = torch.from_numpy(patches).float()
        return img, patches, path_img

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self,data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                #img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        if len(data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return data_info

class Dataset_imagenet_train():
    def __init__(self, data_dir):
        self.label_name = {"electric_ray": 0, "goldfish": 1, "great_white_shark": 2, "hammerhead": 3, "stingray": 4, "tench": 5,
              "tiger_shark": 6}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
    def __getitem__(self, index):
        path_img, _ = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255
        transform = transforms.Compose([
        # transforms.Resize(96),
        # transforms.Resize(128),
        transforms.Resize((256,256)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    #    transforms.Normalize(norm_mean, norm_std),
    ])

        img = transform(img)   # 在这里做transform，转为tensor等等
        img = np.array(img)
        patches = np.reshape(img, (3, 1, 256, 1, 256))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))
        img = torch.from_numpy(img).float()#不改变类型，仍在0-255之间
        patches = torch.from_numpy(patches).float()

        return img, patches, path_img


    def __len__(self):
        return len(self.data_info)

    def get_img_info(self,data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                #img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # img_names = list(filter(lambda x: x.endswith('.png'), img_names))
                img_names = list(filter(lambda x: x.endswith('.JPEG'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        if len(data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return data_info

class Dataset_pascal(Dataset):
    #训练codec
    image_size = 448
    def __init__(self,root,list_file,transform):
        print('data init')
        self.root=root
        # self.train = train
        self.transform=transform
        self.fnames = []#文件名
        self.boxes = []
        self.labels = []
        # self.mean = (123,117,104)#RGB

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))#' '.join(list_file)用空格链接列表里的元素
            # os.system('cat %s > %s' % (a.test, b.test)) 创建b,把a里面的内容复制给b
            list_file = tmp_file
            # print(list_file)
        with open(list_file) as f:
            lines  = f.readlines()

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box=[]
            label=[]
            for i in range(num_boxes):
                x = float(splited[1+5*i])
                y = float(splited[2+5*i])
                x2 = float(splited[3+5*i])
                y2 = float(splited[4+5*i])
                c = splited[5+5*i]
                box.append([x,y,x2,y2])
                label.append(int(c)+1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        # a=os.path.join(self.root, fname)
        # print(fname)
        img = cv2.imread(os.path.join(self.root, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        # if self.train:
        #     #img = self.random_bright(img)
        #     img, boxes = self.random_flip(img, boxes)#翻转
        #     img,boxes = self.randomScale(img,boxes)#随机尺度变换
        #     img = self.randomBlur(img)#随机模糊
        #     img = self.RandomBrightness(img)#随机增加亮度
        #     img = self.RandomHue(img)#随机改变色调
        #     img = self.RandomSaturation(img)#随机改变饱和度
        #     img,boxes,labels = self.randomShift(img,boxes,labels)#随机飘移
        #     img,boxes,labels = self.randomCrop(img,boxes,labels)#随机裁剪
        # # #debug
        # box_show = boxes.numpy().reshape(-1)
        # print(box_show)
        # img_show = self.BGR2RGB(img)
        # pt1=(int(box_show[0]),int(box_show[1])); pt2=(int(box_show[2]),int(box_show[3]))
        # cv2.rectangle(img_show,pt1=pt1,pt2=pt2,color=(0,255,0),thickness=1)
        # plt.figure()
        
        # # cv2.rectangle(img,pt1=(10,10),pt2=(100,100),color=(0,255,0),thickness=1)
        # plt.imshow(img_show)
        # plt.show()
        # #debug
        h,w,_ = img.shape
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img = self.BGR2RGB(img) #because pytorch pretrained model use RGB
        # img = self.subMean(img,self.mean) #减去均值
        img = cv2.resize(img,(self.image_size,self.image_size))
        target = self.encoder(boxes,labels)# 7x7x30
        for t in self.transform:
            img = t(img)

        img = np.array(img)
        patches = np.reshape(img, (3, 1, 448, 1, 448))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))
        img = torch.from_numpy(img).float()#不改变类型，仍在0-255之间
        patches = torch.from_numpy(patches).float()
        
        # return img,target
        # return img,target,patches,fname
        return img,target,patches
    def __len__(self):
        return self.num_samples

    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        # grid_num = 14
        grid_num = 7
        target = torch.zeros((grid_num,grid_num,30))
        cell_size = 1./grid_num
        wh = boxes[:,2:]-boxes[:,:2]
        cxcy = (boxes[:,2:]+boxes[:,:2])/2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 #
            target[int(ij[1]),int(ij[0]),4] = 1
            target[int(ij[1]),int(ij[0]),9] = 1
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
            xy = ij*cell_size #匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample -xy)/cell_size
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            target[int(ij[1]),int(ij[0]),:2] = delta_xy
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
        return target
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    
    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels

    def randomScale(self,bgr,boxes):
        #固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in
        return bgr,boxes,labels


    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

class Dataset_pascal_reconstruct(Dataset):
    image_size = 448
    def __init__(self,root,list_file,transform):
        print('data init')
        self.root=root
        # self.train = train
        self.transform=transform
        self.fnames = []#文件名
        self.boxes = []
        self.labels = []
        # self.mean = (123,117,104)#RGB

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))#' '.join(list_file)用空格链接列表里的元素
            # os.system('cat %s > %s' % (a.test, b.test)) 创建b,把a里面的内容复制给b
            list_file = tmp_file
            # print(list_file)
        with open(list_file) as f:
            lines  = f.readlines()

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box=[]
            label=[]
            for i in range(num_boxes):
                x = float(splited[1+5*i])
                y = float(splited[2+5*i])
                x2 = float(splited[3+5*i])
                y2 = float(splited[4+5*i])
                c = splited[5+5*i]
                box.append([x,y,x2,y2])
                label.append(int(c)+1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        # a=os.path.join(self.root, fname)
        # print(fname)
        img_original = cv2.imread(os.path.join(self.root, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        h,w,_ = img_original.shape
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img = self.BGR2RGB(img_original) #because pytorch pretrained model use RGB
        # img = self.subMean(img,self.mean) #减去均值
        img = cv2.resize(img,(self.image_size,self.image_size))
        target = self.encoder(boxes,labels)# 7x7x30
        for t in self.transform:
            img = t(img)

        img = np.array(img)
        patches = np.reshape(img, (3, 1, 448, 1, 448))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))
        img = torch.from_numpy(img).float()#不改变类型，仍在0-255之间
        patches = torch.from_numpy(patches).float()
        
        # return img,target
        return img_original,target,patches,fname
        # return img,target,patches
    def __len__(self):
        return self.num_samples
    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        # grid_num = 14
        grid_num = 7
        target = torch.zeros((grid_num,grid_num,30))
        cell_size = 1./grid_num
        wh = boxes[:,2:]-boxes[:,:2]
        cxcy = (boxes[:,2:]+boxes[:,:2])/2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 #
            target[int(ij[1]),int(ij[0]),4] = 1
            target[int(ij[1]),int(ij[0]),9] = 1
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
            xy = ij*cell_size #匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample -xy)/cell_size
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            target[int(ij[1]),int(ij[0]),:2] = delta_xy
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
        return target
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

class Dataset_pascal_si(Dataset):
    #训练codec
    image_size = 448
    def __init__(self,root_ori,root_rec,list_file,transform):
        print('data init')
        self.root_ori=root_ori
        self.root_rec=root_rec
        # self.train = train
        self.transform=transform
        self.fnames = []#文件名
        self.boxes = []
        self.labels = []
        # self.mean = (123,117,104)#RGB

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))#' '.join(list_file)用空格链接列表里的元素
            # os.system('cat %s > %s' % (a.test, b.test)) 创建b,把a里面的内容复制给b
            list_file = tmp_file
            # print(list_file)
        with open(list_file) as f:
            lines  = f.readlines()

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box=[]
            label=[]
            for i in range(num_boxes):
                x = float(splited[1+5*i])
                y = float(splited[2+5*i])
                x2 = float(splited[3+5*i])
                y2 = float(splited[4+5*i])
                c = splited[5+5*i]
                box.append([x,y,x2,y2])
                label.append(int(c)+1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        # a=os.path.join(self.root, fname)
        # print(fname)
        img_ori = cv2.imread(os.path.join(self.root_ori, fname))
        img_rec = cv2.imread(os.path.join(self.root_rec, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        # if self.train:
        #     #img = self.random_bright(img)
        #     img, boxes = self.random_flip(img, boxes)#翻转
        #     img,boxes = self.randomScale(img,boxes)#随机尺度变换
        #     img = self.randomBlur(img)#随机模糊
        #     img = self.RandomBrightness(img)#随机增加亮度
        #     img = self.RandomHue(img)#随机改变色调
        #     img = self.RandomSaturation(img)#随机改变饱和度
        #     img,boxes,labels = self.randomShift(img,boxes,labels)#随机飘移
        #     img,boxes,labels = self.randomCrop(img,boxes,labels)#随机裁剪
        # # #debug
        # box_show = boxes.numpy().reshape(-1)
        # print(box_show)
        # img_show = self.BGR2RGB(img)
        # pt1=(int(box_show[0]),int(box_show[1])); pt2=(int(box_show[2]),int(box_show[3]))
        # cv2.rectangle(img_show,pt1=pt1,pt2=pt2,color=(0,255,0),thickness=1)
        # plt.figure()
        
        # # cv2.rectangle(img,pt1=(10,10),pt2=(100,100),color=(0,255,0),thickness=1)
        # plt.imshow(img_show)
        # plt.show()
        # #debug
        h,w,_ = img_ori.shape
        h_r,w_r,_=img_rec.shape
        print(h,w)
        print(h_r,w_r)
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img_ori = self.BGR2RGB(img_ori) #because pytorch pretrained model use RGB
        img_ori = cv2.resize(img_ori,(self.image_size,self.image_size))
        img_rec = self.BGR2RGB(img_rec) #because pytorch pretrained model use RGB
        img_rec = cv2.resize(img_rec,(self.image_size,self.image_size))

        target = self.encoder(boxes,labels)# 7x7x30
        for t in self.transform:
            img_ori = t(img_ori)
            img_rec = t(img_rec)
        # img_ori = np.array(img_ori)
        # patches = np.reshape(img_ori, (3, 1, 448, 1, 448))
        # patches = np.transpose(patches, (0, 1, 3, 2, 4))
        # img_ori = torch.from_numpy(img_ori).float()#不改变类型，仍在0-255之间
        # patches = torch.from_numpy(patches).float()

        # return img,target
        # return img,target,patches,fname
        return img_ori,img_rec,target
    def __len__(self):
        return self.num_samples

    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        # grid_num = 14
        grid_num = 7
        target = torch.zeros((grid_num,grid_num,30))
        cell_size = 1./grid_num
        wh = boxes[:,2:]-boxes[:,:2]
        cxcy = (boxes[:,2:]+boxes[:,:2])/2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 #
            target[int(ij[1]),int(ij[0]),4] = 1
            target[int(ij[1]),int(ij[0]),9] = 1
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
            xy = ij*cell_size #匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample -xy)/cell_size
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            target[int(ij[1]),int(ij[0]),:2] = delta_xy
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
        return target
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    
    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels

    def randomScale(self,bgr,boxes):
        #固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in
        return bgr,boxes,labels


    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

'''
训练集不区分文件夹时的加载

class Dataset_stl_train(Dataset):
    """
    Image shape is (96, 96, 3)  --> 1x1 128x128 x 3 patches
    输入文件夹路径，文件夹里是一堆图片
    返回这些图片每一张的img(tensor0-1),patch(tensor0-1),path单张图片的路径
    训练集loader
    """

    def __init__(self, folder_path):
        # self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.files = glob.glob('%s/*.*' % folder_path)

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img = Image.open(path)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            #transforms.ToTensor(),
            # transforms.Normalize([0.4431991, 0.42826223, 0.39535823], [0.25746644, 0.25306803, 0.26591763])
        ])
        img = transform(img)
        # pad = ((24, 24), (0, 0), (0, 0))

        # img = np.pad(img, pad, 'constant', constant_values=0) / 255
        # img = np.pad(img, pad, mode='edge') / 255.0
        img = np.array(img)
        # print(img.shape)
        # img = np.transpose(img, (2, 0, 1))
        # img = torch.from_numpy(img).float()#from_numpy完成numpy到tensor转变
        # img = img.float()
        # 转成numpy才能reshape

        patches = np.reshape(img, (3, 1, 128, 1, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))
        img = torch.from_numpy(img).float()  # 不改变类型，

        return img, patches, path

    def get_random(self):
        i = np.random.randint(0, len(self.files))
        return self[i]

    def __len__(self):
        return len(self.files)
'''

# class Dataset_imagenet_train(Dataset):
#     '''
#     #  Image shape 是未知的imagenet图片  --> 1x1 256*256 x 3 patches
#     # 输入文件夹路径，文件夹里是一堆图片
#     # 返回这些图片每一张的img(tensor0-1),patch(tensor0-1),path单张图片的路径
#     # '''
#     def __init__(self, folder_path):
#         # self.files = sorted(glob.glob('%s/*.*' % folder_path))#%s的位置将会被替换成新的字符串，替换的内容是%后面的folder_path
#         self.files = glob.glob('%s/*.*' % folder_path)
#     def __getitem__(self, index):
#         path = self.files[index % len(self.files)]
#         img = Image.open(path)
#         # h, w, c = img.shape
#         transform = transforms.Compose([
#             transforms.Resize((256, 256)),#resize函数必须是PIL才可以
#             transforms.ToTensor(),#0-255到0-1
#             #transforms.Resize(256),
#             #transforms.CenterCrop(224),
#             #transforms.ToTensor(),
#             # transforms.Normalize([0.4431991, 0.42826223, 0.39535823], [0.25746644, 0.25306803, 0.26591763])
#         ])

#         img = transform(img)
#         # pad = ((24, 24), (0, 0), (0, 0))

#         # img = np.pad(img, pad, 'constant', constant_values=0) / 255
#         # img = np.pad(img, pad, mode='edge') / 255.0
#         img = np.array(img)
#         # print(img.shape)
#         # img = np.transpose(img, (2, 0, 1))
#         # img = torch.from_numpy(img).float()#from_numpy完成numpy到tensor转变
#         # img = img.float()
#         #转成numpy才能reshap

#         patches = np.reshape(img, (3, 1, 256, 1, 256))
#         patches = np.transpose(patches, (0, 1, 3, 2, 4))
#         img = torch.from_numpy(img).float()#不改变类型，仍在0-255之间
#         patches = torch.from_numpy(patches).float()
#         return img, patches, path

#     def get_random(self):
#         i = np.random.randint(0, len(self.files))
#         return self[i]

#     def __len__(self):
#         return len(self.files)
