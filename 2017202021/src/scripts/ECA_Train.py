from models import Extended_Corr_AE
from torchvision import transforms
import torch,torchvision
import matplotlib.pyplot as plt
import csv
from data_processing import text_feature_get
from multiprocessing import cpu_count
import threading
import time,os
from PIL import Image
n = 2#信号量，保证数据读取完毕后再进行模型的训练
transform = transforms.Compose([
    transforms.Resize((250,300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class TextLoaderThread(threading.Thread):
    def __init__(self):
        super(TextLoaderThread,self).__init__()
    def run(self):
        global _text_data,text_data
        text_data = text_feature_get.get_text_feature(texts=_text_data)
        global n
        n -= 1
#文本信息读取线程

class ImgLoaderThread(threading.Thread):
    def __init__(self):
        super(ImgLoaderThread,self).__init__()
    def run(self):
        global _img_data,img_data
        i = 1
        for img in _img_data:
            if i % 1000 == 0:
                print(i, "images of", len(_img_data), "images have been loaded")
            i += 1
            img_data.append(transform(Image.open(img)).numpy())
        img_data = torch.tensor(img_data).float()
        global n
        n -= 1
#图像信息读取线程


if __name__ == '__main__':
    x = input("input path of data:")
    _text_data = []
    _img_data = []
    texts = list(csv.reader(open(x + '/cxr/report/indiana_reports.csv', encoding='utf-8')))[1:]
    texts = {texts[i][0]: texts[i][6] for i in range(len(texts)) if texts[i][6] != ""}
    imgs = list(csv.reader(open(x + '/cxr/report/indiana_projections.csv', encoding='utf-8')))[1:]

    for i in range(len(imgs)):
        uid = imgs[i][0]
        filename = 'CXR' + imgs[i][1].replace('.dcm', '')
        if uid in texts:
            _text_data.append(texts[uid])
            _img_data.append(x + '/cxr/image/' + filename)
            _text_data.append(texts[uid])
            _img_data.append(x + '/cxr/image/' + 'flip_' + filename)

    text_data = []
    img_data = []
    t1 = TextLoaderThread()
    t2 = ImgLoaderThread()
    t1.start()
    t2.start()
    # 通过两个线程同时对图像数据和文本数据
    while n:
        time.sleep(5)
    # 每5秒主线程检查数据是否读取完毕
    model_name = ['vgg19','alexnet','densenet161','resnet101','squeezenet1_0','inception_v3']
    while True:
        x = int(input("choose which model will be trained (0-5):"))
        if x < 0 or x > 5:
            break
        model = Extended_Corr_AE.ECA(text_size=len(text_data[0]),model_name=model_name[x])
        for i in range(4):
            model.train(texts=text_data, imgs=img_data, num_workers=cpu_count(), batch_size=128, EPOCH=100,alpha=0.3)
            model.save()
    #训练六种不同的模型并保存