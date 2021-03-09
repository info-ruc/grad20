#MNIST数据集测试
import torch,torchvision,fastai
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import time,os
import torch.utils.data as Data
import itertools
from models import Corr_AE,Extended_Corr_AE,CFAE
import cv2 as cv
from torchvision import transforms
from PIL import Image

def get_loss(a,b):
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    result = 0
    for i in range(len(a)):
        result += (a[i]-b[i])**2
    return result/len(a)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if __name__ == '__main__':
    train_data = torchvision.datasets.MNIST(root='F:/python/mnist', train=True)
    '''train_x = []
    for i in range(1000):
        train_x.append(transform(Image.open('F:/mnist/train/'+str(i)+'.png')).numpy())
    train_x = torch.tensor(train_x).float()'''
    _train_y = train_data.targets.numpy()
    train_y = []
    for y in _train_y:
        temp = [0] * 100
        temp[y*10] = 1
        train_y.append(temp)
    train_y = torch.tensor(train_y).float()
    train_x = train_data.data.view(-1,28*28).float()/255

    test_data = torchvision.datasets.MNIST(root='F:/python/mnist', train=False)
    '''test_x = []
    for i in range(100):
        test_x.append(transform(Image.open('F:/mnist/test/'+str(i)+'.png')).numpy())
    test_x = torch.tensor(test_x).float()'''
    test_y = test_data.targets.numpy()
    test_x = test_data.data.view(-1,28*28).float()/255

    #model = Extended_Corr_AE.ECA(text_size=100,model_name='vgg19',pretrained=False)
    #model = Corr_AE.Corr_AE(text_size=100,img_size=28*28)
    model = CFAE.CFAE(text_size=100,img_size=28*28)
    model.train(texts=train_y,imgs=train_x,batch_size=256,EPOCH=200,alpha=0.4,num_workers=6)
    print(model.loss)

    accuracy = 0
    for i in range(len(test_x)):
        x = test_x[i].cuda()
        #x = model.emodel(torch.tensor([test_x[i].numpy()]).cuda())[0]
        #encode, decode = model.img_model(x)
        encode,decode1,decode2 = model.img_model(x)
        result = 0
        min_loss = 1000
        for j in range(10):
            temp = [0] * 100
            temp[j * 10] = 1
            a, b, c= model.text_model(torch.tensor(temp).float().cuda())
            loss = get_loss(a, encode)
            if loss < min_loss:
                min_loss = loss
                result = j
        if result == test_y[i]:
            accuracy += 1
    print("accuracy:",accuracy/len(test_x))


    '''while 1:
        i = int(input('input:'))
        if i == -1:
            break
        img = test_data.data[i]
        plt.imshow(img,cmap='gray')
        plt.show()

        x = test_x[i].cuda()
        encode,decode = model.img_model(x)
        _img = decode.detach().cpu().view(28,28).numpy()*255
        plt.imshow(_img,cmap='gray')
        plt.show()

        result = 0
        min_loss = 1000
        for i in range(10):
            temp = [0]*100
            temp[i*10] = 1
            a,b = model.text_model(torch.tensor(temp).float().cuda())
            loss = get_loss(a,encode)
            print(loss)
            if loss < min_loss:
                min_loss = loss
                result = i
        print(result)'''