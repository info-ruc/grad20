#全模态的自编码器
import DAE,torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time,os,sys
import torch.utils.data as Data
import itertools
class CFAE():
    def __init__(self, text_size: object, img_size: object) -> object:
        self.text_size = text_size
        self.img_size = img_size
        self.text_model = DAE.DAE(input_size=text_size,output_size1=text_size,output_size2=img_size)
        self.img_model = DAE.DAE(input_size=img_size,output_size1=text_size,output_size2=img_size)
        if torch.cuda.is_available():
            try:
                self.img_model = self.img_model.cuda()
                self.text_model = self.text_model.cuda()
            except:
                pass

    def train(self,texts,imgs,learning_rate=0.001,EPOCH=20,alpha=0.2,num_workers=4,batch_size=32,pin_memory=True):
        self.loss = []
        self.text_model.train()
        self.img_model.train()
        data_set = Data.TensorDataset(texts,imgs)
        data_loader = Data.DataLoader(dataset=data_set,num_workers=num_workers,shuffle=True,batch_size=batch_size,pin_memory=pin_memory)
        optimizer = optim.Adam(params=itertools.chain(self.text_model.parameters(), self.img_model.parameters()),
                               lr=learning_rate)
        loss_func = nn.MSELoss().cuda()
        for epoch in range(EPOCH):
            begin = time.time()
            eloss = 0
            i = 0
            for step, data in enumerate(data_loader):
                i += 1
                x, y = data
                x = Variable(x.cuda())
                y = Variable(y.cuda())
                x_encode, x_decode1, x_decode2 = self.text_model(x)
                y_encode, y_decode1, y_decode2 = self.img_model(y)
                loss = alpha * loss_func(x_encode,y_encode) + (1 - alpha) * (loss_func(x,x_decode1)+loss_func(y,x_decode2)+loss_func(x,y_decode1)+loss_func(y,y_decode2))
                eloss += loss.cpu().detach().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("epoch:{e} running time: {n}".format(e=str(epoch), n=str(time.time() - begin)))
            self.loss.append(eloss / i)
    #模型训练
    def save(self):
        try:
            os.mkdir('trained_models/')
            os.mkdir('trained_models/CFAE/')
            path = 'trained_models/CFAE/'
        except:
            path = 'trained_models/CFAE/'
        torch.save(self.text_model,path+"text_model.model")
        torch.save(self.img_model,path+"img_model.model")
        f = open(path+'loss.txt','a')
        for loss in self.loss:
            f.write(str(loss))
            f.write('\n')
    #保存模型及训练损失
    def load(self,path = 'trained_models/CFAE/'):
        try:
            self.text_model = torch.load(path+"text_model.model")
            self.img_model = torch.load(path+"img_model.model")
        except:
            path = input("input path of CFAE:\n")
            self.text_model = torch.load(path + "text_model.model")
            self.img_model = torch.load(path + "img_model.model")
    #加载模型，默认将模型放在项目文件夹下，否则需要自行输入模型位置
    def predict(self,image=None,text=None):
        from data_processing import img_feature_get,text_feature_get
        if torch.cuda.is_available():
            self.text_model.cuda()
            self.img_model.cuda()
        else:
            self.text_model.cpu()
            self.img_model.cpu()
        img_encode, img_decode1,img_decode2 = ([],[],[])
        text_encode, text_decode1, text_decode2 = ([],[],[])
        if image:
            a = img_feature_get.get_img_feature([image])[0]
            if torch.cuda.is_available():
                a = a.cuda()
            img_encode,img_decode1,img_decode2 = self.img_model(a)
        if text:
            a = text_feature_get.get_text_feature([text])[0]
            if torch.cuda.is_available():
                a = a.cuda()
            text_encode,text_decode1,text_decode2 = self.text_model(a)

        if image and text:
            return  img_encode.detach().cpu(),text_encode.detach().cpu()
        if image:
            return img_encode.detach().cpu()
        if text:
            return text_encode.detach().cpu()
    #将输入的数据映射到公共表示区间
    def get_similarity(self,a,b):
        loss_func = nn.MSELoss()
        if torch.cuda.is_available():
            loss_func = loss_func.cuda()
            a = a.cuda()
            b = b.cuda()
        else:
            loss_func = loss_func.cpu()
            a = a.cpu()
            b = b.cpu()
        return loss_func(a,b).detach().cpu().numpy()
    #获取两个向量的均方差