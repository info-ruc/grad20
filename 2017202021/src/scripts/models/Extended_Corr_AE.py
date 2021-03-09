#通过迁移学习实现对图像数据特征的提取，并搭建拓展的跨模态自编码器
import AutoEncoder
import torch,torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time,os,sys
import torch.utils.data as Data
import itertools

class ECA():
    def __init__(self,text_size,model_name,img_feature_size=1000,pretrained=True):
        self.text_size = text_size
        self.img_feature_size = img_feature_size
        self.img_model = AutoEncoder.AutoEncoder(self.img_feature_size)
        self.text_model = AutoEncoder.AutoEncoder(self.text_size)
        self.model_name = model_name
        if self.model_name == "vgg19":
            self.emodel = torchvision.models.vgg19(pretrained=pretrained)
        elif self.model_name == "alexnet":
            self.emodel = torchvision.models.alexnet(pretrained=pretrained)
        elif self.model_name == "densenet161":
            self.emodel = torchvision.models.densenet161(pretrained=pretrained)
        elif self.model_name == "resnet101":
            self.emodel = torchvision.models.resnet101(pretrained=pretrained)
        elif self.model_name == "squeezenet1_0":
            self.emodel = torchvision.models.squeezenet1_0(pretrained=pretrained)
        elif self.model_name == "inception_v3":
            self.emodel = torchvision.models.inception_v3(pretrained=pretrained)
        if torch.cuda.is_available():
            try:
                self.img_model = self.img_model.cuda()
                self.text_model = self.text_model.cuda()
                self.emodel = self.emodel.cuda()
            except:
                pass
        self.loss = []

    def train(self, texts, imgs, learning_rate=0.001, EPOCH=20, alpha=0.2, num_workers=4, batch_size=32,
              pin_memory=True):
        data_set = Data.TensorDataset(texts, imgs)
        data_loader = Data.DataLoader(dataset=data_set, num_workers=num_workers, shuffle=True, batch_size=batch_size,
                                      pin_memory=pin_memory)
        optimizer = optim.Adam(params=itertools.chain(self.text_model.parameters(), self.img_model.parameters(),self.emodel.parameters()),
                               lr=learning_rate)
        loss_func = nn.MSELoss().cuda()
        for epoch in range(EPOCH):
            begin = time.time()
            eloss = 0
            i = 0
            for step,data in enumerate(data_loader):
                i += 1
                text,img = data
                text = Variable(text.cuda())
                img = Variable(img.cuda())
                img_feature = Variable(self.emodel(img))
                img_encode,img_decode = self.img_model(img_feature)
                text_encode,text_decode = self.text_model(text)
                loss =alpha*loss_func(img_encode,text_encode)+ (1-alpha)*(loss_func(img_feature,img_decode) + loss_func(text,text_decode))
                eloss += loss.cpu().detach().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("epoch:{e} running time: {n}".format(e=str(epoch), n=str(time.time() - begin)))
            self.loss.append(eloss/i)
    def save(self):
        try:
            os.mkdir('trained_models/')
        except:
            pass
        try:
            os.mkdir('trained_models/Extended_Corr_AE/')
        except:
            pass
        try:
            os.mkdir('trained_models/Extended_Corr_AE/{e}_Corr_AE/'.format(e=self.model_name))
        except:
            pass
        path = 'trained_models/Extended_Corr_AE/{e}_Corr_AE/'.format(e=self.model_name)
        torch.save(self.text_model,path+"text_model.model")
        torch.save(self.img_model,path+"img_model.model")
        torch.save(self.emodel,path+self.model_name+".model")
        f = open(path+'loss.txt','a')
        for loss in self.loss:
            f.write(str(loss))
            f.write('\n')
    def load(self,path = 'trained_models/Extended_Corr_AE/'):
        try:
            self.text_model = torch.load(path+"{e}_Corr_AE/text_model.model".format(e=self.model_name))
            self.img_model = torch.load(path+"{e}_Corr_AE/img_model.model".format(e=self.model_name))
            self.emodel = torch.load(path+"{e}_Corr_AE/{e}.model".format(e=self.model_name))
        except:
            path = input("input path of Corr_AE:\n")
            self.text_model = torch.load(path + "{e}_Corr_AE/text_model.model".format(e=self.model_name))
            self.img_model = torch.load(path + "{e}_Corr_AE/img_model.model".format(e=self.model_name))
            self.emodel = torch.load(path + "{e}_Corr_AE/{e}.model".format(e=self.model_name))
