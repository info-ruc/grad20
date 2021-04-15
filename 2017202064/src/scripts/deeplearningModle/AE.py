#跨模态的自编码器
import AutoEncoder,torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time,os
import torch.utils.data as Data
import itertools
class Corr_AE():
    def __init__(self,text_size,img_size):
        self.text_size = text_size
        self.img_size = img_size
        self.text_model = AutoEncoder.AutoEncoder(input_size=text_size).to("cuda")
        self.img_model = AutoEncoder.AutoEncoder(input_size=img_size).to("cuda")

    def train(self,texts,imgs,learning_rate=0.001,EPOCH=20,alpha=0.2,num_workers=4,batch_size=32,pin_memory=True):
        self.loss = []
        data_set = Data.TensorDataset(texts,imgs)
        data_loader = Data.DataLoader(dataset=data_set,num_workers=num_workers,shuffle=True,batch_size=batch_size,pin_memory=pin_memory)
        optimizer = optim.Adam(params=itertools.chain(self.text_model.parameters(), self.img_model.parameters()), lr=learning_rate)
        loss_func = nn.MSELoss().cuda()
        for epoch in range(EPOCH):
            begin = time.time()
            eloss = 0
            i = 0
            for step,data in enumerate(data_loader):
                i += 1
                x,y = data
                x = Variable(x.cuda())
                y = Variable(y.cuda())
                x_encode, x_decode = self.text_model(x)
                y_encode, y_decode = self.img_model(y)
                loss = (1-alpha) * (loss_func(x,x_decode) + loss_func(y,y_decode)) + alpha * loss_func(x_encode,y_encode)
                eloss += loss.cpu().detach().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("epoch:{e} running time: {n}".format(e=str(epoch), n=str(time.time() - begin)))
            self.loss.append(eloss/i)

    def save(self):
        try:
            os.mkdir('trained_models/')
            os.mkdir('trained_models/Corr_AE/')
            path = 'trained_models/Corr_AE/'
        except:
            path = 'trained_models/Corr_AE/'
        torch.save(self.text_model,path+"text_model.model")
        torch.save(self.img_model,path+"img_model.model")
        f = open(path+'loss.txt','w')
        for loss in self.loss:
            f.write(str(loss))
            f.write('\n')

    def load(self):
        path = 'trained_models/Corr_AE/'
        try:
            self.text_model = torch.load(path+"text_model.model")
            self.img_model = torch.load(path+"img_model.model")
        except:
            path = input("input path of Corr_AE model:\n")
            self.text_model = torch.load(path + "text_model.model")
            self.img_model = torch.load(path + "img_model.model")