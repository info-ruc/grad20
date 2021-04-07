import os
import time
import re
import pathlib
import shutil

#跳过了cap小于10的情况，也就是描述不够

path = "C:/Users/13170/Desktop/Arxiv6K/"
dataone = "C:/Users/13170/Desktop/dataone/"
count = 1

files = os.listdir(path)   #arri下所有2XXX.XXXXX 路径

for file in files:            #2XXXX中所有文件
    f = os.listdir(path+file)
    for i in f:
        if os.path.splitext(i)[1] == '.tex':
            time.sleep(0.1)
            
            texfile = open(path+file+'/'+i,'rb')
#             print(path+file+'/'+i)        #打开了tex文件
            texread = texfile.read()
#             print(texread)
            
            try:
                texfrombe= re.findall(r'(?<=begin).*?(?=end)',texread.decode(encoding='utf-8'),re.S)
            except:
                pass   
            
            #找到gbegin和end之间的内容，找不到就pass
            
            if texfrombe:
                for j in texfrombe:       #texfrombe中有多个begin到end
                    includegraph = re.findall(r'(?<=\\includegraphics).*?(?=}\n)',j,re.S)      #每个begin到end中includegraphics开头的
                    cap = re.findall(r'(?<=\\caption\{).*?(?=\.})',j,re.S)
                    if not cap:
                        cap = re.findall(r'(?<=\\caption\{).*?(?=\})',j,re.S)#每个begin到end中的caption
               #caption可能以其他形式结尾
#                     print(includegraph)
                    if cap and len(cap[0])<10:
                        continue
                    if includegraph and cap:                                                   #图像存在，caption存在
#                         print(includegraph)
#                         print(j)
#                         print(cap)
#                         print('before graphtail')
                        for i in includegraph:                                #图像可能有多个，但是caption只有一个，一定是多对一或者一对一
                            graphtail = re.findall(r'(?<=\]\{).*',i,re.S)      #去include后面找图像尾巴
                            if not graphtail:                                 #找不到说明没跟【】，那就只考虑{}中的内容
                                graphtail = re.findall(r'(?<=\{).*',i,re.S)
                            print(graphtail)
                            graphpath = pathlib.Path(path+file+'/'+graphtail[0])  #拼接图像路径
#                             print(graphpath)
                            poss = os.listdir(path+file)                        #有可能路径只在file中，但是没有指明是fig文件夹下
                            print(graphpath)
                            try:
                                if os.path.exists(graphpath):
                                    shutil.copy(graphpath,dataone+str(count)+os.path.splitext(graphtail[0])[1])
                                    texwrite = open(dataone+str(count)+'.txt','w',encoding = 'utf-8')
                                    texwrite.write(cap[0])
                                    texwrite.close
                                    count = count +1

                                else:                          #如果路径不存在的话，看看file文件夹下会不会包含这张图片
                                    for dire in poss:
                                        if os.path.exists(path+file+'/'+dire+graphtail[0]):   #与之前的区别就是，我检索了一层文件夹下会不会包含这张图片
                                            shutil.copy(graphpath,dataone+str(count)+os.path.splitext(graphtail[0])[1])
                                            texwrite = open(dataone+str(count)+'.txt','w',encoding = 'utf-8')
                                            texwrite.write(cap[0])
                                            texwrite.close
                                            count = count+1
                            except:
                                pass
            texfile.close