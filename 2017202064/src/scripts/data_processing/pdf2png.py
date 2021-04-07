# 有的图片可能是pdf格式，有的图片可能是jpg格式，有的可能是jpeg格式，统一转化为png格式
import os
import time
import re
import pathlib
import shutil
from multiprocessing.pool import Pool
from pdf2image import convert_from_path

dataone = "C:/Users/13170/Desktop/dataone/"

def pdf2image(file):
    if os.path.splitext(file)[1] == '.pdf':
        pages = convert_from_path(dataone+file,200)
        pages[0].save(dataone+os.path.splitext(file)[0]+'.png','PNG')
        


if __name__ == '__main__':
    files = os.listdir(dataone)
    with Pool(4) as pool:
        pool.map(pdf2image,files)
# for file in files:
#     if os.path.splitext(file)[1] == '.pdf':
#         os.remove(dataone+file)
#         # print("delete")
