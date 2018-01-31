#coding:utf-8
import sys
sys.path.insert(1, "./crnn")

import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable 
import numpy as np
import os
import util
import dataset
from PIL import Image
import models.crnn as crnn
import keys
from math import *
import mahotas
import cv2
import matplotlib.pyplot as plt

def dumpRotateImage(img,degree,pt1,pt2,pt3,pt4):
    height,width=img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)
    
    
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut=imgRotation[int(pt1[1]):int(pt3[1]),int(pt1[0]):int(pt3[0])]
    height,width=imgOut.shape[:2]
    return imgOut

def crnnSource():
    alphabet = keys.alphabet
    converter = util.strLabelConverter(alphabet)
    # model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1).cuda()
    model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1)
    # path = './crnn/samples/netCRNN63.pth'
    path = './crnn/samples/netCRNNcpu.pth'
    model.load_state_dict(torch.load(path))
    return model, converter

def crnnRec(model,converter,im,text_recs):
    index = 0
    for rec in text_recs:
        pt1 = (rec[0],rec[1])
        pt2 = (rec[2],rec[3])
        pt3 = (rec[6],rec[7])
        pt4 = (rec[4],rec[5])
        partImg = dumpRotateImage(im, degrees(atan2(pt2[1]-pt1[1],pt2[0]-pt1[0])),pt1,pt2,pt3,pt4)
        if partImg.shape[0] == 0 or partImg.shape[1] == 0:
            return
        #mahotas.imsave('%s.jpg'%index, partImg)
        # plt.imshow(im, cmap='gray')
        # plt.plot(pt1[0], pt1[1], 'bo')
        # plt.plot(pt2[0], pt2[1], 'bo')
        # plt.plot(pt3[0], pt3[1], 'bo')
        # plt.plot(pt4[0], pt4[1], 'bo')
        # plt.show()
        # return

        image = Image.fromarray(partImg).convert('L')
        #height,width,channel=partImg.shape[:3]
        #print(height,width,channel)
        #print(image.size) 
 
        #image = Image.open('./img/t4.jpg').convert('L')
        scale = image.size[1]*1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        #print(w)
 
        transformer = dataset.resizeNormalize((w, 32))
        # image = transformer(image).cuda()
        image = transformer(image)
        image = image.view(1, *image.size())
        image = Variable(image, volatile=True)
        model.eval()
        preds = model(image)
        _, preds = preds.max(2)
        preds = preds.squeeze(0).squeeze(0)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        #print('%-20s => %-20s' % (raw_pred, sim_pred))
        print(index)
        print(sim_pred)
        index = index + 1

