#!/usr/bin/env python
#coding:utf-8

from CNN import CNN
#  from my_CNN import CNN
from animeface import AnimeFaceDataset
from my_animeface import MyAnimeFaceDataset
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import cv2 as cv
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='process')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

if args.gpu >=0:
    cuda.get_device_from_id(args.gpu).use()

print "!!!!! Load AnimeFace Dataset !!!!!"

dataset = AnimeFaceDataset()
dataset.load_data_target()
data = dataset.data
target = dataset.target
n_outputs = dataset.get_n_types_target()

#  print "n_outputs : ", n_outputs
#  print "data : ", data


print "!!!!! Load MyAnimeFace Dataset !!!!!"
my_dataset = MyAnimeFaceDataset()
my_dataset.load_dataset()
my_data = my_dataset.data

print "my_data.shape : ", my_data.shape
image = my_data[0].transpose(1, 2, 0)
#  image = cv.resize(image, (256,256))

plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

cnn = CNN(data=data, target=target, gpu=args.gpu, n_outputs=n_outputs)

cnn.load_model()
print "cnn.model : ", cnn.model

print cnn.predict(my_data, args.gpu)


plt.show()
