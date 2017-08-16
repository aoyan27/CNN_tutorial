#!/usr/bin/env python
#coding:utf-8

from CNN import CNN
from my_dataset import MyDataset
from chainer import cuda

import argparse

parser = argparse.ArgumentParser(description='process')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

if args.gpu >=0:
    cuda.get_device_from_id(args.gpu).use()

print "!!!!! Load AnimeFace Dataset !!!!!"

dataset = MyDataset()
dataset.load_data_target()
data = dataset.data
target = dataset.target
n_outputs = dataset.n_types_target
#  n_outputs = len(target)
print "n_outputs : ", n_outputs
#  print "data : ", data

cnn = CNN(data=data, target=target, gpu=args.gpu, n_outputs=n_outputs)

cnn.train_and_test(n_epoch=100, batchsize=128)

