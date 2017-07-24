#!/usr/bin/env python
#coding:utf-8

from CNN import CNN
from animeface import AnimeFaceDataset
from chainer import cuda

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

cnn = CNN(data=data, target=target, gpu=args.gpu, n_outputs=n_outputs)

cnn.train_and_test(n_epoch=1000, batchsize=128)

