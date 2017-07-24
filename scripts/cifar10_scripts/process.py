#!/usr/bin/env python
#coding:utf-8

from CNN import CNN
from view_images_cifar10 import ViewCifar10
from chainer import cuda

import argparse

parser = argparse.ArgumentParser(description='process')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

if args.gpu >=0:
    cuda.get_device_from_id(args.gpu).use()

print "!!!!! Load AnimeFace Dataset !!!!!"

dataset = ViewCifar10()
dataset.process()
data = dataset.data
target = dataset.target
test_data = dataset.test_data
test_target = dataset.test_target
n_outputs = dataset.n_types_target
#  print "n_outputs : ", n_outputs
#  print "data : ", data

cnn = CNN(data=data, test_data=test_data, target=target, test_target=test_target, gpu=args.gpu, n_outputs=n_outputs)

cnn.train_and_test(n_epoch=100, batchsize=128)
#  cnn.train_and_test(n_epoch=3, batchsize=128)

