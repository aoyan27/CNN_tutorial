#!/usr/bin/env python
#coding:utf-8

from my_CNN import CNN
from animeface import AnimeFaceDataset
from my_data_augmentation import DataAugment
from chainer import cuda
from sklearn.cross_validation import train_test_split

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

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.1)
#  print "n_outputs : ", n_outputs
#  print "data : ", data
#  print len(data)
#  print len(data_train)
#  print len(target_train)

#  print len(data_test)
#  print len(target_test)

data_augment = DataAugment()
datas, targets = data_augment.generate_data(data, target_train)

#  cnn = CNN(data_train=data_train, data_test=data_test, target_train=target_train, target_test=target_test, n_outputs=n_outputs, gpu=args.gpu)

#  cnn.train_and_test(n_epoch=100, batchsize=128)

