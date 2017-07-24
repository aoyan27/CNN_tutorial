#!/usr/bin/env python
#coding:utf-8

import time
import numpy as np
import six.moves.cPickle as pickle
from sklearn.datasets import fetch_mldata
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class ImageNet(Chain):
    def __init__(self, n_outputs):
        super(ImageNet, self).__init__(
            conv1 = L.Convolution2D(3, 96, 3),
            conv2 = L.Convolution2D(96, 128, 3),
            conv3 = L.Convolution2D(128, 256, 3),
            conv4 = L.Convolution2D(256, 256, 3),
            l3 = L.Linear(256, 1024),
            l4 = L.Linear(1024, n_outputs)
            #  conv1 = L.Convolution2D(3, 32, 5),
            #  conv2 = L.Convolution2D(32, 32, 5),
            #  l3 = L.Linear(2592, 2592),
            #  l4 = L.Linear(2592, n_outputs)

        )

    def forward(self, x_data, y_data, gpu=-1):
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)

        x, t = Variable(x_data), Variable(y_data)
        #  print "x : ", x.data
        #  print "t : ", t.data
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        #  h = F.relu(self.conv1(x))
        #  print "h1 : ", h.data.shape
        #  h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=3)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), ksize=2, stride=2)
        #  h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), ksize=2, stride=2)
        #  print "h2 : ", h2.data
        h = F.dropout(F.relu(self.l3(h)))
        #  print "h3_ : ", h3.data
        y = self.l4(h)
        #  print "y.data : ", y.data
        


        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    
    def predict(self, x_data, gpu=-1):
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
        x = Variable(x_data)
        #  h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        #  h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=3)
        #  h = F.dropout(F.relu(self.l3(h)))
        #  y = self.l4(h)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv4(h)), ksize=2, stride=2)
        h = F.dropout(F.relu(self.l3(h)))
        y = self.l4(h)
        
        return np.argmax(F.softmax(y).data, axis=1)


class CNN:
    def __init__(self, data_train, data_test, target_train, target_test, n_outputs, gpu=-1):
        self.model = ImageNet(n_outputs)
        self.model_name = 'cnn_model'

        if gpu >= 0:
            self.model.to_gpu()

        self.gpu = gpu

        self.x_train = data_train
        self.x_test = data_test
        self.y_train = target_train
        self.y_test = target_test

        self.n_train = len(self.x_train)
        self.n_test = len(self.x_test)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def predict(self, x_data, gpu=-1):
        with chainer.using_config('train', False):
            return self.model.predict(x_data, gpu)


    def train_and_test(self, n_epoch=100, batchsize=100):
        epoch = 1
        best_accuracy = 0
        while epoch <= n_epoch:
            print 'epoch :', epoch
            
            # training
            with chainer.using_config('train', True):
                #  print chainer.configuration.config.train
                perm =np.random.permutation(self.n_train)
                sum_train_accuracy = 0
                sum_train_loss = 0

                for i in xrange(0, self.n_train, batchsize):
                    x_batch = self.x_train[perm[i:i+batchsize]]
                    y_batch = self.y_train[perm[i:i+batchsize]]

                    real_batchsize = len(x_batch)

                    self.model.cleargrads()
                    loss, acc = self.model.forward(x_batch, y_batch, gpu=self.gpu)
                    loss.backward()
                    self.optimizer.update()

                    sum_train_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                    sum_train_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

                print "train mean loss={}, accuracy={}".format(sum_train_loss/self.n_train, sum_train_accuracy/self.n_train)
                
            
            # evoluation
            with chainer.using_config('train', False):
                #  print chainer.configuration.config.train
                sum_test_accuracy = 0
                sum_test_loss = 0

                #  perm =np.random.permutation(self.n_train)
                for i in xrange(0, self.n_test, batchsize):
                #  for i in xrange(0, self.n_train, batchsize):
                    x_batch = self.x_test[i:i+batchsize]
                    y_batch = self.y_test[i:i+batchsize]

                    #  x_batch = self.x_train[perm[i:i+batchsize]]
                    #  y_batch = self.y_train[perm[i:i+batchsize]]

                    real_batchsize = len(x_batch)

                    loss, acc = self.model.forward(x_batch, y_batch, gpu=self.gpu)
                    #  loss.backward()
                    #  self.optimizer.update()

                    sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                    sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

                print "test mean loss={}, accuracy={}".format(sum_test_loss/self.n_test, sum_test_accuracy/self.n_test)
                #  print "test mean loss={}, accuracy={}".format(sum_test_loss/self.n_train, sum_test_accuracy/self.n_train)

            epoch += 1

        print "!!!!!!!!!!!!!!! Save model !!!!!!!!!!!!!!"
        self.dump_model()



    def dump_model(self):
        self.model.to_cpu()
        serializers.save_npz(self.model_name, self.model)

    def load_model(self):
        serializers.load_npz(self.model_name, self.model)
        if self.gpu >= 0:
            self.model.to_gpu()
