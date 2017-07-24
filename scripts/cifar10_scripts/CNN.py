#!/usr/bin/env python
#coding:utf-8

import time
import numpy as np
import six.moves.cPickle as pickle
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

import matplotlib.pyplot as plt

# alex net
#  class ImageNet(Chain):
    #  def __init__(self, n_outputs):
        #  super(ImageNet, self).__init__(
            #  conv1 = L.Convolution2D(3, 96, 11, stride=4),
            #  conv2 = L.Convolution2D(96, 256,  5, pad=2),
            #  conv3 = L.Convolution2D(256, 384,  3, pad=1),
            #  conv4 = L.Convolution2D(384, 384,  3, pad=1),
            #  conv5 = L.Convolution2D(384, 256,  3, pad=1),
            #  fc6 = L.Linear(256, 4096),
            #  fc7 = L.Linear(4096, 4096),
            #  fc8 = L.Linear(4096, 1000),
        #  )

    #  def forward(self, x_data, y_data, gpu=-1):
        #  if gpu >= 0:
            #  x_data = cuda.to_gpu(x_data)
            #  y_data = cuda.to_gpu(y_data)

        #  x, t = Variable(x_data), Variable(y_data)
        #  h = F.max_pooling_2d(F.local_response_normalization(
            #  F.relu(self.conv1(x))), 3, stride=2)
        #  h = F.max_pooling_2d(F.local_response_normalization(
            #  F.relu(self.conv2(h))), 3, stride=2)
        #  h = F.relu(self.conv3(h))
        #  h = F.relu(self.conv4(h))
        #  h = F.max_pooling_2d(F.relu(self.conv5(h)), 2, stride=2)
        #  h = F.dropout(F.relu(self.fc6(h)))
        #  h = F.dropout(F.relu(self.fc7(h)))
        #  y = self.fc8(h)

        #  return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    
    #  def predict(self, x_data, gpu=-1):
        #  if gpu >= 0:
            #  x_data = cuda.to_gpu(x_data)
        #  x = Variable(x_data)
        #  h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        #  h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2, stride=2)
        #  h = F.max_pooling_2d(F.relu(self.conv3(h)), ksize=2, stride=2)
        #  h = F.max_pooling_2d(F.relu(self.conv4(h)), ksize=2, stride=2)
        #  h = F.dropout(F.relu(self.l3(h)))
        #  y = self.l4(h)
        
        #  return np.argmax(F.softmax(y).data, axis=1)

# cifar10
#  class ImageNet(Chain):
    #  def __init__(self, n_outputs):
        #  super(ImageNet, self).__init__(
                #  conv1 = L.Convolution2D(3, 32, 5, stride=1, pad=2),
                #  conv2 = L.Convolution2D(32, 32, 5, stride=1, pad=2),
                #  conv3 = L.Convolution2D(32, 64, 5, stride=1, pad=2),
                #  l4 = L.Linear(1344, 4096),
                #  l5 = L.Linear(4096, n_outputs),
        #  )

    #  def forward(self, x_data, y_data, gpu=-1):
        #  if gpu >= 0:
            #  x_data = cuda.to_gpu(x_data)
            #  y_data = cuda.to_gpu(y_data)

        #  x, t = Variable(x_data), Variable(y_data)
        #  #  print "x : ", x.data
        #  #  print "t : ", t.data
        #  h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=3, stride=2)
        #  #  print "h1 : ", h.data
        #  h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=2)
        #  #  print "h2 : ", h.data
        #  h = F.relu(self.conv3(h))
        #  #  print "h3 : ", h.data
        #  h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        #  #  print "h4 : ", h.data
        #  #  print chainer.config.train
        #  h = F.dropout(F.relu(self.l4(h)))
        #  #  print "h4 : ", h.data
        #  y = self.l5(h)
        #  #  print "y.data : ", y.data

        #  return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    #  def predict(self, x_data, gpu=-1):
        #  if gpu >= 0:
            #  x_data = cuda.to_gpu(x_data)
        #  x = Variable(x_data)
        #  h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=3, stride=2)
        #  h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=2)
        #  h = F.relu(self.conv3(h))
        #  h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        #  h = F.dropout(F.relu(self.l4(h)))
        #  y = self.l5(h)

        #  return np.argmax(F.softmax(y).data, axis=1)

# with batchmormalization
#  class ImageNet(Chain):
    #  def __init__(self, n_outputs):
        #  super(ImageNet, self).__init__(
                #  conv1 = L.Convolution2D(3, 32, 5, stride=1, pad=2),
                #  bn1 = L.BatchNormalization(32),
                #  conv2 = L.Convolution2D(32, 64, 5, stride=1, pad=2),
                #  bn2 = L.BatchNormalization(64),
                #  conv3 = L.Convolution2D(64, 128, 5, stride=1, pad=2),
                #  bn3 = L.BatchNormalization(128),
                #  l4 = L.Linear(4*4*128, 1024),
                #  l5 = L.Linear(1024, n_outputs),
        #  )

    #  def forward(self, x_data, y_data, gpu=-1):
        #  if gpu >= 0:
            #  x_data = cuda.to_gpu(x_data)
            #  y_data = cuda.to_gpu(y_data)

        #  x, t = Variable(x_data), Variable(y_data)
        #  #  print "x : ", x.data
        #  #  print "t : ", t.data
        #  h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), ksize=3, stride=2)
        #  #  print "h1 : ", h.data
        #  h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), ksize=3, stride=2)
        #  #  print "h2 : ", h.data
        #  h = F.max_pooling_2d(F.relu(self.bn3(self.conv3(h))), ksize=3, stride=2)
        #  #  print "h3 : ", h.data
        #  h = F.dropout(F.relu(self.l4(F.dropout(h))))
        #  #  print "h4 : ", h.data
        #  y = self.l5(h)
        #  #  print "y.data : ", y.data

        #  return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    #  def predict(self, x_data, gpu=-1):
        #  if gpu >= 0:
            #  x_data = cuda.to_gpu(x_data)
        #  x = Variable(x_data)
        #  h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=3, stride=2)
        #  h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=2)
        #  h = F.relu(self.conv3(h))
        #  h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        #  h = F.dropout(F.relu(self.l4(h)))
        #  y = self.l5(h)

        #  return np.argmax(F.softmax(y).data, axis=1)

# VGG
class ImageNet(Chain):
    def __init__(self, n_outputs):
        super(ImageNet, self).__init__(
                conv1_1=L.Convolution2D(3, 64, 3, pad=1),
                bn1_1=L.BatchNormalization(64),
                conv1_2=L.Convolution2D(64, 64, 3, pad=1),
                bn1_2=L.BatchNormalization(64),

                conv2_1=L.Convolution2D(64, 128, 3, pad=1),
                bn2_1=L.BatchNormalization(128),
                conv2_2=L.Convolution2D(128, 128, 3, pad=1),
                bn2_2=L.BatchNormalization(128),

                conv3_1=L.Convolution2D(128, 256, 3, pad=1),
                bn3_1=L.BatchNormalization(256),
                conv3_2=L.Convolution2D(256, 256, 3, pad=1),
                bn3_2=L.BatchNormalization(256),
                conv3_3=L.Convolution2D(256, 256, 3, pad=1),
                bn3_3=L.BatchNormalization(256),
                conv3_4=L.Convolution2D(256, 256, 3, pad=1),
                bn3_4=L.BatchNormalization(256),

                fc4=L.Linear(4096, 1024),
                fc5=L.Linear(1024, 1024),
                fc6=L.Linear(1024, 10),
        )

    def forward(self, x_data, y_data, gpu=-1):
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)

        x, t = Variable(x_data), Variable(y_data)
        #  print "x : ", x.data
        #  print "t : ", t.data
        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.relu(self.bn3_4(self.conv3_4(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5)
        h = F.dropout(F.relu(self.fc5(h)), ratio=0.5)
        y = self.fc6(h)

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def predict(self, x_data, gpu=-1):
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
        x = Variable(x_data)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = F.dropout(F.relu(self.l4(h)))
        y = self.l5(h)

        return np.argmax(F.softmax(y).data, axis=1)

class CNN:
    def __init__(self, data, test_data, target, test_target, n_outputs, gpu=-1):
        self.model = ImageNet(n_outputs)
        self.model_name = 'cnn_model_Cifar10'

        if gpu >= 0:
            self.model.to_gpu()

        self.gpu = gpu

        self.x_train = data
        self.x_test = test_data
        self.y_train = target
        self.y_test = test_target

        self.n_train = len(self.x_train)
        self.n_test = len(self.x_test)

        #  self.optimizer = optimizers.Adam()
        #  self.optimizer = optimizers.AdaGrad(lr=0.01)
        #  self.optimizer = optimizers.AdaDelta()
        self.optimizer = optimizers.RMSpropGraves(lr=0.0001, alpha=0.95)
        #  self.optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
        #  self.optimizer = optimizers.SGD(lr=0.1)
        self.optimizer.setup(self.model)

    def predict(self, x_data, gpu=-1):
        with chainer.using_config('train', False):
            return self.model.predict(x_data, gpu)


    def train_and_test(self, n_epoch=100, batchsize=100):
        epoch = 1
        best_accuracy = 0
        
        x_plot = np.array([], dtype=np.int32)
        y_plot_loss = np.array([], dtype=np.float32)
        y_plot_acc = np.array([], dtype=np.float32)

        while epoch <= n_epoch:
            print 'epoch :', epoch
            x_plot = np.append(x_plot, [epoch])
            # training
            with chainer.using_config('train', True):
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
                sum_test_accuracy = 0
                sum_test_loss = 0

                for i in xrange(0, self.n_test, batchsize):
                    x_batch = self.x_test[i:i+batchsize]
                    y_batch = self.y_test[i:i+batchsize]

                    real_batchsize = len(x_batch)

                    #  self.model.zerograds()
                    loss, acc = self.model.forward(x_batch, y_batch, gpu=self.gpu)
                    #  loss.backward()
                    #  self.optimizer.update()

                    sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                    sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

                print "test mean loss={}, accuracy={}".format(sum_test_loss/self.n_test, sum_test_accuracy/self.n_test)
                y_plot_loss = np.append(y_plot_loss, [sum_test_loss/self.n_test])
                y_plot_acc = np.append(y_plot_acc, [sum_test_accuracy/self.n_test])


            epoch += 1

        print "!!!!!!!!!!!!!!! Save model !!!!!!!!!!!!!!"
        self.dump_model()
        
        plt.subplot(2, 1, 1)
        plt.plot(x_plot, y_plot_loss)
        plt.title("Loss curve")
        plt.ylim(0, 2.0)

        plt.subplot(2, 1, 2)
        plt.plot(x_plot, y_plot_acc)
        plt.title("Accuracy curve")
        plt.ylim(0, 1.0)

        plt.show()


    def dump_model(self):
        self.model.to_cpu()
        serializers.save_npz(self.model_name, self.model)

    def load_model(self):
        print "!!! Load model !!!"
        serializers.load_npz(self.model_name, self.model)
        if self.gpu >= 0:
            self.model.to_gpu()
