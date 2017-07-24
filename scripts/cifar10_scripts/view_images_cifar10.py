#!/usr/bin/env python
#coding:utf-8

import os
import six.moves.cPickle as pickle
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

class ViewCifar10:
    def __init__(self):
        self.path_dir_name = u"/home/amsl/CNN_tutorial/cifar-10-batches-py/"
        self.tareget = None
        self.data = None
        self.test_tareget = None
        self.test_data = None
        self.n_types_target = 0

    def unpickle(self, fname):
        fo = open(fname, 'rb')
        d = pickle.load(fo)
        fo.close()

        return d
    
    def make_images(self, datas, test=False):
        for i in xrange(len(datas)):
            #  image = datas[i].reshape(3, 32, 32).transpose(1, 2, 0)
            image = datas[i].reshape(3, 32, 32)
            #  image = cv.resize(image, (96, 96))
            image = image.astype(np.float32)
            image *= (1.0 / 255.0)
            if test:
                self.test_data.append(image)
            else:
                self.data.append(image)

    def meaning(self, images):
        N = len(images)
        sum_image = 0
        for i in xrange(N):
            sum_image += images[i]
        return sum_image / N

    def whiting(self, images):
        white_images = []
        for i in xrange(len(images)):
            img = images[i].copy()
            #  print img.shape
            d, w, h = img.shape
            num_pixels = d * w * h
            mean = img.mean()
            variance = np.mean(np.square(img)) - np.square(mean)
            stddev = np.sqrt(variance)
            min_stddev = 1.0 / np.sqrt(num_pixels)
            scale = stddev if stddev > min_stddev else min_stddev
            img -= mean
            img /= scale

            white_images.append(img)
            
        return white_images

    def pre_process(self, images, mean_image):
        for i in xrange(len(images)):
            images[i] -= mean_image
        return images
        

    def process(self):
        abs_name_meta = self.path_dir_name + "batches.meta"
        abs_name = []
        abs_name.append(self.path_dir_name + "data_batch_1")
        abs_name.append(self.path_dir_name + "data_batch_2")
        abs_name.append(self.path_dir_name + "data_batch_3")
        abs_name.append(self.path_dir_name + "data_batch_4")
        abs_name.append(self.path_dir_name + "data_batch_5")
        
        test_abs_name = []
        test_abs_name.append(self.path_dir_name + "test_batch")
        #  print abs_name
        
        d_meta = self.unpickle(abs_name_meta)
        #  print d_meta
        label_names = d_meta["label_names"]
        self.n_types_target = len(label_names)
        #  print self.n_types_target
        
        self.data = []
        self.target = []
        self.test_data = []
        self.test_target = []

        for i in xrange(5):
            d = self.unpickle(abs_name[i])
            datas = d["data"]
            labels = d["labels"]
            filenames = d["filenames"]
            #  print data
            #  print len(data)
            #  print len(data[0])
            self.make_images(datas)
            self.target.extend(labels)
            #  print len(images)
            #  print images[0].shape
        mean_image = self.meaning(self.data)
        self.data = self.whiting(self.data)

        self.data = np.array(self.data, dtype=np.float32)
        #  self.data = self.pre_process(self.data, mean_image)
        self.target = np.array(self.target, dtype=np.int32)


        for i in xrange(1):
            d_test = self.unpickle(test_abs_name[i])
            datas = d_test["data"]
            labels = d_test["labels"]
            filenames = d_test["filenames"]

            self.make_images(datas, test=True)
            self.test_target.extend(labels)

        self.test_data = self.whiting(self.test_data)
        self.test_data = np.array(self.test_data, dtype=np.float32)
        #  self.test_data = self.pre_process(self.test_data, mean_image)
        self.test_target = np.array(self.test_target, dtype=np.int32)

        #  print labels[0]
        #  print filenames[0]

        #  cv.namedWindow("image")
        #  cv.startWindowThread()
        #  #  cv.imshow("image", images[0])
        #  cv.imshow("image", mean_image)
        #  cv.waitKey(0)
        
        #  image = self.data[0]
        #  image = image.transpose(1, 2, 0)
        
        #  mean_image = mean_image.transpose(1, 2, 0)

        #  white_image = white_images[0].transpose(1, 2, 0)


        #  plt.imshow(image)
        #  plt.imshow(mean_image)
        #  plt.imshow(white_image)
        #  plt.show()

if __name__=="__main__":
    view_cifar = ViewCifar10()
    view_cifar.process()
    #  print view_cifar.data
    print len(view_cifar.data)
    print len(view_cifar.target)
    print len(view_cifar.test_data)
    print len(view_cifar.test_target)
    print view_cifar.n_types_target
