#!/usr/bin/env python
#coding:utf-8

import os
import six.moves.cPickle as pickle
import numpy as np
import cv2 as cv

from chainer import cuda

cuda.Device(0).use()
xp = cuda.cupy

class MyDataset:
    def __init__(self):
        self.data_dir_path = u"/home/amsl/CNN_tutorial/256_ObjectCategories/"
        self.data = None
        self.target = None
        self.n_types_target = 0
        #  self.image_size = 128
        self.image_size = 256
        #  self.image_size = 255

    
    def get_dir_list(self):
        temp = os.listdir(self.data_dir_path)
        #  print temp
        if temp is None:
            return None
        #  for x in temp:
            #  print self.data_dir_path + x
            #  print os.path.isfile(self.data_dir_path + x)
        return sorted([x for x in temp if os.path.isdir(self.data_dir_path + x)])

    def get_class_id(self, file_name):
        dir_list = self.get_dir_list()
        #  print dir_list
        #  myfanc = lambda x: x in file_name
        #  print myfanc(dir_list[1])
        dir_name = filter(lambda x: x in file_name, dir_list)
        return dir_list.index(dir_name[0])


    def load_data_target(self):
        if self.target is None:
            dir_list = self.get_dir_list()
            self.n_types_target = len(dir_list)
            self.target = []
            target_name = []
            #  self.data = xp.array([])
            self.data = []
            for dir_name in dir_list:
                file_list = os.listdir(self.data_dir_path + dir_name)
                for file_name in file_list:
                    root, ext = os.path.splitext(file_name)
                    if ext == u".png" or ext == u".jpg":
                        abs_name = self.data_dir_path + dir_name + "/" + file_name
                        #  print "!!!!! Data Loading !!!!!!"
                        print abs_name

                        class_id = self.get_class_id(abs_name)
                        self.target.append(class_id)
                        target_name.append(str(dir_name))
                        
                        image = cv.imread(abs_name)
                        image = cv.resize(image, (self.image_size, self.image_size))
                        image = image.transpose(2, 0, 1)
                        image = image / 255.0
                        self.data.append(image)

            self.data = np.array(self.data, np.float32)
            self.target = np.array(self.target, dtype=np.int32)

                        #  cv.namedWindow("window")
                        #  cv.startWindowThread()
                        #  cv.imshow("window", image)
                        #  cv.waitKey(0)

        


if __name__=="__main__":
    dataset = MyDataset()
    #  print dataset.get_class_id("/home/amsl/CNN_tutorial/256_ObjectCategories/001.ak47/001_0001.jpg")
    
    dataset.load_data_target()
    print dataset.data

