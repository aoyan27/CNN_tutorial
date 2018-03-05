#!/usr/bin/env python
#coding:utf-8

import os
import six.moves.cPickle as pickle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class MyAnimeFaceDataset:
    def __init__(self):
        #  self.data_dir_path = u"/home/amsl/CNN_tutorial/animeface-character-dataset/thumb/000_hatsune_miku/"
        #  self.data_dir_path = u"/home/amsl/CNN_tutorial/animeface-character-dataset/thumb/136_shirley_fenette/"
        self.data_dir_path = u"/home/amsl/my_CNN_tutorial/animeface-character-dataset/predict/"
        self.data = None
        self.target = None
        self.n_types_target = -1
        self.image_size = 32
        #  self.image_size = 64

    def get_path_list(self):
        temp = os.listdir(self.data_dir_path)
        #  print temp
        if temp is None:
            return None
        #  for x in temp:
            #  print self.data_dir_path + x
            #  print os.path.isfile(self.data_dir_path + x)
        return sorted([x for x in temp if os.path.isfile(self.data_dir_path + x)])

    def load_dataset(self):
        if self.data is None:
            path_list = self.get_path_list()
            self.data = []

            perm_list = np.random.permutation(path_list)

            for file_name in perm_list:
                root, ext = os.path.splitext(file_name)
                if ext == u".png" or ext == u".jpg":
                    abs_path = self.data_dir_path + file_name
                    #  print "abs_path : ", abs_path
                    image = cv.imread(abs_path)
                    image = cv.resize(image, (self.image_size, self.image_size))
                    image = image.transpose(2, 0, 1)
                    image = image / 255.0
                    self.data.append(image)
                    break

        self.data = np.array(self.data, dtype=np.float32)

                    #  print image

                    #  cv.namedWindow("window")
                    #  cv.startWindowThread()
                    #  cv.imshow("window", image)
                    #  cv.waitKey(0)

                    #  plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                    #  plt.show()




if __name__=="__main__":
    dataset = MyAnimeFaceDataset()
    x = dataset.get_path_list()
    #  print x

    dataset.load_dataset()
    print dataset.data
