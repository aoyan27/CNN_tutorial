#!/usr/bin/env python
#coding:utf-8

import os
import six.moves.cPickle as pickle
import numpy as np
import cv2 as cv

class AnimeFaceDataset:
    def __init__(self):
        #  self.data_dir_path = u"/home/amsl/CNN_tutorial/animeface-character-dataset/thumb/"
        self.data_dir_path = u"/home/amsl/my_CNN_tutorial/animeface-character-dataset/thumb/"
        self.data = None
        self.target = None
        self.n_types_target = -1
        self.dump_name = u"dataset"
        self.image_size = 32
        #  self.image_size = 64
        #  self.image_size = 128

    def get_dir_list(self):
        tmp = os.listdir(self.data_dir_path)
        if tmp is None:
            return None
        return sorted([x for x in tmp if os.path.isdir(self.data_dir_path+x)])

    def get_class_id(self, fname):
        dir_list = self.get_dir_list()
        dir_name = filter(lambda x: x in fname, dir_list)
        return dir_list.index(dir_name[0])

    def load_data_target(self):
        if os.path.exists(self.dump_name):
            self.load_dataset()
        if self.target is None:
            dir_list = self.get_dir_list()
            ret = {}
            self.target = []
            target_name = []
            self.data = []
            for dir_name in dir_list:
                file_list = os.listdir(self.data_dir_path + dir_name)
                for file_name in file_list:
                    root, ext = os.path.splitext(file_name)
                    if ext == u".png":
                        abs_name = self.data_dir_path + dir_name + "/" + file_name
                        print abs_name
                        
                        class_id = self.get_class_id(abs_name)
                        self.target.append(class_id)
                        target_name.append(str(dir_name))
                        
                        image = cv.imread(abs_name)
                        image = cv.resize(image, (self.image_size, self.image_size))
                        image = image.transpose(2, 0, 1)
                        image = image / 255.0
                        self.data.append(image)

            self.index2name = {}
            for i in xrange(len(self.target)):
                self.index2name[self.target[i]] = target_name[i]

        print self.index2name
        self.data = np.array(self.data, np.float32)
        self.target = np.array(self.target, np.int32)

        #  self.dump_dataset()

    def get_n_types_target(self):
        if self.target is None:
            self.load_data_target()

        if self.n_types_target is not -1:
            return self.n_types_target

        tmp = {}
        for target in self.target:
            tmp[target] = 0
        return len(tmp)

    def dump_dataset(self):
        pickle.dump((self.data, self.target, self.index2name), open(self.dump_name, "wb"), -1)

    def load_dataset(self):
        self.data, self.target, self.index2name = pickle.load(open(self.dump_name, "rb"))


if __name__=="__main__":
    dataset = AnimeFaceDataset()
    dataset.load_data_target()
    print dataset.get_n_types_target()

    print len(dataset.target)
