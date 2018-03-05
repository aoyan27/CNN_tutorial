#!/usr/bin/env python
#coding:utf-8

import os
import numpy as np
import cv2 as cv
import itertools

class DataAugment:
    def __init__(self):
        self.data_dir_path = u"/home/amsl/my_CNN_tutorial/animeface-character-dataset/thumb/"
        
        self.big_data = None
        
        self.data_names = None
        
        self.target = None
        self.data = None
        self.HC_data = None
        self.LC_data = None

        self.Blur_data = None

        self.Gauss_data = None

        self.SP_data = None

        self.Hflip_data = None
        self.Vflip_data = None

        # contrast調整パラメータ
        self.min_table = 50
        self.max_table = 205
        self.gamma1 = 0.75
        self.gamma2 = 1.5

        self.LUT_HC = []
        self.LUT_LC = []

        # smoothingパラメータ
        self.one_side_size = 10
        
        # gauss noiseパラメータ
        self.mean = 0
        self.sigma = 40

        # salt and pepperパラメータ
        self.s_vs_p = 0.05
        self.amount = 0.004

    def make_lookuptable_gamma(self):
        for i in xrange(256):
            lut_g1[i] = 255 * pow(float(i) / 255, 1.0 / self.gamma1)
            lut_g2[i] = 255 * pow(float(i) / 255, 1.0 / self.gamma2)
        return lut_g1, lut_g2

    def make_lookuptable(self):
        diff_table = self.max_table - self.min_table
        lut_hc = np.arange(256, dtype=np.uint8)
        lut_lc = np.arange(256, dtype=np.uint8)

        for i in xrange(0, self.min_table):
            lut_hc[i] = 0
        for i in xrange(self.min_table, self.max_table):
            lut_hc[i] = 255 * (i - self.min_table) / diff_table
        for i in xrange(self.max_table, 255):
            lut_hc[i] = 255

        for i in xrange(256):
            lut_lc[i] = self.min_table + i * (diff_table) / 255
        
        return lut_hc, lut_lc


    def contrast_adjustment(self, images):
        hc_images = []
        lc_images = []
        for i in xrange(len(images)):
            hc_images.append(cv.LUT(images[i], self.LUT_HC))
            lc_images.append(cv.LUT(images[i], self.LUT_LC))
        
        return hc_images, lc_images
    

    def smoothing(self, images):
        blur_images = []
        average_square = (self.one_side_size, self.one_side_size)
        for i in xrange(len(images)):
            blur_images.append(cv.blur(images[i], average_square))
        
        return blur_images
            
    def gauss_noise(self, images):
        gauss_images = []
        for i in xrange(len(images)):
            row, col, ch = images[i].shape
            gauss = np.random.normal(self.mean, self.sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            gauss_images.append(images[i] + gauss)

        return gauss_images
    
    def salt_and_pepper(self, images):
        sp_images = []
        salt = self.s_vs_p
        pepper = (1.0 - self.s_vs_p)
        for i in xrange(len(images)):
            row, col, ch = images[i].shape
            sp_image = images[i].copy()

            for x, y in itertools.product(*map(range, (row, col))):
                r = np.random.random()
                if r < salt:
                    sp_image[x][y] = (255, 255, 255)
                elif r > pepper:
                    sp_image[x][y] = (0, 0, 0)

            sp_images.append(sp_image)

        return sp_images

    def flipping(self, images):
        hflip_images = []
        vflip_images = []
        for i in xrange(len(images)):
            hflip_images.append(cv.flip(images[i], 1))
            vflip_images.append(cv.flip(images[i], 0))

        return hflip_images, vflip_images


    
    def get_dir_list(self):
        tmp = os.listdir(self.data_dir_path)
        if tmp is None:
            return None
        return sorted([x for x in tmp if os.path.isdir(self.data_dir_path + x)])

    def read_images(self):
        dir_list = self.get_dir_list()
        images = []
        names = []
        for dir_name in dir_list:
            file_list = os.listdir(self.data_dir_path + dir_name)
            for file_name in file_list:
                root, ext = os.path.splitext(file_name)
                if ext == u".png" or ext == u".jpg":
                    abs_path = self.data_dir_path + dir_name + "/" + file_name
                    
                    names.append(abs_path)
                    
                    image = cv.imread(abs_path)
                    
                    images.append(image)

        return images, names

    def save_images(self, images, filter_name):
        for i in xrange(len(self.data_names)):
            root, ext = os.path.splitext(self.data_names[i])
            save_path = root + "_" + filter_name + ext
            print save_path
            cv.imwrite(save_path, images[i])
    
    def generate_data(self, data, target):
        self.data = []
        self.target = []
        #  self.data_names = []
        #  self.data, self.data_names = self.read_images()
        self.data = data
        self.target = target
        #  print self.data
        #  print self.data_names

        self.HC_data = []
        self.LC_data = []
        self.LUT_HC, self.LUT_LC = self.make_lookuptable()
        self.HC_data, self.LC_data = self.contrast_adjustment(self.data)

        self.Blur_data = []
        self.Blur_data = self.smoothing(self.data)

        self.Gauss_data = []
        self.Gauss_data = self.gauss_noise(self.data)

        self.SP_data = []
        self.SP_data = self.salt_and_pepper(self.data)

        self.Hflip_data = []
        self.Vflip_data = []
        self.Hflip_data, self.Vflip_data = self.flipping(self.data)

        all_big_data = self.data
        all_target = self.target

        all_big_data.extend(self.HC_data)
        all_target.extend(self.target)
        all_big_data.extend(self.LC_data)
        all_target.extend(self.target)

        all_big_data.extend(self.Blur_data)
        all_target.extend(self.target)

        all_big_data.extend(self.Gauss_data)
        all_target.extend(self.target)

        all_big_data.extend(self.SP_data)
        all_target.extend(self.target)

        all_big_data.extend(self.Hflip_data)
        all_target.extend(self.target)
        all_big_data.extend(self.Vflip_data)
        all_target.extend(self.target)

        return all_big_data, all_target


        
    def main(self):
        self.data = []
        self.data_names = []
        self.data, self.data_names = self.read_images()
        #  print self.data
        #  print self.data_names

        self.HC_data = []
        self.LC_data = []
        self.LUT_HC, self.LUT_LC = self.make_lookuptable()
        self.HC_data, self.LC_data = self.contrast_adjustment(self.data)

        self.Blur_data = []
        self.Blur_data = self.smoothing(self.data)

        self.Gauss_data = []
        self.Gauss_data = self.gauss_noise(self.data)

        self.SP_data = []
        self.SP_data = self.salt_and_pepper(self.data)

        self.Hflip_data = []
        self.Vflip_data = []
        self.Hflip_data, self.Vflip_data = self.flipping(self.data)
        

        self.big_data = self.data

        self.big_data.extend(self.HC_data)
        self.big_data.extend(self.LC_data)

        self.big_data.extend(self.Blur_data)
        
        self.big_data.extend(self.Gauss_data)

        self.big_data.extend(self.SP_data)

        self.big_data.extend(self.Hflip_data)
        self.big_data.extend(self.Vflip_data)

        self.save_images(self.HC_data, "hc")
        self.save_images(self.LC_data, "lc")
        self.save_images(self.Blur_data, "blur")
        self.save_images(self.Gauss_data, "gauss")
        self.save_images(self.SP_data, "sp")
        self.save_images(self.Hflip_data, "Hflip")
        self.save_images(self.Vflip_data, "Vflip")

        #  cv.namedWindow("Original")
        #  cv.namedWindow("High_contrast")
        #  cv.namedWindow("Low_contrast")
        #  cv.namedWindow("Blur")
        #  cv.namedWindow("Gauss")
        #  cv.namedWindow("Salt_and_Pepper")
        #  cv.namedWindow("Horizontal_flip")
        #  cv.namedWindow("Vertical_flip")
        #  cv.namedWindow("Big_data")
        #  cv.startWindowThread()

        #  for i in xrange(len(self.data)):
            #  cv.imshow("Original", self.data[i])
            #  cv.imshow("High_contrast", self.HC_data[i])
            #  cv.imshow("Low_contrast", self.LC_data[i])
            #  cv.imshow("Blur", self.Blur_data[i])
            #  cv.imshow("Gauss", self.Gauss_data[i])
            #  cv.imshow("Salt_and_Pepper", self.SP_data[i])
            #  cv.imshow("Horizontal_flip", self.Hflip_data[i])
            #  cv.imshow("Vertical_flip", self.Vflip_data[i])
            #  cv.waitKey(0)

        #  for i in xrange(len(self.big_data)):
            #  cv.imshow("Big_data", self.big_data[i])
            #  cv.waitKey(0)

if __name__=="__main__":
    test_data_augmentation = DataAugment()
    test_data_augmentation.main()
