# coding=utf-8
import os
import pdb
from PIL import Image
import glob
import pandas as pd
from pandas import DataFrame
import csv
import cv2
import random
import numpy as np
import os

images_dir = '/data/shimr/teeth/'
anno_path = '/home/shimr/teeth_new/1_936_nov_18.csv'
output_path = '/home/shimr/teeth_new/crowddet_teeth/teethcode_2021_jan30/1_936_mar_20_crop_0_725train.csv'


def crop_image(in_path,out_path):
  #print(in_path)
  img = cv2.imread(in_path)
  #print(img.shape)
  cropped = img[100:707,673:1280,:]#[310:497,719:919,:]#[310:497,719:1023,:]#[100:707,673:1280,:]
  #cropped = img[50:727,623:1280,:]
  cv2.imwrite(out_path, cropped)

np.random.seed(0)

csv_num=0
r = csv.reader(open(anno_path))
#lines = [l for l in r]
#print(lines)
teeth_row = 3
patient_row = 0
output_csv = []
former_id = -1
last_flag = 'unknown'
prefix = "cropped_image_a"
for line in r:
    csv_num+=1
    if csv_num ==1 :#or csv_num>2:
        continue
    patient_id = str(line[patient_row]).zfill(3)
    tooth_id = str(line[teeth_row])
    print("patient_id",patient_id)
    #print("tooth_id",tooth_id)
    patient_id_path = os.path.join(images_dir,patient_id)
    if not os.path.exists(patient_id_path):
        continue
    teeth_files = os.listdir(patient_id_path)
    existed = False

    for dir in teeth_files:    #if tooth_id in dir[4:7] and dir.endswith('tif') and (not 'crop' in dir):
         #print(dir)
         #print(patient_id_path)
         if not ',' in dir:
            #print("no ,")
            continue
         if tooth_id in dir.split(',')[1] and 'tif' in dir and (not 'crop' in dir):
            existed = True
            image_file = dir
            tooth_tif_path = os.path.join(patient_id_path, dir)
    if existed:
        cropped_path = os.path.join(patient_id_path, prefix+image_file)
        #crop_image(tooth_tif_path,cropped_path)
        line[1] =  cropped_path
        ran = np.random.rand(1)
        if former_id == patient_id:
            line[2] = last_flag
            former_id = patient_id
        elif ran <= 0.8:
          line[2] = 'train'
          last_flag = 'train'
          former_id = patient_id
        elif ran > 0.8:
          line[2] = 'test'
          last_flag = 'test'
          former_id = patient_id
        else:
          line[2] = 'val'
          last_flag = 'val'
          former_id = patient_id

        crop_image(tooth_tif_path,cropped_path)
        output_csv.append(line)
    else:
        print("fail")

writer = csv.writer(open(output_path,'w'))
writer.writerows(output_csv)


