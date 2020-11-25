# coding=utf-8
import os
import pdb
from PIL import Image
import glob
import pandas as pd
from pandas import DataFrame
import csv
import cv2

images_dir = '/data2/xdw/ai_teeth_images/'
anno_path = '/data2/xdw/teeth_annotation_small.csv'
output_path = '/data2/xdw/teeth_annotation_small_output.csv'


def crop_image(in_path,out_path):
  img = cv2.imread(in_path)
  print(img.shape)
  cropped = img[100:707,673:1280,:] 
  #cropped = img[50:727,623:1280,:] 
  cv2.imwrite(out_path, cropped)


csv_num=0
r = csv.reader(open(anno_path))
#lines = [l for l in r]
#print(lines)
output_csv = []
for line in r:
    csv_num+=1
    if csv_num ==1 :#or csv_num>2:
        continue
    patient_id = str(line[0]).zfill(3)
    tooth_id = str(line[3])
    print("patient_id",patient_id)
    print("tooth_id",tooth_id)
    patient_id_path = images_dir+patient_id+'/'
    if not os.path.exists(patient_id_path):
        continue
    teeth_files = os.listdir(patient_id_path)
    existed = False
    
    for dir in teeth_files:
        if tooth_id in dir[4:7] and dir.endswith('tif') and (not 'crop' in dir):
            existed = True
            image_file = dir
            tooth_tif_path = patient_id_path+ dir
    if existed:       
        cropped_path = patient_id_path+"cropped_image"+image_file
        crop_image(tooth_tif_path,cropped_path)
        line[1] =  cropped_path
        output_csv.append(line)
    else:
        print("fail")

writer = csv.writer(open(output_path,'w'))
writer.writerows(output_csv)


