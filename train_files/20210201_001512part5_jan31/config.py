import os

BATCH_SIZE = 16
PROPOSAL_NUM = 6
CAT_NUM = 4
INPUT_SIZE = (448, 448)  # (w, h)
LR = 0.0001
WD = 1e-4
SAVE_FREQ = 1
resume = ''
test_model = 'model.ckpt'

save_dir = '/data/model_save/' #保存模型的路径
file_dir = '/home/xiangdawei/crowddet_teeth/teethcode_2021_jan30/train_files/'  #测试结果
file_dir_test = '/home/xiangdawei/crowddet_teeth/teethcode_2021_jan30/test_files/'

#测试路径
load_model_path = os.path.join(save_dir, '20201226_232734part3_dec26_test0.2','model_param.pkl')
anno_csv_path = "/home/xiangdawei/crowddet_teeth/teethcode_2021_jan30/1_936_nov_18_output.csv"
test_anno_csv_path = anno_csv_path
use_attribute = ['11','12','21','22']

##只改这里
use_part = 5#(比如part1-1)
use_gpu = '0' #str(use_part%8)
save_name = 'part{}_jan31'.format(use_part)
need_attributes_idx_total = [[7,8,9],\
                              [32,33,34,35],\
                              [14,17,20,29,26,23],\
                              [15,18,21,30,27,24],\
                              [16,19,22],\
                              [31,28,25],
                              [10,11],
                             [12],
                             [13]]

for i in range(len(need_attributes_idx_total)):
    for j in range(len(need_attributes_idx_total[i])):
        need_attributes_idx_total[i][j] -= 3
need_attributes_idx = need_attributes_idx_total[use_part]
max_epoch = 3000
use_uniform_mean = '12'

#如果不是4个牙位一起算的话，use uniform要等于use_attribute
