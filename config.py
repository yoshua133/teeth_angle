import os

BATCH_SIZE = 16
PROPOSAL_NUM = 6
CAT_NUM = 4
INPUT_SIZE = (224, 224)  # (w, h)
bigger = False

LR = 0.0001
WD = 1e-4
SAVE_FREQ = 1
resume = ''
use_attribute = ['11','12','21','22']

test_model = 'model.ckpt'
model_name = 'resnext101_32x8d'
model_size = '101'
pretrain = False

flip_prob = 0
loss_weight_mask_thres = -1

save_dir = '/data/shimr/model_save/' #保存模型的路径
file_dir = '/home/shimr/teeth_new/crowddet_teeth/teethcode_2021_jan30/train_files/'  #测试结果
file_dir_test = '/home/shimr/teeth_new/crowddet_teeth/teethcode_2021_jan30/test_files/'

#测试路径
load_model_path = os.path.join(save_dir, '20210423_155047part0_apr18_revised_crop1_725_aug_p_0.2_attri_9resnext101_32x8d_101pretrain-False','model_param.pkl')
anno_csv_path = "/home/shimr/teeth_new/crowddet_teeth/teethcode_2021_jan30/apr14_936_after_revise2_crop_1_725train.csv"#1_936_nov_18_725train_output.csv"
test_anno_csv_path = "/home/shimr/teeth_new/crowddet_teeth/teethcode_2021_jan30/apr14_936_after_revise2_crop_1_725train.csv"


##只改这里
use_part = 0#(比如part1-1)，对应need_attributes_idx_total中的第(use_part+1)行
use_gpu = '6' #str(use_part%8) 通过nvidia-smi命令查看空闲的gpu编号，一个编号是一张卡，不要一次占两张卡
need_attributes_idx_total = [[7,8,9],\
                              [32,33,34,35],\
                              [14,17,20,29,26,23],\
                              [15,18,21,30,27,24],\
                              [16,19,22],\
                              [31,28,25],
                              [10,11],
                             [12],
                             [13]]
save_name = 'part{}_apr30_revised_crop1_540_aug_p_{}_attri_{}'.format(use_part,flip_prob,need_attributes_idx_total[0][0])+ model_name+'_'+ model_size+"pretrain-"+str(pretrain)+"size"+str(INPUT_SIZE[0])
test_save_name = 'part{}_apr18_revised_crop1_725_aug_p_{}_attri_{}'.format(use_part,flip_prob,need_attributes_idx_total[0][0])+ model_name+'_'+ model_size+"pretrain-"+str(pretrain)

for i in range(len(need_attributes_idx_total)):
    for j in range(len(need_attributes_idx_total[i])):
        need_attributes_idx_total[i][j] -= 3
need_attributes_idx = need_attributes_idx_total[use_part]
max_epoch = 3000
use_uniform_mean = '12'

#如果不是4个牙位一起算的话，use uniform要等于use_attribute
