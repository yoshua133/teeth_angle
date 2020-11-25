BATCH_SIZE = 16
PROPOSAL_NUM = 6
CAT_NUM = 4
INPUT_SIZE = (448, 448)  # (w, h)
LR = 0.0001
WD = 1e-4
SAVE_FREQ = 1
resume = ''
test_model = 'model.ckpt'
save_dir = '/data2/xdw/teeth_model/resnet/'
file_dir = '/data2/xdw/NTS-Net/train_files/'
use_attribute = ['11','12','21','22']
need_attributes_idx = [4,5,6]
max_epoch = 3000
use_uniform_mean = '12'
#如果不是4个牙位一起算的话，use uniform要等于use_attribute