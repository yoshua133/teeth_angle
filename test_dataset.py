# coding=gbk
import os
import shutil
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir,use_attribute, file_dir_test, max_epoch, need_attributes_idx,use_uniform_mean,test_anno_csv_path, use_gpu, load_model_path,save_name
from core import model, dataset,resnet
from core.utils import init_log, progress_bar
import pandas as pd
from IPython import embed
import time


os.environ['CUDA_VISIBLE_DEVICES'] = use_gpu
start_epoch = 0


file_dir_test = os.path.join(file_dir_test, save_name)
if not os.path.exists(file_dir_test):
    os.makedirs(file_dir_test)



# read dataset
testset = dataset.tooth_dataset_test(anno_path=test_anno_csv_path)
testloader = torch.utils.data.DataLoader(testset)
# define model
num_of_need_attri = len(need_attributes_idx)
net = resnet.resnet50(pretrained=False, num_fc=1, num_classes = num_of_need_attri )
if load_model_path:
    ckpt = torch.load(load_model_path)
    for name in list(ckpt.keys()):
        ckpt[name.replace('module.','')] = ckpt[name]
        del ckpt[name]
    net.load_state_dict(ckpt)
    


# define optimizers
raw_parameters = list(net.parameters())



net = net.cuda()
net = DataParallel(net)


average_loss = [[111.1,111.1,111.1,111.1]]
head=['cur_use_attri','teeth_place']
for pre_name in ['target','output']:
    for attr_id in need_attributes_idx:
        head.append(pre_name+'_'+str(attr_id))
print(head)
#save_name =  'part6_dec4'#str(datetime.now().strftime('%Y%m%d_%H%M%S')) 

test_loss = 0
test_ori_loss = 0 
test_num = 0 
net.eval()
output_csv = []
total_time = 0

for i, data in enumerate(testloader):
    with torch.no_grad():
        img, target = data[0].cuda(), data[1].cuda()    
        cur_use_attri, index = data[2],data[3]
        #embed()         
        batch_size = img.size(0)
        #print('test batch size',batch_size)#bs=1
        test_num += batch_size
        start = time.time()
        output= net(img)
        end = time.time()
        total_time += (end-start)
        target_unnorm = (target.cpu().numpy()* testset.attributes_std[use_uniform_mean])+testset.attributes_mean[use_uniform_mean]
        output_unnorm = (output.cpu().numpy()* testset.attributes_std[use_uniform_mean])+testset.attributes_mean[use_uniform_mean]
        cur_row =[]
        cur_row.append(str(cur_use_attri[0]))#.item()))
        cur_row.append(str(index))
        print('target_unnorm',target_unnorm)
        print('output_unnorm',output_unnorm)
        for tar in target_unnorm.reshape(-1):
            #print('t',tar)
            cur_row.append(str(tar))
        for out in output_unnorm.reshape(-1) :
            cur_row.append(str(out))
       
        ori_delta = (output-target).abs().cpu().numpy()
        unnorm_delta = ori_delta * testset.attributes_std[use_uniform_mean]
        
        output_csv.append(cur_row)
        
output_csv.insert(0,[str(total_time),str(test_num),str(total_time/test_num)])
loss_csv=pd.DataFrame(columns=head,data=output_csv)
loss_csv.to_csv(file_dir_test+'/{}_test_dataset.csv'.format(save_name),encoding='gbk')
