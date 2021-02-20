import os
import numpy as np
import shutil
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir,use_attribute, file_dir, max_epoch, need_attributes_idx,use_uniform_mean,anno_csv_path, use_gpu, save_name
from core import model, dataset,resnet
from core.utils import init_log, progress_bar
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = use_gpu
start_epoch = 0

save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S')+save_name)
file_dir = os.path.join(file_dir, datetime.now().strftime('%Y%m%d_%H%M%S')+save_name)
if not os.path.exists(save_dir):
  os.makedirs(save_dir)
creterion = torch.nn.L1Loss()

# read dataset
trainset = dataset.tooth_dataset_train(anno_path=anno_csv_path)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8, drop_last=False)
testset = dataset.tooth_dataset_test(anno_path=anno_csv_path)
testset.attributes_mean = trainset.attributes_mean
testset.attributes_std = trainset.attributes_std
print("test mean",testset.attributes_mean)
print("test std",testset.attributes_std)
testloader = torch.utils.data.DataLoader(testset)
# define model
num_of_need_attri = len(need_attributes_idx)
print("use attribute",need_attributes_idx)
net = resnet.resnet50(pretrained=False, num_fc=1, num_classes = num_of_need_attri )
if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1


# define optimizers
raw_parameters = list(net.parameters())


raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)

schedulers = [MultiStepLR(raw_optimizer, milestones=[160, 200], gamma=0.1)]
net = net.cuda()
net = DataParallel(net)

average_loss = [[111.1,111.1,111.1,111.1]]
head=['train_loss_unit_degree','train_ori_loss_unit_std','test_loss','test_ori_loss']




for epoch in range(start_epoch, max_epoch):
    for scheduler in schedulers:
        scheduler.step()

    # begin training
    print('--' * 50)
    net.train()
    train_num = 0
    train_loss = 0
    train_ori_loss = 0
    for i, data in enumerate(trainloader):
        img, target = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        print("batch size",batch_size)
        train_num += batch_size
        raw_optimizer.zero_grad()
        output = net(img)
        loss = creterion(output, target)
        ori_delta = (output-target).abs().cpu().detach().numpy()
        ori_delta_mean = ori_delta.mean()
        if train_num %100 ==0 and np.random.random()<0.1:
            print("target",target)
            print("outputs",output)
            print("loss",loss)
            print("unnorm delta",ori_delta * (trainset.attributes_std[use_uniform_mean]).reshape(-1))
            print("ori  delta",ori_delta)

        train_ori_loss += ori_delta_mean * batch_size
        unnorm_delta = ori_delta * (trainset.attributes_std[use_uniform_mean]).reshape(-1)
        train_loss += unnorm_delta.mean() * batch_size
        loss.backward()
        raw_optimizer.step()
        progress_bar(i, len(trainloader), 'train')

    if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    if not os.path.exists(file_dir):
            os.mkdir(file_dir)
    if epoch<1:
        shutil.copy( 'config.py', file_dir+'/config.py')
        shutil.copy( 'train.py', file_dir+'/train.py')
        shutil.copy( 'core/dataset.py', file_dir+'/dataset.py')
        shutil.copy( 'core/resnet.py', file_dir+'/resnet.py')
    if epoch % 5 == 0 or epoch==1:
        test_loss = 0
        test_ori_loss = 0
        test_num = 0
        net.eval()
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, target = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                #print('test batch size',batch_size)#bs=1
                test_num += batch_size
                raw_optimizer.zero_grad()
                output= net(img)
                # calculate loss
                #print("target",target)
                #print("target type",type(target))
                #print("outputs",output)
                #print("output type",type(output))
                #print("loss",loss)
                #loss = creterion(output, target)
                ori_delta = (output-target).abs().cpu().numpy()
                unnorm_delta = ori_delta * (trainset.attributes_std[use_uniform_mean]).reshape(-1)
                #loss is the mean distance between two tensor
                test_loss += unnorm_delta.mean()*batch_size
                test_ori_loss += ori_delta.mean()*batch_size
                # calculate accuracy
        print("epoch:{} mean loss, L1 gap divided by std".format(epoch),test_loss/test_num,"  ori loss ",\
          test_ori_loss/test_num)
        print("test_num",test_num)
        #train_ori_loss = trainset.attributes_std[use_uniform_mean][0]*train_loss.item()/train_num
        #test_ori_loss = trainset.attributes_std[use_uniform_mean][0]*test_loss.item()/test_num
        average_loss.append([train_loss/train_num, train_ori_loss/train_num, test_loss/test_num, test_ori_loss/test_num])
        if test_loss/test_num < average_loss[0][2]:
            average_loss[0] = [train_loss/train_num, train_ori_loss/train_num, test_loss/test_num, test_ori_loss/test_num]
        loss_csv=pd.DataFrame(columns=head,data=average_loss)
        loss_csv.to_csv(file_dir+'/{}_loss.csv'.format(save_name),encoding='gbk')
        f = open(file_dir+'/{}_mean.txt'.format(save_name),'w')
        f.write(str(trainset.attributes_mean))
        f.close()
        f2 = open(file_dir+'/{}_std.txt'.format(save_name),'w')
        f2.write(str(trainset.attributes_std))
        f2.close()

        net_state_dict = net.state_dict()
        torch.save(net_state_dict,save_dir+'/model_param.pkl')
