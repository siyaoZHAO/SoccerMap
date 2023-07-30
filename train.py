import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import network
import torch.nn as nn
import time
from torch.utils.data import Subset, TensorDataset
from torchvision import transforms

class Trainer:
    def __init__(self, arg):
        # 其他参数
        self.arg = arg
        # 训练参数
        self.batch_size = arg['batch_size']
        self.lr = arg['lr']
        self.save_frequency = arg['save_frequency']
        self.ratio = arg['ratio']
        self.save_root = arg['save_root']
        self.epoch = arg['epoch']
        self.train_accuracy = arg['train_accuracy']
        self.train_accuracy_frequency = arg['train_accuracy_frequency']
        self.test_frequency = arg['test_frequency']
        self.validation_frequency = arg['validation_frequency']

        # 数据处理
        self.features, self.sucs, self.dess = self.load_data()
        print('init model')
        self.net = network.SoccerMap()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_func = network.MyLoss()

        self.net.cuda()

        # 划分训练集、验证集、测试集
        self.train_set, self.validation_set, self.test_set = self.split_data()
        self.train_loader = Data.DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = Data.DataLoader(dataset=self.validation_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = Data.DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=True)

    def getMeanStd(self, features):
        N, C, H, W = features.shape[:4]
        features = features.view(N, C, -1)
        nb_samples = 0.
        # 创建3维的空列表
        channel_mean = torch.zeros(C)
        channel_std = torch.zeros(C)
        channel_mean += features.mean(2).sum(0)
        # 展平后，w,h属于第二维度，对他们求标准差，sum(0)为将同一纬度的数据累加
        channel_std += features.std(2).sum(0)
        # 获取所有batch的数据，这里为1
        nb_samples += N
        # 获取同一batch的均值和标准差
        channel_mean /= nb_samples
        channel_std /= nb_samples
        return channel_mean, channel_std
      
    def load_data(self):
        '''
        读取训练所需要的数据并返回
        如果不存在，调用preprocess生成需要的数据并读取返回
        '''
        print('准备读取数据')
        features = np.load('dataset/' + '/feature.npy', allow_pickle=True)
        sucs = np.load('dataset/' + '/value.npy', allow_pickle=True)
        dess = np.load('dataset/' + '/des.npy', allow_pickle=True)
        # features = np.load('dataset/' + '/9614_feature.npy', allow_pickle=True)
        # sucs = np.load('dataset/' + '/9614_suc.npy', allow_pickle=True)
        # dess = np.load('dataset/' + '/9614_des.npy', allow_pickle=True)
        print('读取：features sucs dess')
        print('读取完毕')
        return features, sucs, dess

    def split_data(self):
        '''
        把数据分割成训练集和测试集
        '''
        features = torch.tensor(self.features, dtype=torch.float)
        sucs = torch.tensor(self.sucs, dtype=torch.int)
        dess = torch.tensor(self.dess, dtype=torch.int)
        # 特征归一化
        mean,std = self.getMeanStd(features)
        features = transforms.Normalize(mean,std)(features)
        
        data_set = TensorDataset(features, dess, sucs)
        ratio_sum = sum(self.ratio)
        for i in range(len(self.ratio)):
            self.ratio[i] = self.ratio[i] / ratio_sum
        print('正在拆分数据，训练数据占比：', str(self.ratio[0]), ';验证数据占比：', str(self.ratio[1]), ';测试数据占比：', str(self.ratio[2]))
        train_data_num = int(len(features) * self.ratio[0])
        validation_data_num = int(len(features) * self.ratio[1])
        test_data_num = int(len(features) * self.ratio[2])
        train_set = Subset(data_set, range(train_data_num))
        validation_set = Subset(data_set, range(train_data_num, train_data_num + validation_data_num))
        test_set = Subset(data_set, range(train_data_num + validation_data_num, train_data_num + validation_data_num + test_data_num))
        return train_set, validation_set, test_set

    def train(self):
        print('开始训练')
        net_pos = torch.load('pos.npk')
        for ep in range(1, self.epoch):
            for step, (X, destination, success) in enumerate(self.train_loader):
                X_cuda = X.cuda()
                X_cuda_pos = X_cuda[:,0:13,:,:]
                Y_cuda_pos = net_pos(X_cuda_pos)
                X_cuda = torch.concat((X_cuda[:,4:19,:,:], Y_cuda_pos), 1)
                d_cuda = destination.cuda()
                s_cuda = success.cuda()
                output = self.net(X_cuda)  # 喂数据并前向传播
                loss = self.loss_func(output, d_cuda, s_cuda)  # 计算损失
                self.optimizer.zero_grad()  # 清除梯度
                loss.backward()  # 计算梯度，误差回传
                self.optimizer.step()  # 根据计算的梯度，更新网络中的参数
                # 损失
            print('epoch: {}, loss: {}'.format(ep, loss.data.item()))

            if self.train_accuracy and ep % self.train_accuracy_frequency == 0 and ep > 0:
                self.cal_train_accuracy()
            #if ep % self.test_frequency == 0 and ep > 0:
            #    self.test()  # 这一部分只是为了看看，不能参与选择Model！！！！！！！！！！！可以直接注释掉
            if ep % self.validation_frequency == 0 and ep > 0:
                error = self.cal_validation_accuracy()
            if ep % self.save_frequency == 0 and ep > 0:
                save_name = self.save_root + 'ep' + str(ep) + '.npk'
                torch.save(self.net, save_name)
                print(save_name, '模型已保存')

    def cal_train_accuracy(self):
        print('训练样本准确率如下')
        num = 0  # 计数器，一共统计十万个训练样本的结果即可
        nll = 0  # 预测偏离均方差
        with torch.no_grad():
          for step, (X, destination, success) in enumerate(self.train_loader):
              if num > 1e6:
                  break
              X = X.cuda()
              destination = destination.cuda()
              success = success.cuda()
              output_matrix = self.net(X)
  
              for i in range(len(output_matrix)):
                  om = output_matrix[i]
                  dn = destination[i].item()
                  lb = 1
                  nll += -torch.log(om.view(1, -1)[0][dn]) * lb - torch.log(1 - om.view(1, -1)[0][dn]) * (1 - lb)
  
              num += self.batch_size
          print('接球点nll误差:')
          error = nll / (num+0.00000000001)
          print(error)
        return error

    def cal_validation_accuracy(self):
        print('测试验证集表现')
        num = 0  # 计数器，一共统计十万个训练样本的结果即可
        nll = 0  # 预测偏离均方差
        with torch.no_grad():
          for step, (X, destination, success) in enumerate(self.validation_loader):
              X = X.cuda()
              destination = destination.cuda()
              success = success.cuda()
              output_matrix = self.net(X)
  
              for i in range(len(output_matrix)):
                  om = output_matrix[i]
                  dn = destination[i].item()
                  lb = 1
                  nll += -torch.log(om.view(1, -1)[0][dn]) * lb - torch.log(1 - om.view(1, -1)[0][dn]) * (1 - lb)
              num += self.batch_size
          print('接球点nll误差:')
          error = nll / (num + 0.00000000001)
          print(error)
        return error

    def test(self):
        print('开始测试')
        num = 0  # 计数器，一共统计十万个训练样本的结果即可
        nll = 0  # 预测偏离均方差
        with torch.no_grad():
          for step, (X, destination, success) in enumerate(self.test_loader):
              X = X.cuda()
              destination = destination.cuda()
              success = success.cuda()
              output_matrix = self.net(X)
  
              for i in range(len(output_matrix)):
                  om = output_matrix[i]
                  dn = destination[i].item()
                  lb = 1
                  nll += -torch.log(om.view(1, -1)[0][dn]) * lb - torch.log(1 - om.view(1, -1)[0][dn]) * (1 - lb)
              num += self.batch_size
          print('接球点nll误差:')
          error = nll / (num + 0.00000000001)
          print(error)
        return error
