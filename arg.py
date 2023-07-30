"""arg参数设定"""

arg = {
       # 训练参数
       'batch_size': 32,
       'lr': 1e-6,#0.000001,  # learning rate

       'epoch': 50,  # 训练代数
       'save_frequency': 10,  # 多少个epoch保存一次模型
       'ratio': [8, 1, 1],  # 划分数据集占比 训练集：验证集：测试集
       'train_accuracy': False,  # 是否计算训练集上的精度
       'train_accuracy_frequency': 10,  # 训练多少代，统计一下训练集上的精度
       'test_frequency': 5,  # 训练多少代测试一次
       'save_root': 'net/',
       # 验证集参数
       'validation_frequency': 5,
       # 测试参数
       'test': False,  # False 就训练 True 就测试
       'save_test_result': False,  # 是否保存测试结果，给出example和eg的延申应用需要用到test_result
       }