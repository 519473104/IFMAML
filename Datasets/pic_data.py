import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras.utils import Sequence
# import torch
# from Datasets.QPZZ_path import T0, T3, T_valid
from Datasets.Tsinghua_path_10C import T0, T3, T_valid
# from picture_path import T0, T3, T_valid
from torch.utils import data
from PIL import Image
from MetaFD.my_utils.init_utils import sample_label_shuffle

def read_directory(directory_name, height, width, normal):
    # height=64
    # width=64
    # normal=1
    file_list = os.listdir(directory_name)
    img = []
    for each_file in file_list:
        img0 = Image.open(directory_name + '/' + each_file)
        gray = img0.resize((height, width))
        img.append(np.array(gray).astype(np.float))
    if normal:
        data = np.array(img) / 255.0  # 归一化
    else:
        data = np.array(img)
    data = data.reshape(-1, 3, height, width)
    return data

class Data_PU():
    def __int__(self, T1 = True):
        if T1:
            self.train = T3
            self.test = T0
        else:
            self.train = T4w
            self.test = T6w  # 4 new classes
    def get_data(self, train_mode=True, data_size = 10 ,height=64, width=64,normal=1):
        data_file = T3 if train_mode else T0
        n_way = len(data_file)  # the num of categories
        data_set = []
        for i in range(n_way):
            if train_mode == True:
                data_size = 10
                data = read_directory(data_file[i], height, width, normal)
                data = data[:data_size]
                data_set.append(data)
            else:
                data_size = 30
                data = read_directory(data_file[i], height, width, normal)
                data = data[:data_size]
                data_set.append(data)
        data_set = np.stack(data_set, axis=0)  # (n_way, n, sample_len)
        data_set = np.asarray(data_set, dtype=np.float32)
        label = np.arange(n_way, dtype=np.int32).reshape(n_way, 1)
        label = np.repeat(label, data_size, axis=1)  # [n_way, examples]
        return data_set, label  # [Nc,num_each_way,1,1024], [Nc, 50]

class Data_PU_valid():
    # def __int__(self, T1 = True):
    #     self.valid = T_valid
    def get_valid_data(self, data_size = 20 ,height=64, width=64,normal=1):
        data_file = T_valid
        n_way = len(data_file)  # the num of categories
        data_set = []
        for i in range(n_way):
            data_size = 20
            data = read_directory(data_file[i], height, width, normal)
            data = data[:data_size]
            data_set.append(data)
        data_set = np.stack(data_set, axis=0)  # (n_way, n, sample_len)
        data_set = np.asarray(data_set, dtype=np.float32)
        label = np.arange(n_way, dtype=np.int32).reshape(n_way, 1)
        label = np.repeat(label, data_size, axis=1)  # [n_way, examples]
        return data_set, label  # [Nc,num_each_way,1,1024], [Nc, 50]

train_data = Data_PU().get_data(train_mode=True)
test_data = Data_PU().get_data(train_mode=False)
print(train_data[0].shape)
print(test_data[0].shape)

# class MAML_Dataset(data.Dataset):
#     def __init__(self, mode, ways):
#         super().__init__()
#         self.height = 64
#         self.width = 64
#         self.__getdata__(mode)
#
#     def __getdata__(self, mode):
#         data = Data_PU().get_data(train_mode=True, height=self.height, width=self.width, normal=1)
#         if mode == 'train':
#             self.x, self.y = data[0][:, :50], data[1][:, :50]
#         elif mode == 'validation':
#             self.x, self.y = data[0][:, 50:100], data[1][:, 50:100]
#         else:
#             if mode == 'test':
#                 data = Data_PU().get_data(train_mode=False,height=self.height, width =self.width, normal=1)
#                 self.x, self.y = data[0], data[1]
#             else:
#                 exit('Mode error')
#         # self.x = np.expand_dims(self.x, axis=-1)  # x: (n_way, n, len, 1), y: (n_way, n)
#         self.x = self.x.reshape([-1, 3, self.height, self.width])  # x: (n_way*n, len, 1), y: (n_way*n)
#         self.y = self.y.reshape(-1)
#         self.x, self.y = sample_label_shuffle(self.x, self.y)
#         print(f'x-shape: {self.x.shape}, y-shape: {self.y.shape}')
#
#     def __getitem__(self, item):
#         x = self.x[item]  # (NC, l)
#         y = self.y[item]
#         return x, y  # , label
#
#     def __len__(self):
#         return len(self.x)

class MAML_Dataset(data.Dataset):
    def __init__(self, mode, ways):
        super().__init__()
        self.height = 64
        self.width = 64
        self.__getdata__(mode)

    def __getdata__(self, mode):
        if mode == 'train':
            data = Data_PU().get_data(train_mode=True,height=self.height, width =self.width, normal=1)
            self.x, self.y = data[0], data[1]
        else:
            if mode == 'validation':
                data = Data_PU_valid().get_valid_data(height=self.height, width=self.width, normal=1)
                self.x, self.y = data[0], data[1]
            elif mode == 'test':
                data = Data_PU().get_data(train_mode=False, height=self.height, width=self.width, normal=1)
                self.x, self.y = data[0], data[1]
            else:
                exit('Mode error')
        # self.x = np.expand_dims(self.x, axis=-1)  # x: (n_way, n, len, 1), y: (n_way, n)
        self.x = self.x.reshape([-1, 3, self.height, self.width])  # x: (n_way*n, len, 1), y: (n_way*n)
        self.y = self.y.reshape(-1)
        self.x, self.y = sample_label_shuffle(self.x, self.y)
        print(f'x-shape: {self.x.shape}, y-shape: {self.y.shape}')

    def __getitem__(self, item):
        x = self.x[item]  # (NC, l)
        y = self.y[item]
        return x, y  # , label

    def __len__(self):
        return len(self.x)

# class MAML_Dataset(data.Dataset):
#     def __init__(self, mode, ways):
#         super().__init__()
#         self.height = 64
#         self.width = 64
#         self.__getdata__(mode)
#
#     def __getdata__(self, mode):
#         if mode == 'train':
#             data = Data_PU().get_data(train_mode=True,height=self.height, width =self.width, normal=1)
#             self.x, self.y = data[0], data[1]
#         else:
#             data = Data_PU().get_data(train_mode=False,height=self.height, width=self.width, normal=1)
#             if mode == 'validation':
#                 self.x, self.y = data[0][:, :50], data[1][:, :50]
#             elif mode == 'test':
#                 self.x, self.y = data[0][:, 100:200], data[1][:, 100:200]
#             else:
#                 exit('Mode error')
#         # self.x = np.expand_dims(self.x, axis=-1)  # x: (n_way, n, len, 1), y: (n_way, n)
#         self.x = self.x.reshape([-1, 3, self.height, self.width])  # x: (n_way*n, len, 1), y: (n_way*n)
#         self.y = self.y.reshape(-1)
#         self.x, self.y = sample_label_shuffle(self.x, self.y)
#         print(f'x-shape: {self.x.shape}, y-shape: {self.y.shape}')
#
#     def __getitem__(self, item):
#         x = self.x[item]  # (NC, l)
#         y = self.y[item]
#         return x, y  # , label
#
#     def __len__(self):
#         return len(self.x)

if __name__ == "__main__":
    import learn2learn as l2l

    train_dataset = l2l.data.MetaDataset(MAML_Dataset(mode='train', ways=5))
    valid_dataset = l2l.data.MetaDataset(MAML_Dataset(mode='validation', ways=5))
    test_dataset = l2l.data.MetaDataset(MAML_Dataset(mode='test', ways=5))
    shots = 5  # 注意要保证: shots*2*ways >= len(self.x)
    ways = 5
    num_tasks = 100

    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=[
        l2l.data.transforms.NWays(train_dataset, ways),
        l2l.data.transforms.KShots(train_dataset, 2 * shots),
        l2l.data.transforms.LoadData(train_dataset),
        l2l.data.transforms.RemapLabels(train_dataset),
        l2l.data.transforms.ConsecutiveLabels(train_dataset),
    ], num_tasks=num_tasks)

    for i in range(num_tasks+1):
        task = train_tasks.sample()
        data, labels = task
        print(data.shape, labels.shape)
        # print(data)
        print(labels)
        print(f'{i+1}')