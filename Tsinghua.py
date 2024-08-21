import numpy as np
import os
import scipy.io as scio
import cv2
from normalize255 import normalize,sparse_coding
import time
from PIL import Image

start = time.perf_counter()
Samples = 10
ImageL = 64
ImageW = ImageL
ImageSize = ImageL * ImageW
dataPoints = ImageSize
signal_cut1 = np.zeros((Samples, ImageSize))
signal_cut2 = np.zeros((Samples, ImageSize))
signal_cut3 = np.zeros((Samples, ImageSize))

OPERATING = ["V1run_0A","V1run_0dot4A","V2run_0A","V2run_0dot4A","V4run_0A","V4run_0dot4A"]
FILES = ["H", "IB_IRC", "PB_IRC","PG_CT","SG_TM", "RG_TM", "PG_TRC","PB_REF","PB_ORF","IB_IRF"]

data_dir = r'I:\data\PGB Experiment @ Tsinghua\Datasets'
output_dir = r'F:\2023_10.22\Tsinghua_spca'

for i in range(0,len(OPERATING)):
    operate_name = OPERATING[i]
    for j in range(0,len(FILES)):
        file_name = FILES[j]
        if j == 1:
            FAULT_NAMES = ["H"]
        if j == 2:
            FAULT_NAMES = ["IB_IRC"]
        if j == 3:
            FAULT_NAMES = ["PB_IRC"]
        if j == 4:
            FAULT_NAMES = ["PG_CT"]
        if j == 5:
            FAULT_NAMES = ["SG_TM"]
        if j == 6:
            FAULT_NAMES = ["RG_TM"]
        if j == 7:
            FAULT_NAMES = ["PG_TRC"]
        if j == 8:
            FAULT_NAMES = ["PB_REF"]
        if j == 9:
            FAULT_NAMES = ["PB_ORF"]
        if j == 10:
            FAULT_NAMES = ["IB_IRF"]

    for iFault in range(0,len(FILES)):
        fault_name = FILES[iFault]

        txt_name = os.path.join(fault_name +"_"+operate_name+"_1")
        mat_path = os.path.join(data_dir,fault_name, txt_name +'.mat')

        data = scio.loadmat(mat_path)
        data = data[fault_name +"_"+operate_name+"_1"]

        data_vibration = data[:,0]
        data_current = data[:,4]
        data_torque = data[:,6]

        data_new = [data_vibration, data_current, data_torque]

        data_new = np.array(data_new)
        data_enhance = sparse_coding(data_new.T, 3, 100, 1, 1e-4) #Proposed
        print(data_enhance.shape)

        random_series = np.random.randint(0, (data_enhance.shape[0] - dataPoints), Samples)

        # signal_cut1, signal_cut2, signal_cut3 = [], [], []


        for i_cut  in range(Samples):
            cut_index = random_series[i_cut]
            signal_cut1[i_cut,:] = normalize(data_enhance[cut_index: cut_index + dataPoints,0])
            signal_cut2[i_cut,:] = normalize(data_enhance[cut_index: cut_index + dataPoints,1])
            signal_cut3[i_cut,:] = normalize(data_enhance[cut_index: cut_index + dataPoints,2])

        for i_image in range(Samples):
            img_pre1, img_pre2, img_pre3 = signal_cut1[i_image],signal_cut2[i_image],signal_cut3[i_image]
            img1 = np.reshape(img_pre1,(int(np.sqrt(dataPoints)),int(np.sqrt(dataPoints))))
            img2 = np.reshape(img_pre2, (int(np.sqrt(dataPoints)), int(np.sqrt(dataPoints))))
            img3 = np.reshape(img_pre3, (int(np.sqrt(dataPoints)), int(np.sqrt(dataPoints))))

            imgRGB = np.dstack((img1, img2, img3))
            imgRGB = cv2.transpose(imgRGB)
            output_path_subF = os.path.join(output_dir, operate_name,fault_name)
            if not os.path.exists(output_path_subF):
                os.makedirs(output_path_subF)
            # image_save_path = os.path.join(output_path_subF, fault_name + "_" + str(i_image) + '.png')
            # cv2.imwrite(image_save_path, imgRGB)

end = time.perf_counter()
print('Running time: %s Seconds' % (end - start))
            # np.savetxt(image_save_path,transformer,fmt='%f', delimiter=",")
