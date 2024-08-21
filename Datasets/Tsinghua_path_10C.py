import os
import numpy as np
from PIL import Image
from torch.utils import data
from MetaFD.my_utils.init_utils import one_hot_encode, sample_label_shuffle
from torchvision import transforms as tfs

root_dir = r"I:\2023_10.22\msf-rescnn-fd-main\toImgs\Tsinghua_64_spca10C"
# root_dir = r"E:\msf-rescnn-fd-main\toImgs\Tsinghua_64_WT10C"

def get_file(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    # if len(file_list) != 1:  # 若文件中存在不止一个文件，则存在歧义
    #     print('There are {} files in [{}]'.format(len(file_list), root_path))
    #     exit()
    assert len(file_list) == 1, 'There are {} files in [{}]'.format(len(file_list), root_path)  #
    return file_list[0]

# NC
Healthy_V1run_0A = os.path.join(root_dir, r'V1run_0A\Healthy')
Healthy_V1run_0dot4A = os.path.join(root_dir, r'V1run_0dot4A\Healthy')
Healthy_V2run_0A = os.path.join(root_dir, r'V2run_0A\Healthy')
Healthy_V2run_0dot4A = os.path.join(root_dir, r'V2run_0dot4A\Healthy')
Healthy_V4run_0A = os.path.join(root_dir, r'V4run_0A\Healthy')
Healthy_V4run_0dot4A = os.path.join(root_dir, r'V4run_0dot4A\Healthy')

Healthy = [Healthy_V1run_0A , Healthy_V1run_0dot4A, Healthy_V2run_0A, Healthy_V2run_0dot4A, Healthy_V4run_0A, Healthy_V4run_0dot4A]
# mosun
IB_IRC_V1run_0A = os.path.join(root_dir, r'V1run_0A\IB_IRC')
IB_IRC_V1run_0dot4A = os.path.join(root_dir, r'V1run_0dot4A\IB_IRC')
IB_IRC_V2run_0A = os.path.join(root_dir, r'V2run_0A\IB_IRC')
IB_IRC_V2run_0dot4A = os.path.join(root_dir, r'V2run_0dot4A\IB_IRC')
IB_IRC_V4run_0A = os.path.join(root_dir, r'V4run_0A\IB_IRC')
IB_IRC_V4run_0dot4A = os.path.join(root_dir, r'V4run_0dot4A\IB_IRC')

IB_IRC = [IB_IRC_V1run_0A, IB_IRC_V1run_0dot4A ,IB_IRC_V2run_0A,IB_IRC_V2run_0dot4A, IB_IRC_V4run_0A, IB_IRC_V4run_0dot4A]

# IR + OR
PB_IRC_V1run_0A = os.path.join(root_dir, r'V1run_0A\PB_IRC')
PB_IRC_V1run_0dot4A = os.path.join(root_dir, r'V1run_0dot4A\PB_IRC')
PB_IRC_V2run_0A = os.path.join(root_dir, r'V2run_0A\PB_IRC')
PB_IRC_V2run_0dot4A = os.path.join(root_dir, r'V2run_0dot4A\PB_IRC')
PB_IRC_V4run_0A = os.path.join(root_dir, r'V4run_0A\PB_IRC')
PB_IRC_V4run_0dot4A = os.path.join(root_dir, r'V4run_0dot4A\PB_IRC')

PB_IRC = [PB_IRC_V1run_0A, PB_IRC_V1run_0dot4A ,PB_IRC_V2run_0A , PB_IRC_V2run_0dot4A, PB_IRC_V4run_0A,PB_IRC_V4run_0dot4A ]
# OR+IR
PG_CT_V1run_0A = os.path.join(root_dir, r'V1run_0A\PG_CT')
PG_CT_V1run_0dot4A = os.path.join(root_dir, r'V1run_0dot4A\PG_CT')
PG_CT_V2run_0A = os.path.join(root_dir, r'V2run_0A\PG_CT')
PG_CT_V2run_0dot4A = os.path.join(root_dir, r'V2run_0dot4A\PG_CT')
PG_CT_V4run_0A = os.path.join(root_dir, r'V4run_0A\PG_CT')
PG_CT_V4run_0dot4A = os.path.join(root_dir, r'V4run_0dot4A\PG_CT')

PG_CT = [PG_CT_V1run_0A ,PG_CT_V1run_0dot4A, PG_CT_V2run_0A, PG_CT_V2run_0dot4A, PG_CT_V4run_0A, PG_CT_V4run_0dot4A]

# OR
SG_TM_V1run_0A = os.path.join(root_dir, r'V1run_0A\SG_TM')
SG_TM_V1run_0dot4A = os.path.join(root_dir, r'V1run_0dot4A\SG_TM')
SG_TM_V2run_0A = os.path.join(root_dir, r'V2run_0A\SG_TM')
SG_TM_V2run_0dot4A = os.path.join(root_dir, r'V2run_0dot4A\SG_TM')
SG_TM_V4run_0A = os.path.join(root_dir, r'V4run_0A\SG_TM')
SG_TM_V4run_0dot4A = os.path.join(root_dir, r'V4run_0dot4A\SG_TM')

SG_TM =[SG_TM_V1run_0A,SG_TM_V1run_0dot4A,SG_TM_V2run_0A,SG_TM_V2run_0dot4A,SG_TM_V4run_0A,SG_TM_V4run_0dot4A]

# 6
RG_TM_V1run_0A = os.path.join(root_dir, r'V1run_0A\RG_TM')
RG_TM_V1run_0dot4A = os.path.join(root_dir, r'V1run_0dot4A\RG_TM')
RG_TM_V2run_0A = os.path.join(root_dir, r'V2run_0A\RG_TM')
RG_TM_V2run_0dot4A = os.path.join(root_dir, r'V2run_0dot4A\RG_TM')
RG_TM_V4run_0A = os.path.join(root_dir, r'V4run_0A\RG_TM')
RG_TM_V4run_0dot4A = os.path.join(root_dir, r'V4run_0dot4A\RG_TM')

RG_TM =[RG_TM_V1run_0A,RG_TM_V1run_0dot4A,RG_TM_V2run_0A,RG_TM_V2run_0dot4A,RG_TM_V4run_0A,RG_TM_V4run_0dot4A]
# 7
PG_TRC_V1run_0A = os.path.join(root_dir, r'V1run_0A\PG_TRC')
PG_TRC_V1run_0dot4A = os.path.join(root_dir, r'V1run_0dot4A\PG_TRC')
PG_TRC_V2run_0A = os.path.join(root_dir, r'V2run_0A\PG_TRC')
PG_TRC_V2run_0dot4A = os.path.join(root_dir, r'V2run_0dot4A\PG_TRC')
PG_TRC_V4run_0A = os.path.join(root_dir, r'V4run_0A\PG_TRC')
PG_TRC_V4run_0dot4A = os.path.join(root_dir, r'V4run_0dot4A\PG_TRC')

PG_TRC =[PG_TRC_V1run_0A,PG_TRC_V1run_0dot4A,PG_TRC_V2run_0A,PG_TRC_V2run_0dot4A,PG_TRC_V4run_0A,PG_TRC_V4run_0dot4A]
# 8
PB_REF_V1run_0A = os.path.join(root_dir, r'V1run_0A\PB_REF')
PB_REF_V1run_0dot4A = os.path.join(root_dir, r'V1run_0dot4A\PB_REF')
PB_REF_V2run_0A = os.path.join(root_dir, r'V2run_0A\PB_REF')
PB_REF_V2run_0dot4A = os.path.join(root_dir, r'V2run_0dot4A\PB_REF')
PB_REF_V4run_0A = os.path.join(root_dir, r'V4run_0A\PB_REF')
PB_REF_V4run_0dot4A = os.path.join(root_dir, r'V4run_0dot4A\PB_REF')

PB_REF =[PB_REF_V1run_0A,PB_REF_V1run_0dot4A,PB_REF_V2run_0A,PB_REF_V2run_0dot4A,PB_REF_V4run_0A,PB_REF_V4run_0dot4A]

# 9
PB_ORF_V1run_0A = os.path.join(root_dir, r'V1run_0A\PB_ORF')
PB_ORF_V1run_0dot4A = os.path.join(root_dir, r'V1run_0dot4A\PB_ORF')
PB_ORF_V2run_0A = os.path.join(root_dir, r'V2run_0A\PB_ORF')
PB_ORF_V2run_0dot4A = os.path.join(root_dir, r'V2run_0dot4A\PB_ORF')
PB_ORF_V4run_0A = os.path.join(root_dir, r'V4run_0A\PB_ORF')
PB_ORF_V4run_0dot4A = os.path.join(root_dir, r'V4run_0dot4A\PB_ORF')

PB_ORF =[PB_ORF_V1run_0A,PB_ORF_V1run_0dot4A,PB_ORF_V2run_0A,PB_ORF_V2run_0dot4A,PB_ORF_V4run_0A,PB_ORF_V4run_0dot4A]

# 10
IB_IRF_V1run_0A = os.path.join(root_dir, r'V1run_0A\IB_IRF')
IB_IRF_V1run_0dot4A = os.path.join(root_dir, r'V1run_0dot4A\IB_IRF')
IB_IRF_V2run_0A = os.path.join(root_dir, r'V2run_0A\IB_IRF')
IB_IRF_V2run_0dot4A = os.path.join(root_dir, r'V2run_0dot4A\IB_IRF')
IB_IRF_V4run_0A = os.path.join(root_dir, r'V4run_0A\IB_IRF')
IB_IRF_V4run_0dot4A = os.path.join(root_dir, r'V4run_0dot4A\IB_IRF')

IB_IRF =[IB_IRF_V1run_0A,IB_IRF_V1run_0dot4A,IB_IRF_V2run_0A,IB_IRF_V2run_0dot4A,IB_IRF_V4run_0A,IB_IRF_V4run_0dot4A]


# # Tasks with 5-way
T3 = [Healthy[0], IB_IRC[0], PB_IRC[0],PG_CT[0] ,SG_TM[0]]
T0 = [Healthy[5], IB_IRC[5], PB_IRC[5],PG_CT[5] ,SG_TM[5]]
#
#
T_valid = [Healthy[5], IB_IRC[5], PB_IRC[5],PG_CT[5] ,SG_TM[5]]
# T_valid = [Healthy[2], IB_IRC[4],PB_IRC[1] ,PG_CT[1] ,SG_TM[3]] # V4_0.4A→V1
# # T_valid = [Healthy[2], IB_IRC[3],PB_IRC[1] ,PG_CT[0] ,SG_TM[3]] # V4_0.4A→V4
# T_valid = [Healthy[5], IB_IRC[4],PB_IRC[1] ,PG_CT[1] ,SG_TM[0]] # V2→V2_0.4A
# # T_valid = [Healthy[0], IB_IRC[3],PB_IRC[5] ,PG_CT[2] ,SG_TM[3]] # V4→V1_0.4A
# T_valid = [Healthy[0], IB_IRC[1],PB_IRC[1] ,PG_CT[4] ,SG_TM[3]] # V2→V4_0.4A
# # T_valid = [Healthy[0], IB_IRC[5],PB_IRC[5] ,PG_CT[4] ,SG_TM[3]] # V2→V1_0.4A
# T_valid = [Healthy[0], IB_IRC[1],PB_IRC[1] ,PG_CT[2] ,SG_TM[5]] # V4→V1
# T_valid = [Healthy[1], IB_IRC[3],PB_IRC[3] ,PG_CT[4] ,SG_TM[0]] # V2→V4
# T_valid = [Healthy[2], IB_IRC[3],PB_IRC[4] ,PG_CT[5] ,SG_TM[5]] # V1→V1_0.4A


if __name__ == "__main__":

    print(T0)
    print(T3)
    pass