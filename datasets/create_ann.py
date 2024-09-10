import torch
import argparse
import glob
import os
import os.path as osp
import re
import shutil
import sys


def generate_anno_file(train_list, file_name='meta_info_Vimeo90K_GT.txt'):
    """Generate anno file for Vimeo90K datasets from the official train list.

    Args:
        train_list (str): Train list path for Vimeo90K datasets.
        file_name (str): Saved file name. Default: 'meta_info_Vimeo90K_GT.txt'.
    """

    print(f'Generate annotation files {file_name}...')
    # read official train list
    with open(train_list) as f:
        lines = [line.rstrip() for line in f]
    txt_file = osp.join(osp.dirname(train_list), file_name)
    with open(txt_file, 'w') as f:
        for line in lines:
            f.write(f'{line} (256, 448, 3)\n')


def generate_anno_file1(train_list, file_name='meta_info_Vimeo90K_GT.txt'):
    """Generate anno file for Vimeo90K datasets from the official train list.

    Args:
        train_list (str): Train list path for Vimeo90K datasets.
        file_name (str): Saved file name. Default: 'meta_info_Vimeo90K_GT.txt'.
    """

    print(f'Generate annotation files {file_name}...')
    # read official train list
    with open(train_list) as f:
        lines = [line.rstrip() for line in f]
    txt_file = osp.join(osp.dirname(train_list), file_name)
    with open(txt_file, 'w') as f:
        for line in lines:
            f.write(f'{line} 7\n')        

def generate_anno_file_reds(train_list, file_name='meta_info_Vimeo90K_GT.txt'):
    print(f'Generate annotation files {file_name}...')
    save_path = osp.join(train_list, file_name)
    exclusive_list = [0, 11, 15, 20]
    pre_date = []  # ['001', '002', ..., '269']
    for i in range(270):
        if i in exclusive_list:
            continue
        else:
            pre_date.append('{:0>3}'.format(i))

    with open(save_path, 'w') as f:
        for v in pre_date:
            f.write(v + '\n')

if __name__ == '__main__':
    generate_anno_file_reds('.', file_name='meta_info_REDS_GT.txt')