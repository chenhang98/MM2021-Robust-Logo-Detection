import multiprocessing as mp
import json, os, cv2
from tqdm import tqdm
from argparse import ArgumentParser
from imagecorruptions import corrupt
import cv2
import numpy as np
import random
import os.path as osp

sourceroot = '../data/tianchi/train'
targetroot = '../data/tianchi/train_glass_zoom'

def processimg(arg):
    name = arg
    categorys = ['glass_blur', 'zoom_blur'] #'glass_blur'
    img = cv2.imread(os.path.join(sourceroot, name))
    s = np.random.choice([1,2,3,4])
    index = np.random.choice(len(categorys), p = [0.3, 0.7])
    corrupted_image = corrupt(img, corruption_name=categorys[index], severity=s)
    cv2.imwrite(os.path.join(targetroot, name), corrupted_image)


if __name__ == "__main__":
    parser = ArgumentParser(description='Generate data')
    parser.add_argument('-n', '--num-proc', help='num of process', 
        type=int, default=16)
    args = parser.parse_args()

    if not osp.exists(targetroot):
        os.makedirs(targetroot)

    names = os.listdir(sourceroot)
    random.shuffle(names)

    with mp.Pool(processes=args.num_proc) as p:
        with tqdm(total=len(names)) as pbar:
            for _ in p.imap_unordered(processimg, names):
                pbar.update()