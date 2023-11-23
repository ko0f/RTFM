import csv
import argparse
import numpy as np
import os
import glob
import numpy as np
from scipy.io import loadmat
from os import walk
from tqdm import tqdm

'''
Ground Truth file is the Y of our dataset.

This script reads Matlab files with start-end frame indexes of abnormal events, then 
creates a `gt` array with 0/1 values for normal/abnormal frames, each item represents a 
frame. All video Y labels are concatinated into one big `gt` array.

The extracted feature npy files are only used to validate number of frames against matlab files.
'''

parser = argparse.ArgumentParser(description='RTFM GT generator')
parser.add_argument('--output', default='list/gt.npy', help='file of ground truth')
parser.add_argument('--input', type=str, default='list/filenames.csv', help='list of features npy files')
parser.add_argument('--frequency', type=int, default=16)
args = parser.parse_args()

file_list = []
with open(args.input) as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    for row in reader:
        file_list.append(row)

num_frame = 0
gt = []

for row in tqdm(file_list):
    file = row[0]
    if not os.path.exists(file):
        continue
    features = np.load(file.strip('\n'), allow_pickle=True)
    # features = [t.cpu().detach().numpy() for t in features]
    features = np.array(features, dtype=np.float32)
    num_frame = features.shape[0] * args.frequency

    for i in range(0, num_frame):
        gt.append(0.0)

    if len(row) > 1:
        for j in range(0, int((len(row)-1)/2)):
            val = row[1 + j * 2]
            if val:
                start = int(val)  
                end = int(row[2 + j * 2])
                for f in range(start, end):
                    gt[f] = 1.0
            else: 
                break


gt = np.array(gt, dtype=float)
np.save(args.output, gt)
print(len(gt))





