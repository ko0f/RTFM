import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
import csv

torch.set_default_tensor_type('torch.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        if test_mode:
            self.file_list = args.list_test
        else:
            self.file_list = args.list_train

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        print('LIST test {0}, is_normal {1}\n{2}\n----------------------------------------------------'.format(test_mode, is_normal, self.list))


    def _parse_list(self):
        rows = []
        with open(self.file_list) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=['fn'])
            for row in reader:
                rows.append(row)

        if not self.test_mode:
            self.list = [k['fn'] for k in rows if self.is_normal and 'abnormal' not in k['fn'] or not self.is_normal and 'abnormal' in k['fn']]
        else:
            self.list = [k['fn'] for k in rows]

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
