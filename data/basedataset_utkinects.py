import torch
import numpy as np
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from numpy.random import randint
from utils import *
import pdb
import random
import re
import cv2
import xml.etree.ElementTree as ET

class BaseDataset(Dataset):
    def __init__(self, vid_list, actions_dict, features_path, depth_features_path, gt_path, pad_idx, n_class,
                 n_query=8,  mode='train', obs_perc=0.2, args=None):
        self.n_class = n_class
        self.actions_dict = actions_dict
        self.pad_idx = pad_idx
        self.features_path = features_path
        self.depth_features_path = depth_features_path
        self.gt_path = gt_path
        self.mode = mode
        self.sample_rate = args.sample_rate
        self.vid_list = list()
        self.n_query = n_query
        self.args = args
        self.NONE = self.n_class - 1

        if self.mode == 'train' or self.mode == 'val':
            for vid in vid_list:
                self.vid_list.append([vid, .4])
                self.vid_list.append([vid, .45])
                self.vid_list.append([vid, .2])
                self.vid_list.append([vid, .25])
                self.vid_list.append([vid, .3])
                self.vid_list.append([vid, .35])
                self.vid_list.append([vid, .5])
                self.vid_list.append([vid, .55])
                self.vid_list.append([vid, .6])
                self.vid_list.append([vid, .65])
        elif self.mode == 'test' :
            for vid in vid_list:
                self.vid_list.append([vid, obs_perc])

        self._make_input(vid, 0.2)

    def xmltodepth(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        tag = os.path.basename(file_path).replace('.xml', '')
        depth_node = root.find(tag)

        if depth_node is None:
            raise ValueError(f"cannot find the tag named {tag}")

        width = int(depth_node.find("width").text)
        height = int(depth_node.find("height").text)

        data_node = depth_node.find("data")

        if data_node is None or not data_node.text:
            raise ValueError("empty data")

        depth_values = np.fromstring(data_node.text.strip(), sep=' ')

        if depth_values.size != width * height:
            raise ValueError(f"data size mismatch: prediction: {width * height}, real: {depth_values.size}")

        # data resize
        depth_data = depth_values.reshape((height, width))
        depth_data = cv2.resize(depth_data, (int(height/2), int(width/2)))
        depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)
        return depth_normalized

    def __getitem__(self, idx):
        vid_file, obs_perc = self.vid_list[idx]
        obs_perc = float(obs_perc)
        item = self._make_input(vid_file, obs_perc)
        return item


    def _make_input(self, vid_file, obs_perc ):
        vid_file = vid_file.split('/')[-1]
        vid_name = vid_file

        gt_file = os.path.join(self.gt_path, vid_file)
        feature_file = os.path.join(self.features_path, vid_file.split('.')[0]+".npy")
        features = np.load(feature_file)

        depth_feature_file = os.path.join(self.depth_features_path, vid_file.split('.')[0]+".npy")
        depth_features = np.load(depth_feature_file)

        with open(gt_file, 'r') as file_ptr:
            lines = file_ptr.readlines()
            valid_lines = [line.strip() for line in lines if len(line.strip().split(',')) == 3]

        #file_ptr = open(gt_file, 'r')
        #all_content = file_ptr.read().split('\n')[:-1]
        all_content = [line.split(',')[1] for line in valid_lines]
        vid_len = len(all_content)
        observed_len = int(obs_perc*vid_len)
        pred_len = int(0.5*vid_len)

        start_frame = 0

        # feature slicing
        features = features[start_frame : start_frame + observed_len] #[S, C]
        features = features[::self.sample_rate]

        depth_features = depth_features[start_frame : start_frame + observed_len] #[S, C]
        depth_features = depth_features[::self.sample_rate]

        past_content = all_content[start_frame : start_frame + observed_len] #[S]
        past_content = past_content[::self.sample_rate]
        past_label = self.seq2idx(past_content)

        if np.shape(features)[0] != len(past_content) :
            features = features[:len(past_content),]

        if np.shape(depth_features)[0] != len(past_content) :
            depth_features = depth_features[:len(past_content),]

        future_content = \
        all_content[start_frame + observed_len: start_frame + observed_len + pred_len] #[T]
        future_content = future_content[::self.sample_rate]
        trans_future, trans_future_dur = self.seq2transcript(future_content)
        trans_future = np.append(trans_future, self.NONE)
        trans_future_target = trans_future #target


        # add padding for future input seq
        trans_seq_len = len(trans_future_target)
        diff = self.n_query - trans_seq_len
        if diff > 0 :
            tmp = np.ones(diff)*self.pad_idx
            trans_future_target = np.concatenate((trans_future_target, tmp))
            tmp_len = np.ones(diff+1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))
        elif diff < 0 :
            trans_future_target = trans_future_target[:self.n_query]
            trans_future_dur = trans_future_dur[:self.n_query]
        else :
            tmp_len = np.ones(1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))


        item = {'features':torch.Tensor(features),
                'depth_features':torch.Tensor(depth_features),
                'past_label':torch.Tensor(past_label),
                'trans_future_dur':torch.Tensor(trans_future_dur),
                'trans_future_target' : torch.Tensor(trans_future_target),
                }

        return item


    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''

        b_features = [item['features'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_trans_future_dur = [item['trans_future_dur'] for item in batch]
        b_trans_future_target = [item['trans_future_target'] for item in batch]
        b_depth_features = [item['depth_features'] for item in batch]

        batch_size = len(batch)

        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
                                                         padding_value=self.pad_idx)
        b_trans_future_dur = torch.nn.utils.rnn.pad_sequence(b_trans_future_dur, batch_first=True,
                                                        padding_value=self.pad_idx)
        b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self.pad_idx)
        b_depth_features = torch.nn.utils.rnn.pad_sequence(b_depth_features, batch_first=True, padding_value=0)

        batch = [b_features, b_depth_features, b_past_label, b_trans_future_dur, b_trans_future_target]

        return batch


    def __len__(self):
        return len(self.vid_list)

    def seq2idx(self, seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            idx[i] = self.actions_dict[seq[i].replace(' ', '')]
        return idx

    def seq2transcript(self, seq):
        transcript_action = []
        transcript_dur = []
        action = seq[0].replace(' ', '')
        transcript_action.append(self.actions_dict[action])
        last_i = 0
        for i in range(len(seq)):
            if action != seq[i].replace(' ', ''):
                action = seq[i].replace(' ', '')
                transcript_action.append(self.actions_dict[action])
                duration = (i-last_i)/len(seq)
                last_i = i
                transcript_dur.append(duration)
        duration = (len(seq)-last_i)/len(seq)
        transcript_dur.append(duration)
        return np.array(transcript_action), np.array(transcript_dur)