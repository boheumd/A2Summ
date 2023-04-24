import os
import random
import torch
import h5py
import numpy as np
import json
import math
from tqdm import tqdm

from helpers.bbox_helper import get_loc_label, get_ctr_label
from helpers.vsumm_helper import get_keyshot_summ

class MSMODataset(object):
    def __init__(self, mode='train', args=None):
        self.gt = json.load(open('{}/{}/annotation/{}.json'.format(args.data_root, args.dataset, mode)))
        self.id_list = list(self.gt.keys())

        self.video_dict = np.load('{}/{}/feature/video_resnet50_{}.npy'.format(args.data_root, args.dataset, mode), allow_pickle=True).item()
        self.text_dict = np.load('{}/{}/feature/text_roberta_{}.npy'.format(args.data_root, args.dataset, mode), allow_pickle=True).item()
        if args.dataset == 'Daily_Mail':
            self.video_summ_dict = np.load('{}/{}/feature/video_summ_resnet50_{}.npy'.format(args.data_root, args.dataset, mode), allow_pickle=True).item()
        else:
            self.video_summ_dict = {}

        for id in tqdm(self.id_list):
            self.video_dict[id] = torch.tensor(self.video_dict[id]).to(torch.float32)
            self.text_dict[id] = torch.tensor(self.text_dict[id]).to(torch.float32)
            
            if args.dataset == 'Daily_Mail':
                self.video_summ_dict[id] = torch.tensor(self.video_summ_dict[id]).to(torch.float32)
            else:
                self.video_summ_dict[id] = torch.zeros(1).to(torch.float32)
        self.dataset = args.dataset

    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, index):
        id = self.id_list[index]

        video = self.video_dict[id] # [T, 2048]
        video_summ = self.video_summ_dict[id]
        text = self.text_dict[id] # [N, 768]

        num_frame = video.shape[0]
        num_keyframe = video_summ.shape[0]
        num_sentence = text.shape[0]
        
        if self.dataset == 'Daily_Mail':
            video_label = torch.tensor(self.gt[id]['video_label'], dtype=torch.long)
            assert torch.sum(video_label) == num_keyframe
        else:
            video_label = torch.zeros(num_frame).to(torch.long)
            video_label[0] = 1
        text_label = torch.tensor(self.gt[id]['text_label'], dtype=torch.long)

        article_sentence = self.gt[id]['article_sentence']
        highlight = self.gt[id]['highlight']

        mask_video = torch.ones(num_frame, dtype=torch.long)
        mask_video_summ = torch.ones(num_keyframe, dtype=torch.long)
        mask_text = torch.ones(num_sentence, dtype=torch.long)

        video_to_text_mask = torch.zeros(1)
        text_to_video_mask = torch.zeros(1)
        return video, video_summ, text, mask_video, mask_video_summ, mask_text, video_label, text_label, article_sentence, highlight, video_to_text_mask, text_to_video_mask

class BLiSSDataset(object):
    def __init__(self, mode='train', args=None):
        self.gt = json.load(open('{}/{}/annotation/{}.json'.format(args.data_root, args.dataset, mode)))
        self.clip_id_list = list(self.gt.keys())

        video_feature_path = '{}/{}/feature/video_clip_{}.npy'.format(args.data_root, args.dataset, mode)
        text_feature_path = '{}/{}/feature/text_roberta_{}.npy'.format(args.data_root, args.dataset, mode)
        video_summ_feature_path = '{}/{}/feature/video_summ_clip_{}.npy'.format(args.data_root, args.dataset, mode)
        video_feature_dict = np.load(video_feature_path, allow_pickle=True).item()
        text_feature_dict = np.load(text_feature_path, allow_pickle=True).item()
        video_summ_feature_dict = np.load(video_summ_feature_path, allow_pickle=True).item()

        self.video_dict = {}
        self.video_summ_dict = {}
        self.text_dict = {}
        for clip_id in tqdm(self.clip_id_list):
            self.video_dict[clip_id] = torch.tensor(video_feature_dict[clip_id]).to(torch.float32)
            self.text_dict[clip_id] = torch.tensor(text_feature_dict[clip_id]).to(torch.float32)
            self.video_summ_dict[clip_id] = torch.tensor(video_summ_feature_dict[clip_id]).to(torch.float32)
            
    def __len__(self):
        return len(self.clip_id_list)
    
    def __getitem__(self, index):
        clip_id = self.clip_id_list[index]
        video_id = self.gt[clip_id]['video_id']

        video = self.video_dict[clip_id] # [T, 512]
        video_summ = self.video_summ_dict[clip_id] # [N, 512]
        text = self.text_dict[clip_id] # [T, 768]

        video_label = torch.tensor(self.gt[clip_id]['video_label'], dtype=torch.long)
        text_label = torch.tensor(self.gt[clip_id]['text_label'], dtype=torch.long)

        num_frame = self.gt[clip_id]['num_frame']
        num_keyframe = self.gt[clip_id]['num_keyframe']
        num_sentence = self.gt[clip_id]['num_sentence']

        assert torch.sum(video_label) == num_keyframe
            
        sentence = self.gt[clip_id]['sentence']
        highlight = self.gt[clip_id]['highlight']
        time_index = self.gt[clip_id]['sentence_time']
        
        video_to_text_mask = torch.zeros((num_frame, num_sentence), dtype=torch.long)
        text_to_video_mask = torch.zeros((num_sentence, num_frame), dtype=torch.long)
        for j in range(num_sentence):
            start_frame, end_frame = time_index[j]
            video_to_text_mask[start_frame: end_frame, j] = 1
            text_to_video_mask[j, start_frame: end_frame] = 1
        
        mask_video = torch.ones(num_frame, dtype=torch.long)
        mask_video_summ = torch.ones(num_keyframe, dtype=torch.long)
        mask_text = torch.ones(num_sentence, dtype=torch.long)

        return video, video_summ, text, mask_video, mask_video_summ, mask_text, video_label, text_label, sentence, highlight, video_to_text_mask, text_to_video_mask

class VideoSummDataset(object):
    def __init__(self, keys, args=None):
        self.keys = keys
        self.video_dict = h5py.File('{}/{}/feature/eccv16_dataset_{}_google_pool5.h5'.format(args.data_root, args.dataset, args.dataset.lower()), 'r')

        text_feature_path = '{}/{}/feature/text_roberta.npy'.format(args.data_root, args.dataset)
        text_feature_dict = np.load(text_feature_path, allow_pickle=True).item()
        video_id_list = text_feature_dict.keys()

        self.text_dict = {}
        for video_id in video_id_list:
            self.text_dict[video_id] = torch.from_numpy(text_feature_dict[video_id]).to(torch.float32)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        video_name = key.split('/')[-1]
        video_file = self.video_dict[video_name]

        video = torch.from_numpy(video_file['features'][...].astype(np.float32)) # [T, 1024]
        text = self.text_dict[video_name] # [T, 1024]

        gtscore = video_file['gtscore'][...].astype(np.float32) # [T]
        change_points = video_file['change_points'][...].astype(np.int32) # [S, 2], S: number of segments, each row stores indices of a segment
        n_frames = video_file['n_frames'][...].astype(np.int32) # [N], N: number of frames, N = T * 15
        n_frame_per_seg = video_file['n_frame_per_seg'][...].astype(np.int32) # [S], indicates number of frames in each segment
        picks = video_file['picks'][...].astype(np.int32) # [T], posotions of subsampled frames in original video

        user_summary = np.zeros(0, dtype=np.float32)
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)

        gtscore -= gtscore.min()
        gtscore /= gtscore.max()

        keyshot_summ, gtscore_upsampled = get_keyshot_summ(gtscore, change_points, n_frames, n_frame_per_seg, picks)
        target = keyshot_summ[::15]

        video_cls_label = target
        video_loc_label = get_loc_label(target)
        video_ctr_label = get_ctr_label(target, video_loc_label)

        video_cls_label = torch.from_numpy(video_cls_label)
        video_loc_label = torch.from_numpy(video_loc_label)
        video_ctr_label = torch.from_numpy(video_ctr_label)

        num_frame = video.shape[0]
        num_sentence = text.shape[0]
        frame_sentence_ratio = int(math.ceil(num_frame / num_sentence))
        text_cls_label = np.zeros((num_sentence), dtype=bool)
        for j in range(num_sentence):
            start_frame = j * frame_sentence_ratio
            end_frame = min((j + 1) * frame_sentence_ratio, num_frame)
            if video_cls_label[start_frame: end_frame].any():
                text_cls_label[j] = True

        text_loc_label = get_loc_label(text_cls_label)
        text_ctr_label = get_ctr_label(text_cls_label, text_loc_label)

        text_cls_label = torch.from_numpy(text_cls_label)
        text_loc_label = torch.from_numpy(text_loc_label)
        text_ctr_label = torch.from_numpy(text_ctr_label)
        
        video_to_text_mask = torch.zeros((num_frame, num_sentence), dtype=torch.long)
        text_to_video_mask = torch.zeros((num_sentence, num_frame), dtype=torch.long)
        for j in range(num_sentence):
            start_frame = j * frame_sentence_ratio
            end_frame = min((j + 1) * frame_sentence_ratio, num_frame)
            video_to_text_mask[start_frame: end_frame, j] = 1
            text_to_video_mask[j, start_frame : end_frame] = 1

        mask_video = torch.ones(num_frame, dtype=torch.long)
        mask_text = torch.ones(num_sentence, dtype=torch.long)
        
        ratio = 0.15
        return video, text, mask_video, mask_text, video_cls_label, video_loc_label, video_ctr_label, text_cls_label, text_loc_label, text_ctr_label, \
            user_summary, n_frames, ratio, n_frame_per_seg, picks, change_points, video_to_text_mask, text_to_video_mask


def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

def my_collate_fn(batch):
    batched_output_list = []
    for i in range(len(batch[0])):
        batched_output = [item[i] for item in batch]
        batched_output_list.append(batched_output)
    return batched_output_list
