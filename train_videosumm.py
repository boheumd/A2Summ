import logging
import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

from models import *
from losses import *
from datasets import *
from utils import *

from helpers.bbox_helper import nms
from helpers.vsumm_helper import bbox2summary, get_summ_f1score

logger = logging.getLogger()

def train_videosumm(args, split, split_idx):
    batch_time = AverageMeter('time')
    data_time = AverageMeter('time')

    model = Model_VideoSumm(args=args)
    model = model.to(args.device)
    calc_contrastive_loss = Dual_Contrastive_Loss().to(args.device)

    parameters = [p for p in model.parameters() if p.requires_grad] + \
                    [p for p in calc_contrastive_loss.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs('{}/checkpoint'.format(args.model_dir), exist_ok=True)
    
    max_train_fscore = -1
    max_val_fscore = -1
    best_val_epoch = 0

    # model testing, load from checkpoint
    checkpoint_path = None
    if args.checkpoint and args.test:
        checkpoint_path = '{}/model_best_split{}.pt'.format(args.checkpoint, split_idx)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("load checkpoint from {}".format(checkpoint_path))
        model.load_state_dict(checkpoint['model_state_dict'])

    train_set = VideoSummDataset(keys=split['train_keys'], args=args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, 
                                                drop_last=False, pin_memory=True, 
                                                worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)
    val_set = VideoSummDataset(keys=split['test_keys'], args=args)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
                                                drop_last=False, pin_memory=True, 
                                                worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)
    if args.test:
        val_fscore = evaluate_videosumm(model, val_loader, args, epoch=0)
        logger.info(f'F-score: {val_fscore:.4f}')
        return val_fscore, best_val_epoch, max_train_fscore

    logger.info('\n' + str(model))

    for epoch in range(args.start_epoch, args.max_epoch):
        model.train()
        stats = AverageMeter('loss', 'cls_loss', 'loc_loss', 'ctr_loss', 'inter_contrastive_loss', 'intra_contrastive_loss')
        data_length = len(train_loader)
        end = time.time()
        for k, (video_list, text_list, mask_video_list, mask_text_list, \
                video_cls_label_list, video_loc_label_list, video_ctr_label_list, \
                text_cls_label_list, text_loc_label_list, text_ctr_label_list, \
                user_summary_list, n_frames_list, ratio_list, n_frame_per_seg_list, picks_list, change_points_list, \
                video_to_text_mask_list, text_to_video_mask_list) in enumerate(train_loader):
            data_time.update(time=time.time() - end)
                
            batch_size = len(video_list)

            video = pad_sequence(video_list, batch_first=True)
            text = pad_sequence(text_list, batch_first=True)

            mask_video = pad_sequence(mask_video_list, batch_first=True)
            mask_text = pad_sequence(mask_text_list, batch_first=True)
            
            video_cls_label = pad_sequence(video_cls_label_list, batch_first=True)
            video_loc_label = pad_sequence(video_loc_label_list, batch_first=True)
            video_ctr_label = pad_sequence(video_ctr_label_list, batch_first=True)

            text_cls_label = pad_sequence(text_cls_label_list, batch_first=True)
            text_loc_label = pad_sequence(text_loc_label_list, batch_first=True)
            text_ctr_label = pad_sequence(text_ctr_label_list, batch_first=True)
            
            for i in range(len(video_to_text_mask_list)):
                video_to_text_mask_list[i] = video_to_text_mask_list[i].to(args.device)
                text_to_video_mask_list[i] = text_to_video_mask_list[i].to(args.device)

            video, text = video.to(args.device), text.to(args.device)
            mask_video, mask_text = mask_video.to(args.device), mask_text.to(args.device)

            video_cls_label = video_cls_label.to(args.device) #[B, T]
            video_loc_label = video_loc_label.to(args.device) #[B, T, 2]
            video_ctr_label = video_ctr_label.to(args.device) #[B, T]

            text_cls_label = text_cls_label.to(args.device) #[B, T]
            text_loc_label = text_loc_label.to(args.device) #[B, T, 2]
            text_ctr_label = text_ctr_label.to(args.device) #[B, T]

            video_pred_cls, video_pred_loc, video_pred_ctr, text_pred_cls, text_pred_loc, text_pred_ctr, contrastive_pairs = \
                model(video=video, text=text,  mask_video=mask_video, mask_text=mask_text, 
                        video_label=video_cls_label, text_label=text_cls_label, 
                        video_to_text_mask_list=video_to_text_mask_list,  text_to_video_mask_list=text_to_video_mask_list)

            cls_loss = calc_cls_loss(video_pred_cls, video_cls_label.to(torch.long), mask=mask_video) + \
                        calc_cls_loss(text_pred_cls, text_cls_label.to(torch.long), mask=mask_text)

            loc_loss = calc_loc_loss(video_pred_loc, video_loc_label, video_cls_label) + \
                        calc_loc_loss(text_pred_loc, text_loc_label, text_cls_label)

            ctr_loss = calc_ctr_loss(video_pred_ctr, video_ctr_label, video_cls_label) + \
                        calc_ctr_loss(text_pred_ctr, text_ctr_label, text_cls_label)

            inter_contrastive_loss, intra_contrastive_loss = calc_contrastive_loss(contrastive_pairs)
            inter_contrastive_loss = inter_contrastive_loss * args.lambda_contrastive_inter
            intra_contrastive_loss = intra_contrastive_loss * args.lambda_contrastive_intra
            
            loss = cls_loss + loc_loss + ctr_loss + inter_contrastive_loss + intra_contrastive_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(loss=loss.item(), cls_loss=cls_loss.item(),
                         loc_loss=loc_loss.item(), ctr_loss=ctr_loss.item(), 
                         inter_contrastive_loss=inter_contrastive_loss.item(), 
                         intra_contrastive_loss=intra_contrastive_loss.item())
            
            batch_time.update(time=time.time() - end)
            end = time.time()

            if (k + 1) % args.print_freq == 0:
                logger.info(f'[Train] Epoch: {epoch+1}/{args.max_epoch} Iter: {k+1}/{data_length} '
                            f'Time: {batch_time.time:.3f} Data: {data_time.time:.3f} '
                            f'Loss: {stats.cls_loss:.4f}/{stats.loc_loss:.4f}/{stats.ctr_loss:.4f}/{stats.inter_contrastive_loss:.4f}/{stats.intra_contrastive_loss:.4f}/{stats.loss:.4f}')

        save_checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'max_val_fscore': max_val_fscore,
            'max_train_fscore': max_train_fscore,
        }

        if (epoch + 1) % args.eval_freq == 0:
            val_fscore = evaluate_videosumm(model, val_loader, args, epoch=epoch)
            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                best_val_epoch = epoch + 1
                torch.save(save_checkpoint, '{}/checkpoint/model_best_split{}.pt'.format(args.model_dir, split_idx))
            
            logger.info(f'[Eval]  Epoch: {epoch+1}/{args.max_epoch} F-score: {val_fscore:.4f}/{max_val_fscore:.4f}\n\n')

            args.writer.add_scalar(f'Split{split_idx}/Val/max_fscore', max_val_fscore, epoch+1)
            args.writer.add_scalar(f'Split{split_idx}/Val/fscore', val_fscore, epoch+1)

        args.writer.add_scalar(f'Split{split_idx}/Train/loss', stats.loss, epoch+1)
        args.writer.add_scalar(f'Split{split_idx}/Train/cls_loss', stats.cls_loss, epoch+1)
        args.writer.add_scalar(f'Split{split_idx}/Train/loc_loss', stats.loc_loss, epoch+1)
        args.writer.add_scalar(f'Split{split_idx}/Train/ctr_loss', stats.ctr_loss, epoch+1)
        args.writer.add_scalar(f'Split{split_idx}/Train/inter_contrastive_loss', stats.inter_contrastive_loss, epoch+1)
        args.writer.add_scalar(f'Split{split_idx}/Train/intra_contrastive_loss', stats.intra_contrastive_loss, epoch+1)

    return max_val_fscore, best_val_epoch, max_train_fscore


@torch.no_grad()
def evaluate_videosumm(model, val_loader, args, epoch=None):
    model.eval()
    stats = AverageMeter('fscore')

    data_length = len(val_loader)
    with torch.no_grad():
        for k, (video_list, text_list, mask_video_list, mask_text_list, \
                video_cls_label_list, video_loc_label_list, video_ctr_label_list, \
                text_cls_label_list, text_loc_label_list, text_ctr_label_list, \
                user_summary_list, n_frames_list, ratio_list, n_frame_per_seg_list, picks_list, change_points_list, \
                video_to_text_mask_list, text_to_video_mask_list) in enumerate(val_loader):

            batch_size = len(video_list)

            video = pad_sequence(video_list, batch_first=True)
            text = pad_sequence(text_list, batch_first=True)

            mask_video = pad_sequence(mask_video_list, batch_first=True)
            mask_text = pad_sequence(mask_text_list, batch_first=True)
            
            video_cls_label = pad_sequence(video_cls_label_list, batch_first=True)
            video_loc_label = pad_sequence(video_loc_label_list, batch_first=True)
            video_ctr_label = pad_sequence(video_ctr_label_list, batch_first=True)

            text_cls_label = pad_sequence(text_cls_label_list, batch_first=True)
            text_loc_label = pad_sequence(text_loc_label_list, batch_first=True)
            text_ctr_label = pad_sequence(text_ctr_label_list, batch_first=True)
            
            for i in range(len(video_to_text_mask_list)):
                video_to_text_mask_list[i] = video_to_text_mask_list[i].to(args.device)
                text_to_video_mask_list[i] = text_to_video_mask_list[i].to(args.device)

            video, text = video.to(args.device), text.to(args.device)
            mask_video, mask_text = mask_video.to(args.device), mask_text.to(args.device)

            video_cls_label = video_cls_label.to(args.device) #[B, T]
            video_loc_label = video_loc_label.to(args.device) #[B, T, 2]
            video_ctr_label = video_ctr_label.to(args.device) #[B, T]

            text_cls_label = text_cls_label.to(args.device) #[B, T]
            text_loc_label = text_loc_label.to(args.device) #[B, T, 2]
            text_ctr_label = text_ctr_label.to(args.device) #[B, T]

            pred_cls_batch, pred_bboxes_batch = model.predict(video=video, text=text, 
                                                            mask_video=mask_video, mask_text=mask_text, 
                                                            video_label=video_cls_label, text_label=text_cls_label, 
                                                            video_to_text_mask_list=video_to_text_mask_list, 
                                                            text_to_video_mask_list=text_to_video_mask_list) #[B, T], [B, T, 2]
            mask_video_bool = mask_video.cpu().numpy().astype(bool)
            

            for i in range(batch_size):
                video_length = np.sum(mask_video_bool[i])
                pred_cls = pred_cls_batch[i, mask_video_bool[i]] #[T]
                pred_bboxes = np.clip(pred_bboxes_batch[i, mask_video_bool[i]], 0, video_length).round().astype(np.int32) #[T, 2]

                pred_cls, pred_bboxes = nms(pred_cls, pred_bboxes, args.nms_thresh)
                pred_summ, pred_summ_upsampled, pred_score, pred_score_upsampled = bbox2summary(
                    video_length, pred_cls, pred_bboxes, change_points_list[i], n_frames_list[i], n_frame_per_seg_list[i], picks_list[i], proportion=ratio_list[i], seg_score_mode='mean')
                
                eval_metric = 'max' if args.dataset == 'SumMe' else 'avg'
                fscore = get_summ_f1score(pred_summ_upsampled, user_summary_list[i], eval_metric=eval_metric)

                stats.update(fscore=fscore)
            if (k + 1) % args.print_freq == 0:
                logger.info(f'[Eval]  Epoch: {epoch+1}/{args.max_epoch} Iter: {k+1}/{data_length} F-score: {stats.fscore:.4f}')
    return stats.fscore




