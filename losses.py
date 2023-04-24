import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def calc_cls_loss(pred: torch.Tensor,
                  target: torch.Tensor,
                  mask: torch.Tensor = None,
                  ) -> torch.Tensor:
    """Compute classification loss on both positive and negative samples.

    :param pred: Predicted class. Sized [B, N].
    :param target: Class target where 1 marks positive, and 0
        marks ignored. Sized [B, N].
    :param kind: Loss type. Choose from (focal, cross-entropy).
    :param mask: indicts the valid segments for each video
    :return: Scalar loss value.
    """

    pred = torch.sigmoid(pred)
    pred = torch.stack([1 - pred, pred], dim=-1)
    mask = mask.to(torch.bool)
    loss = focal_loss(pred, target, reduction='none')
    loss = loss[mask, :]
    loss = torch.mean(loss)
    return loss


def focal_loss(pred: torch.Tensor,
               target: torch.Tensor,
               alpha: float = 0.25,
               gamma: float = 2,
               reduction: str = 'sum'
               ) -> torch.Tensor:
    """Compute focal loss for binary classification.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    :param pred: Predicted confidence. Sized [B, N, D].
    :param target: Ground truth target. Sized [B, N].
    :param alpha: Alpha parameter in focal loss.
    :param gamma: Gamma parameter in focal loss.
    :param reduction: Aggregation type. Choose from (sum, mean, none).
    :return: Scalar loss value.
    """
    B, _, num_classes = pred.shape
    t = F.one_hot(target, num_classes)

    p_t = pred * t + (1 - pred) * (1 - t)
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    fl = -alpha_t * (1 - p_t).pow(gamma) * p_t.clamp(min=1e-7).log()

    ## TODO: update the sum to mean aross the batch axis
    if reduction == 'sum':
        fl = fl.sum()
    elif reduction == 'mean':
        fl = fl.mean()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Invalid reduction mode {reduction}')

    return fl


def iou_offset(offset_a: torch.Tensor,
               offset_b: torch.Tensor,
               eps: float = 1e-8
               ) -> torch.Tensor:
    """Compute IoU offsets between multiple offset pairs.

    :param offset_a: Offsets of N positions. Sized [N, 2].
    :param offset_b: Offsets of N positions. Sized [N, 2].
    :param eps: Small floating value to prevent division by zero.
    :return: IoU values of N positions. Sized [N].
    """
    left_a, right_a = offset_a[:, 0], offset_a[:, 1]
    left_b, right_b = offset_b[:, 0], offset_b[:, 1]

    length_a = left_a + right_a
    length_b = left_b + right_b

    intersect = torch.min(left_a, left_b) + torch.min(right_a, right_b)
    intersect[intersect < 0] = 0
    union = length_a + length_b - intersect
    union[union <= 0] = eps

    iou = intersect / union
    return iou


def calc_loc_loss(pred_loc_batch: torch.Tensor,
                  test_loc_batch: torch.Tensor,
                  cls_label: torch.Tensor,
                  kind: str = 'soft-iou',
                  eps: float = 1e-8
                  ) -> torch.Tensor:
    """Compute soft IoU loss for regression only on positive samples.

    :param pred_loc_batch: Predicted offsets. Sized [B, N, 2].
    :param test_loc_batch: Ground truth offsets. Sized [B, N, 2].
    :param cls_label: Class label specifying positive samples.
    :param kind: Loss type. Choose from (soft-iou, smooth-l1).
    :param eps: Small floating value to prevent division by zero.
    :return: Scalar loss value.
    """
    cls_label = cls_label.to(torch.bool)
    batch_size = cls_label.shape[0]
    
    loss_sum = 0
    for i in range(batch_size):
        pred_loc = pred_loc_batch[i, cls_label[i]]
        test_loc = test_loc_batch[i, cls_label[i]]

        if kind == 'soft-iou':
            iou = iou_offset(pred_loc, test_loc)
            loss = -torch.log(iou + eps).mean()
        elif kind == 'smooth-l1':
            loss = F.smooth_l1_loss(pred_loc, test_loc)
        else:
            raise ValueError(f'Invalid loss type {kind}')
        loss_sum += loss

    loss = loss_sum / batch_size
    return loss


def calc_ctr_loss(pred_batch, test_batch, pos_mask):
    pos_mask = pos_mask.to(torch.bool) #[B, T]
    batch_size = pos_mask.shape[0]

    loss_sum = 0
    for i in range(batch_size):
        pred = pred_batch[i, pos_mask[i]] #[M]
        test = test_batch[i, pos_mask[i]] #[M]
        loss = F.binary_cross_entropy(pred, test)
        loss_sum += loss
        
    loss = loss_sum / batch_size
    return loss

@torch.no_grad()
def calc_text_rouge(article_sentence_list, highlight_list, selected_sentence_index_list, dataset=None, rouge=None):
    batch_size = len(selected_sentence_index_list)

    R1_sum = 0
    R2_sum = 0
    RL_sum = 0
    for i in range(batch_size):
        sorted_index_list = sorted(selected_sentence_index_list[i])
        selected_sentence_list = []
        for selected_sentence_index in sorted_index_list:
            selected_sentence_list.append(article_sentence_list[i][selected_sentence_index])
        
        evaluated_sentence = ' '.join(selected_sentence_list)
        if isinstance(highlight_list[i], list):
            reference_sentence = ' '.join(highlight_list[i])
        elif isinstance(highlight_list[i], str):
            reference_sentence = highlight_list[i]
        scores = rouge.score(evaluated_sentence, reference_sentence)
        R1_sum += scores['rouge1'][2]
        R2_sum += scores['rouge2'][2]
        RL_sum += scores['rougeLsum'][2]
    
    R1_mean = R1_sum / batch_size
    R2_mean = R2_sum / batch_size
    RL_mean = RL_sum / batch_size
    return R1_mean, R2_mean, RL_mean

@torch.no_grad()
def calc_video_cos(video, gt_summ, keyframe_index_list, mask_video_summ=None, dataset=None):
    batch_size = len(keyframe_index_list)
    gt_summ = F.normalize(gt_summ, dim=-1)

    cos_sim_sum = 0
    for i in range(batch_size):
        if dataset == 'Daily_Mail':
            pred_summ = video[i][keyframe_index_list[i]]
            pred_summ = F.normalize(pred_summ, dim=1)
            sim_mat = gt_summ[i, mask_video_summ[i]] @ pred_summ.permute(1, 0)
            sim_mat = sim_mat.detach().cpu().numpy()
        elif dataset == 'BLiSS':
            pred_summ = F.normalize(video[i], dim=1)
            sim_mat = gt_summ[i, mask_video_summ[i]] @ pred_summ.permute(1, 0)
            sim_mat = sim_mat - torch.min(sim_mat)
            sim_mat = sim_mat / torch.max(sim_mat).clamp(min=1e-6)
            sim_mat = sim_mat[:, keyframe_index_list[i]]
            sim_mat = sim_mat.detach().cpu().numpy()

        # select the largest-K pairwise cosine simialrity (K = num_key_frame)
        num_key_frame = len(keyframe_index_list[i])
        match_mat = np.zeros((num_key_frame, num_key_frame), dtype=int)
        sorted_index = np.dstack(np.unravel_index(np.argsort(-sim_mat.ravel()), sim_mat.shape))[0] #[N*N, 2]
        select_key_frame_count = 0
        for j in range(sorted_index.shape[0]):
            m, n = sorted_index[j]
            if not match_mat[m, :].any() and not match_mat[:, n].any():
                match_mat[m, n] = 1
                select_key_frame_count += 1
            if select_key_frame_count >= num_key_frame:
                break
        
        cos_sim = np.sum(sim_mat * match_mat) / np.sum(match_mat)
        cos_sim_sum += cos_sim
    
    cos_sim_mean = cos_sim_sum / batch_size
    return cos_sim_mean


class NCE(nn.Module):
    def __init__(self):
        super(NCE, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, q, k, neg, device='cuda:0'):
        q = F.normalize(q, dim=1) #[1, C]
        k = F.normalize(k, dim=1) #[1, C]
        neg = F.normalize(neg, dim=1) #[T, C]
        l_pos = q @ k.T #[1, 1]
        l_neg = q @ neg.T #[1, T]
        logits = torch.cat([l_pos, l_neg], dim=1) #[1, 1 + T]
        logits *= self.logit_scale #[1, 1 + T]
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        loss = F.cross_entropy(logits, labels)
        return loss


class Dual_Contrastive_Loss(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.logit_scale_inter = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.NCE_video = NCE()
        self.NCE_text = NCE()
        
    def forward(self, contrastive_pairs):
        if len(contrastive_pairs) == 0:
            return torch.zeros(1).cuda(), torch.zeros(1).cuda()
        cls_video = contrastive_pairs['cls_video']
        cls_text = contrastive_pairs['cls_text']
        key_video_list = contrastive_pairs['key_video_list']
        nonkey_video_list = contrastive_pairs['nonkey_video_list']
        key_text_list = contrastive_pairs['key_text_list']
        nonkey_text_list = contrastive_pairs['nonkey_text_list']
            
        B = cls_video.shape[0]
        device = cls_video.device
        
        ########## Inter-Sample Contrastive Loss ##########
        cls_video = F.normalize(cls_video.squeeze(1), dim=1) #[B, C]
        cls_text = F.normalize(cls_text.squeeze(1), dim=1) #[B, C]

        # cosine similarity as logits
        logits_per_video = self.logit_scale_inter.exp() * cls_video @ cls_text.t() #[B, B]
        logits_per_text = logits_per_video.t() #[B, B]
        
        target = torch.arange(B).to(device)
        inter_contrastive_loss_video = F.cross_entropy(logits_per_video, target)
        inter_contrastive_loss_text = F.cross_entropy(logits_per_text, target)
        inter_contrastive_loss = (inter_contrastive_loss_video + inter_contrastive_loss_text) / 2
        
        ########## Intra-Sample Contrastive Loss ##########
        intra_contrastive_loss = 0
        for i in range(B):
            intra_contrastive_loss_video = self.NCE_video(
                torch.mean(key_video_list[i], dim=0, keepdim=True),
                torch.mean(key_text_list[i], dim=0, keepdim=True),
                nonkey_video_list[i],
                device
            )
            intra_contrastive_loss_text = self.NCE_text(
                torch.mean(key_text_list[i], dim=0, keepdim=True),
                torch.mean(key_video_list[i], dim=0, keepdim=True),
                nonkey_text_list[i],
                device
            )
            intra_contrastive_loss += (intra_contrastive_loss_video + intra_contrastive_loss_text) / 2
        intra_contrastive_loss /= B
        
        return inter_contrastive_loss, intra_contrastive_loss

