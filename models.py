import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from scipy import ndimage
from helpers.bbox_helper import offset2bbox


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dims,
                 k_dims=None,
                 v_dims=None,
                 h_dims=None,
                 o_dims=None,
                 heads=8,
                 p=0.1,
                 bias=True):
        super(MultiHeadAttention, self).__init__()

        self._q_dims = dims
        self._k_dims = k_dims or dims
        self._v_dims = v_dims or dims
        self._h_dims = h_dims or dims
        self._o_dims = o_dims or dims
        self._heads = heads
        self._p = p
        self._bias = bias
        self._head_dims = self._h_dims // heads

        self.q = nn.Linear(self._q_dims, self._h_dims, bias=bias)
        self.k = nn.Linear(self._k_dims, self._h_dims, bias=bias)
        self.v = nn.Linear(self._v_dims, self._h_dims, bias=bias)
        self.m = nn.Linear(self._h_dims, self._o_dims, bias=bias)

        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)

        self.reset_parameters()

    def __repr__(self):
        return ('{}(q_dims={}, k_dims={}, v_dims={}, h_dims={}, o_dims={}, '
                'heads={}, p={}, bias={})'.format(self.__class__.__name__,
                                                  self._q_dims, self._k_dims,
                                                  self._v_dims, self._h_dims,
                                                  self._o_dims, self._heads,
                                                  self._p, self._bias))

    def reset_parameters(self):
        for m in (self.q, self.k, self.v, self.m):
            nn.init.xavier_normal_(m.weight, gain=1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, q, k=None, v=None, mask=None):
        v = v if torch.is_tensor(v) else k if torch.is_tensor(k) else q
        k = k if torch.is_tensor(k) else q

        q = self.q(q).transpose(0, 1).contiguous()
        k = self.k(k).transpose(0, 1).contiguous()
        v = self.v(v).transpose(0, 1).contiguous()

        b = q.size(1) * self._heads

        q = q.view(-1, b, self._head_dims).transpose(0, 1)
        k = k.view(-1, b, self._head_dims).transpose(0, 1)
        v = v.view(-1, b, self._head_dims).transpose(0, 1)

        att = torch.bmm(q, k.transpose(1, 2)) / self._head_dims**0.5

        if mask is not None:
            mask = torch.where(mask > 0, .0, float('-inf'))
            mask = mask.repeat_interleave(self._heads, dim=0)
            att += mask

        att = att.softmax(-1)

        if self.drop1 is not None:
            att = self.drop1(att)

        m = torch.bmm(att, v).transpose(0, 1).contiguous()
        m = m.view(m.size(0), -1, self._h_dims).transpose(0, 1)
        m = self.m(m)

        if self.drop2 is not None:
            m = self.drop2(m)

        return m

class FFN(nn.Module):
    def __init__(self, num_input, p=0.1, ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(num_input, num_input * ratio)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p)
        self.fc2 = nn.Linear(num_input * ratio, num_input)
        self.drop2 = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MultiWayTransformer(nn.Module):
    def __init__(self, num_hidden, dropout_attn=0.1):
        super().__init__()
        self.norm1_fused = nn.LayerNorm(num_hidden)
        self.attn_fusion = MultiHeadAttention(num_hidden, p=dropout_attn)

        self.norm2_video = nn.LayerNorm(num_hidden)
        self.ffn_video = FFN(num_hidden, p=dropout_attn, ratio=4)

        self.norm2_text = nn.LayerNorm(num_hidden)
        self.ffn_text = FFN(num_hidden, p=dropout_attn, ratio=4)
    
    def forward(self, fused, mask_fused, N_video, N_text):
        residual = fused

        fused = self.norm1_fused(fused)
        fused = self.attn_fusion(fused, fused, fused, mask=mask_fused)
        residual = residual + fused

        residual_video, residual_text = torch.split(residual, [N_video, N_text], dim=1)

        video = self.norm2_video(residual_video)
        video = self.ffn_video(video)
        residual_video = residual_video + video

        text = self.norm2_text(residual_text)
        text = self.ffn_text(text)
        residual_text = residual_text + text

        return residual_video, residual_text

# For Daily_Mail/CNN datasets
class Model_MSMO(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        num_input_video = args.num_input_video
        num_input_text = args.num_input_text
        num_hidden = args.num_hidden
        
        self.ratio = args.ratio

        self.proj_fc_video = nn.Sequential(
                                nn.Linear(num_input_video, num_hidden, bias=True),
                                nn.Dropout(args.dropout_video),
                            )
        self.proj_fc_text = nn.Sequential(
                                nn.Linear(num_input_text, num_hidden, bias=True),
                                nn.Dropout(args.dropout_text),
                            )

        self.pos_embed_video = nn.Parameter(torch.zeros(1, 5000, num_hidden))
        self.pos_embed_text = nn.Parameter(torch.zeros(1, 5000, num_hidden))
        self.type_video = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.type_text = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.cls_token_text = nn.Parameter(torch.zeros(1, 1, num_hidden))
        
        self.cls_mask_video = torch.ones([1, 1])
        self.cls_mask_text = torch.ones([1, 1])

        self.multiway_list = nn.ModuleList([MultiWayTransformer(num_hidden, dropout_attn=args.dropout_attn)] * args.num_layers)

        self.norm_video = nn.LayerNorm(num_hidden)
        self.norm_text = nn.LayerNorm(num_hidden)

        self.fc_video = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(args.dropout_fc),
            nn.Linear(num_hidden, 1),
        )
        self.fc_text = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(args.dropout_fc),
            nn.Linear(num_hidden, 1),
        )

        self.num_layers = args.num_layers
        
        nn.init.trunc_normal_(self.pos_embed_video, std=.02)
        nn.init.trunc_normal_(self.pos_embed_text, std=.02)
        nn.init.trunc_normal_(self.type_video, std=.02)
        nn.init.trunc_normal_(self.type_text, std=.02)
        nn.init.trunc_normal_(self.cls_token_video, std=.02)
        nn.init.trunc_normal_(self.cls_token_text, std=.02)

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def select_contrastive_embedding(self, score, embedding, mask, label):
        B = score.shape[0]

        key_embedding_list = []
        nonkey_embedding_list = []
        for i in range(B):
            length = torch.sum(mask[i].to(torch.long))
            key_embedding_num = max(1, length // self.ratio)
            nonkey_embedding_num = max(1, length // self.ratio)
            
            key_embedding_index = label[i].to(torch.bool)
            key_embedding = embedding[i, key_embedding_index]
            
            key_embedding_index_expand = ndimage.binary_dilation(label[i].cpu().detach().numpy(), iterations=4).astype(np.int32)
            key_embedding_index_expand = torch.from_numpy(key_embedding_index_expand)
            
            score_i = score[i, :length]
            score_i = F.softmax(score_i, dim=-1)
        
            _, idx_DESC = score_i.sort(descending=True)
            
            non_key_embedding_index = []
            for j in range(idx_DESC.shape[0]):
                if key_embedding_index_expand[idx_DESC[j]] == 0:
                    non_key_embedding_index.append(idx_DESC[j].item())
                if len(non_key_embedding_index) >= nonkey_embedding_num:
                    break
            
            if len(non_key_embedding_index) == 0:
                non_key_embedding_index.append(idx_DESC[-1])
            
            nonkey_embedding = embedding[i, non_key_embedding_index]

            key_embedding_list.append(key_embedding)
            nonkey_embedding_list.append(nonkey_embedding)
        return key_embedding_list, nonkey_embedding_list


    def forward(self, **kwargs):
        video = kwargs['video']
        text = kwargs['text']
        mask_video = kwargs['mask_video']
        mask_text = kwargs['mask_text']
        video_label = kwargs['video_label']
        text_label = kwargs['text_label']

        B = video.shape[0]
        video = self.proj_fc_video(video)
        text = self.proj_fc_text(text)

        # prepend the [CLSV] and [CLST] tokens to the video and text feature sequences
        video = torch.cat([self.cls_token_video.expand(B, -1, -1), video], dim=1)
        text = torch.cat([self.cls_token_text.expand(B, -1, -1), text], dim=1)
        mask_video = torch.cat([self.cls_mask_video.expand(B, -1).to(mask_video), mask_video], dim=1)
        mask_text = torch.cat([self.cls_mask_text.expand(B, -1).to(mask_text), mask_text], dim=1)

        # add positional embedding
        B, N_video, C = video.shape
        B, N_text, C = text.shape
        video = video + self.pos_embed_video[:, :N_video, :] + self.type_video
        text = text + self.pos_embed_text[:, :N_text, :] + self.type_text

        fused = torch.cat([video, text], dim=1)
        mask_fused = torch.cat([mask_video, mask_text], dim=1) #[B, N_video+N_text]
        mask_fused = mask_fused.unsqueeze(1).expand(-1, N_video+N_text, -1) #[B, N_video+N_text, N_video+N_text]
        # multiway transformer layers
        for i in range(self.num_layers):
            video, text = self.multiway_list[i](fused, mask_fused, N_video, N_text)
            fused = torch.cat([video, text], dim=1)
        video = self.norm_video(video)
        text = self.norm_text(text)

        cls_video, video = torch.split(video, [1, N_video-1], dim=1)
        cls_text, text = torch.split(text, [1, N_text-1], dim=1)

        pred_video = self.fc_video(video).squeeze(-1) #[B, N]
        pred_text = self.fc_text(text).squeeze(-1) #[B, N]

        # select contrastive pairs for the intra-sample constrastive loss
        key_video_list, nonkey_video_list = self.select_contrastive_embedding(pred_video, video, mask_video[:, 1:], video_label)
        key_text_list, nonkey_text_list = self.select_contrastive_embedding(pred_text, text, mask_text[:, 1:], text_label)

        contrastive_pairs = {
            'key_video_list': key_video_list,
            'nonkey_video_list': nonkey_video_list,
            'key_text_list': key_text_list,
            'nonkey_text_list': nonkey_text_list,
            'cls_video': cls_video,
            'cls_text': cls_text,
        }
        
        return pred_video, pred_text, contrastive_pairs

# For BLiSS dataset
class Model_BLiSS(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        num_input_video = args.num_input_video
        num_input_text = args.num_input_text
        num_hidden = args.num_hidden
        
        self.ratio = args.ratio

        self.proj_fc_video = nn.Sequential(
                                nn.Linear(num_input_video, num_hidden, bias=True),
                                nn.Dropout(args.dropout_video),
                            )
        self.proj_fc_text = nn.Sequential(
                                nn.Linear(num_input_text, num_hidden, bias=True),
                                nn.Dropout(args.dropout_text),
                            )

        self.pos_embed_video = nn.Parameter(torch.zeros(1, 5000, num_hidden))
        self.pos_embed_text = nn.Parameter(torch.zeros(1, 5000, num_hidden))
        self.pos_embed_segment = nn.Parameter(torch.zeros(1, 200, num_hidden))
        self.type_video = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.type_text = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.cls_token_text = nn.Parameter(torch.zeros(1, 1, num_hidden))
        
        self.cls_mask_video = torch.ones([1, 1])
        self.cls_mask_text = torch.ones([1, 1])

        self.multiway_list = nn.ModuleList([MultiWayTransformer(num_hidden, dropout_attn=args.dropout_attn)] * args.num_layers)

        self.norm_video = nn.LayerNorm(num_hidden)
        self.norm_text = nn.LayerNorm(num_hidden)

        self.fc_video = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(args.dropout_fc),
            nn.Linear(num_hidden, 1),
        )
        self.fc_text = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(args.dropout_fc),
            nn.Linear(num_hidden, 1),
        )

        self.num_layers = args.num_layers
        
        nn.init.trunc_normal_(self.pos_embed_video, std=.02)
        nn.init.trunc_normal_(self.pos_embed_text, std=.02)
        nn.init.trunc_normal_(self.pos_embed_segment, std=.02)
        nn.init.trunc_normal_(self.type_video, std=.02)
        nn.init.trunc_normal_(self.type_text, std=.02)
        nn.init.trunc_normal_(self.cls_token_video, std=.02)
        nn.init.trunc_normal_(self.cls_token_text, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def select_contrastive_embedding(self, score, embedding, mask, label):
        B = score.shape[0]
            
        key_embedding_list = []
        nonkey_embedding_list = []
        for i in range(B):
            length = torch.sum(mask[i].to(torch.long))
            key_embedding_num = max(1, length // self.ratio)
            nonkey_embedding_num = max(1, length // self.ratio)
            
            key_embedding_index = label[i].to(torch.bool)
            key_embedding = embedding[i, key_embedding_index]

            key_embedding_index_expand = ndimage.binary_dilation(label[i].cpu().detach().numpy(), iterations=4).astype(np.int32)
            key_embedding_index_expand = torch.from_numpy(key_embedding_index_expand)
            
            score_i = score[i, :length]
            score_i = F.softmax(score_i, dim=-1)
        
            _, idx_DESC = score_i.sort(descending=True)
            
            non_key_embedding_index = []
            for j in range(idx_DESC.shape[0]):
                # not key_embedding_index
                if key_embedding_index_expand[idx_DESC[j]] == 0:
                    non_key_embedding_index.append(idx_DESC[j].item())
                if len(non_key_embedding_index) >= nonkey_embedding_num:
                    break
            
            nonkey_embedding = embedding[i, non_key_embedding_index]

            key_embedding_list.append(key_embedding)
            nonkey_embedding_list.append(nonkey_embedding)
        return key_embedding_list, nonkey_embedding_list
            

    def forward(self, **kwargs):
        video = kwargs['video']
        text = kwargs['text']
        mask_video = kwargs['mask_video']
        mask_text = kwargs['mask_text']
        video_label = kwargs['video_label']
        text_label = kwargs['text_label']
        video_to_text_mask_list = kwargs['video_to_text_mask_list'] # time correspondence mask between video and text
        text_to_video_mask_list = kwargs['text_to_video_mask_list'] # time correspondence mask between text and video

        B = video.shape[0]
        video = self.proj_fc_video(video)
        text = self.proj_fc_text(text)

        # prepend the [CLSV] and [CLST] tokens to the video and text feature sequences
        video = torch.cat([self.cls_token_video.expand(B, -1, -1), video], dim=1)
        text = torch.cat([self.cls_token_text.expand(B, -1, -1), text], dim=1)
        mask_video = torch.cat([self.cls_mask_video.expand(B, -1).to(mask_video), mask_video], dim=1) #[B, N_video]
        mask_text = torch.cat([self.cls_mask_text.expand(B, -1).to(mask_text), mask_text], dim=1) #[B, N_text]

        # add positional embedding and segment embedding
        B, N_video, C = video.shape
        B, N_text, C = text.shape
        video = video + self.pos_embed_video[:, :N_video, :] + self.type_video
        text = text + self.pos_embed_text[:, :N_text, :] + self.type_text + self.pos_embed_segment[:, :N_text, :]

        # generate global attention mask with time correspondence
        # N_video: 1 ([CLSV] token) + number of video frames with padding (since batchsize > 1)
        # N_text: 1 ([CLST] token) + number of text sentences with padding (since batchsize > 1)
        # N_video_valid: number of actual video frames for each data sample
        # N_text_valid: number of actual text frames for each data sample
        mask_fused = torch.zeros((B, N_video+N_text, N_video+N_text), dtype=torch.long).to(mask_video)
        for i in range(B):
            mask_fused[i, :N_video, :N_video] = mask_video[i].view(1, N_video).expand(N_video, -1) #[N_video, N_video]
            mask_fused[i, N_video:, N_video:] = mask_text[i].view(1, N_text).expand(N_text, -1) #[N_text, N_text]
            
            N_video_valid, N_text_valid = video_to_text_mask_list[i].shape #[N_video_valid, N_text_valid]
            mask_fused[i, 1:1+N_video_valid, 1+N_video:1+N_video+N_text_valid] = video_to_text_mask_list[i] #[N_video_valid, N_text_valid] not consider the [CLS] token
            mask_fused[i, 1+N_video:1+N_video+N_text_valid:, 1:1+N_video_valid] = text_to_video_mask_list[i] #[N_text-1, N_video-1] not consider the [CLS] token
            pos_embed_segment_video = video_to_text_mask_list[i].to(torch.float32) @ self.pos_embed_segment[0, :N_text_valid, :] # [N_video_valid, C]
            video[i, 1:1+N_video_valid, :] = video[i, 1:1+N_video_valid, :] + pos_embed_segment_video

        # multiway transformer layers
        fused = torch.cat([video, text], dim=1)
        for i in range(self.num_layers):
            video, text = self.multiway_list[i](fused, mask_fused, N_video, N_text)
            fused = torch.cat([video, text], dim=1)
        video = self.norm_video(video)
        text = self.norm_text(text)

        cls_video, video = torch.split(video, [1, N_video-1], dim=1)
        cls_text, text = torch.split(text, [1, N_text-1], dim=1)

        pred_video = self.fc_video(video).squeeze(-1) #[B, N]
        pred_text = self.fc_text(text).squeeze(-1) #[B, N]

        # select contrastive pairs for the intra-sample constrastive loss
        key_video_list, nonkey_video_list = self.select_contrastive_embedding(pred_video, video, mask_video[:, 1:], video_label)
        key_text_list, nonkey_text_list = self.select_contrastive_embedding(pred_text, text, mask_text[:, 1:], text_label)

        contrastive_pairs = {
            'key_video_list': key_video_list,
            'nonkey_video_list': nonkey_video_list,
            'key_text_list': key_text_list,
            'nonkey_text_list': nonkey_text_list,
            'cls_video': cls_video,
            'cls_text': cls_text,
        }
        
        return pred_video, pred_text, contrastive_pairs

# For SumMe/TVSum datasets
class Model_VideoSumm(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        num_input_video = args.num_input_video
        num_input_text = args.num_input_text
        num_hidden = args.num_hidden
        
        self.ratio = args.ratio

        self.proj_fc_video = nn.Sequential(
                                nn.Linear(num_input_video, num_hidden, bias=True),
                                nn.Dropout(args.dropout_video),
                            )
        self.proj_fc_text = nn.Sequential(
                                nn.Linear(num_input_text, num_hidden, bias=True),
                                nn.Dropout(args.dropout_text),
                            )

        self.pos_embed_video = nn.Parameter(torch.zeros(1, 5000, num_hidden))
        self.pos_embed_text = nn.Parameter(torch.zeros(1, 5000, num_hidden))
        self.pos_embed_segment = nn.Parameter(torch.zeros(1, 5000, num_hidden))
        self.type_video = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.type_text = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.cls_token_text = nn.Parameter(torch.zeros(1, 1, num_hidden))
        
        self.cls_mask_video = torch.ones([1, 1])
        self.cls_mask_text = torch.ones([1, 1])

        self.multiway_list = nn.ModuleList([MultiWayTransformer(num_hidden, dropout_attn=args.dropout_attn)] * args.num_layers)

        self.norm_video = nn.LayerNorm(num_hidden)
        self.norm_text = nn.LayerNorm(num_hidden)

        self.fc_video = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(args.dropout_fc),
            nn.LayerNorm(num_hidden),
        )
        self.fc_video_cls = nn.Linear(num_hidden, 1)
        self.fc_video_loc = nn.Linear(num_hidden, 2)
        self.fc_video_ctr = nn.Linear(num_hidden, 1)

        self.fc_text = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(args.dropout_fc),
            nn.LayerNorm(num_hidden),
        )
        self.fc_text_cls = nn.Linear(num_hidden, 1)
        self.fc_text_loc = nn.Linear(num_hidden, 2)
        self.fc_text_ctr = nn.Linear(num_hidden, 1)

        self.num_layers = args.num_layers
        
        nn.init.trunc_normal_(self.pos_embed_video, std=.02)
        nn.init.trunc_normal_(self.pos_embed_text, std=.02)
        nn.init.trunc_normal_(self.pos_embed_segment, std=.02)
        nn.init.trunc_normal_(self.type_video, std=.02)
        nn.init.trunc_normal_(self.type_text, std=.02)
        nn.init.trunc_normal_(self.cls_token_video, std=.02)
        nn.init.trunc_normal_(self.cls_token_text, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def select_contrastive_embedding(self, score, embedding, mask, label):
        B = score.shape[0]
            
        key_embedding_list = []
        nonkey_embedding_list = []
        for i in range(B):
            length = torch.sum(mask[i].to(torch.long))
            key_embedding_num = max(1, length // self.ratio)
            nonkey_embedding_num = max(1, length // self.ratio)
            
            key_embedding_index = label[i].to(torch.bool)
            key_embedding = embedding[i, key_embedding_index]

            key_embedding_index_expand = ndimage.binary_dilation(label[i].cpu().detach().numpy(), iterations=4).astype(np.int32)
            key_embedding_index_expand = torch.from_numpy(key_embedding_index_expand)
            
            score_i = score[i, :length]
            score_i = F.softmax(score_i, dim=-1)
        
            _, idx_DESC = score_i.sort(descending=True)
            
            non_key_embedding_index = []
            for j in range(idx_DESC.shape[0]):
                if key_embedding_index_expand[idx_DESC[j]] == 0:
                    non_key_embedding_index.append(idx_DESC[j].item())
                if len(non_key_embedding_index) >= nonkey_embedding_num:
                    break
            
            nonkey_embedding = embedding[i, non_key_embedding_index]

            key_embedding_list.append(key_embedding)
            nonkey_embedding_list.append(nonkey_embedding)
        return key_embedding_list, nonkey_embedding_list
            
    def forward(self, **kwargs):
        video = kwargs['video']
        text = kwargs['text']
        mask_video = kwargs['mask_video']
        mask_text = kwargs['mask_text']
        video_label = kwargs['video_label']
        text_label = kwargs['text_label']
        video_to_text_mask_list = kwargs['video_to_text_mask_list'] # time correspondence mask between video and text
        text_to_video_mask_list = kwargs['text_to_video_mask_list'] # time correspondence mask between text and video
    
        B = video.shape[0]
        video = self.proj_fc_video(video)
        text = self.proj_fc_text(text)
        residual_video = video
        residual_text = text

        # prepend the [CLSV] and [CLST] tokens to the video and text feature sequences
        video = torch.cat([self.cls_token_video.expand(B, -1, -1), video], dim=1)
        text = torch.cat([self.cls_token_text.expand(B, -1, -1), text], dim=1)
        mask_video = torch.cat([self.cls_mask_video.expand(B, -1).to(mask_video), mask_video], dim=1) #[B, N_video]
        mask_text = torch.cat([self.cls_mask_text.expand(B, -1).to(mask_text), mask_text], dim=1) #[B, N_text]

        # add positional embedding and segment embedding
        B, N_video, C = video.shape
        B, N_text, C = text.shape
        video = video + self.pos_embed_video[:, :N_video, :] + self.type_video
        text = text + self.pos_embed_text[:, :N_text, :] + self.type_text + self.pos_embed_segment[:, :N_text, :]

        # generate global attention mask with time correspondence
        # N_video: 1 ([CLSV] token) + number of video frames with padding (since batchsize > 1)
        # N_text: 1 ([CLST] token) + number of text sentences with padding (since batchsize > 1)
        # N_video_valid: number of actual video frames for each data sample
        # N_text_valid: number of actual text frames for each data sample
        mask_fused = torch.zeros((B, N_video+N_text, N_video+N_text), dtype=torch.long).to(mask_video) # [B, N_video+N_text, N_video+N_text]
        for i in range(B):
            mask_fused[i, :N_video, :N_video] = mask_video[i].view(1, N_video).expand(N_video, -1) #[N_video, N_video]
            mask_fused[i, N_video:, N_video:] = mask_text[i].view(1, N_text).expand(N_text, -1) #[N_text, N_text]
            
            N_video_valid, N_text_valid = video_to_text_mask_list[i].shape #[N_video_valid, N_text_valid]
            mask_fused[i, 1:1+N_video_valid, 1+N_video:1+N_video+N_text_valid] = video_to_text_mask_list[i] #[N_video_valid, N_text_valid] not consider the [CLS] token
            mask_fused[i, 1+N_video:1+N_video+N_text_valid:, 1:1+N_video_valid] = text_to_video_mask_list[i] #[N_text-1, N_video-1] not consider the [CLS] token
            pos_embed_segment_video = video_to_text_mask_list[i].to(torch.float32) @ self.pos_embed_segment[0, :N_text_valid, :] # [N_video_valid, C]
            video[i, 1:1+N_video_valid, :] = video[i, 1:1+N_video_valid, :] + pos_embed_segment_video

        # multiway transformer layers
        fused = torch.cat([video, text], dim=1)
        for i in range(self.num_layers):
            video, text = self.multiway_list[i](fused, mask_fused, N_video, N_text)
            fused = torch.cat([video, text], dim=1)
        cls_video, video = torch.split(video, [1, N_video-1], dim=1)
        cls_text, text = torch.split(text, [1, N_text-1], dim=1)
        
        video = self.norm_video(residual_video + video)
        text = self.norm_text(residual_text + text)
        video = self.fc_video(video)
        text = self.fc_text(text)

        pred_video_cls = self.fc_video_cls(video).squeeze(-1) #[B, N]
        pred_text_cls = self.fc_text_cls(text).squeeze(-1) #[B, N]
        pred_video_loc = self.fc_video_loc(video).exp() #[B, N, 2]
        pred_video_ctr = self.fc_video_ctr(video).squeeze(-1).sigmoid() #[B, N]
        pred_text_loc = self.fc_text_loc(text).exp() #[B, N, 2]
        pred_text_ctr = self.fc_text_ctr(text).squeeze(-1).sigmoid() #[B, N]

        # select contrastive pairs for the intra-sample constrastive loss
        key_video_list, nonkey_video_list = self.select_contrastive_embedding(pred_video_cls, video, mask_video[:, 1:], video_label)
        key_text_list, nonkey_text_list = self.select_contrastive_embedding(pred_text_cls, text, mask_text[:, 1:], text_label)
        
        contrastive_pairs = {
            'key_video_list': key_video_list,
            'nonkey_video_list': nonkey_video_list,
            'key_text_list': key_text_list,
            'nonkey_text_list': nonkey_text_list,
            'cls_video': cls_video,
            'cls_text': cls_text,
        }

        return pred_video_cls, pred_video_loc, pred_video_ctr, pred_text_cls, pred_text_loc, pred_text_ctr, contrastive_pairs

    def predict(self, **kwargs):
        pred_video_cls, pred_video_loc, pred_video_ctr, pred_text_cls, pred_text_loc, pred_text_ctr, contrastive_pairs = self(**kwargs)

        pred_video_cls = pred_video_cls.sigmoid() 
        pred_video_cls *= pred_video_ctr
        pred_video_cls /= pred_video_cls.max() + 1e-8

        pred_video_cls = pred_video_cls.cpu().numpy()
        pred_video_loc = pred_video_loc.cpu().numpy()

        pred_video_bboxes = offset2bbox(pred_video_loc)
        return pred_video_cls, pred_video_bboxes
