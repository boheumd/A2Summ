import json 
import argparse
import numpy as np

_DATASET_HYPER_PARAMS = {
    "SumMe":{
        "lr":1e-3,
        "weight_decay": 1e-3,
        "max_epoch": 300,
        "batch_size": 4,
        "seed": 666,

        "num_input_video": 1024,
        "num_input_text": 768,
        "num_hidden": 128,
        "num_layers": 2,

        "dropout_video": 0.1,
        "dropout_text": 0.1,
        "dropout_attn": 0.5,
        "dropout_fc": 0.5,

        "lambda_contrastive_inter": 0.1,
        "lambda_contrastive_intra": 3.0,
        "ratio": 16,
    },

    "TVSum":{
        "lr":1e-3,
        "weight_decay": 1e-5,
        "max_epoch": 300,
        "batch_size": 4,
        "seed": 666,

        "num_input_video": 1024,
        "num_input_text": 768,
        "num_hidden": 128,
        "num_layers": 2,

        "dropout_video": 0.1,
        "dropout_text": 0.1,
        "dropout_attn": 0.5,
        "dropout_fc": 0.5,

        "lambda_contrastive_inter": 0.1,
        "lambda_contrastive_intra": 1.0,
        "ratio": 16,
    },

    "BLiSS":{
        "lr":1e-3,
        "weight_decay": 1e-7,
        "max_epoch": 50,
        "batch_size": 64,
        "seed": 12345,

        "num_input_video": 512,
        "num_input_text": 768,
        "num_hidden": 128,
        "num_layers": 6,

        "dropout_video": 0.1,
        "dropout_text": 0.1,
        "dropout_attn": 0.1,
        "dropout_fc": 0.5,

        "lambda_contrastive_inter": 0.01,
        "lambda_contrastive_intra": 0.001,
        "ratio": 4,
    },

    "Daily_Mail":{
        "lr":2e-4,
        "weight_decay": 1e-7,
        "max_epoch": 100,
        "batch_size": 4,
        "seed": 12345,

        "num_input_video": 2048,
        "num_input_text": 768,
        "num_hidden": 256,
        "num_layers": 2,

        "dropout_video": 0.1,
        "dropout_text": 0.1,
        "dropout_attn": 0.1,
        "dropout_fc": 0.5,

        "lambda_contrastive_inter": 0.001,
        "lambda_contrastive_intra": 0.001,
        "ratio": 8,
    },

    "CNN":{
        "lr":2e-4,
        "weight_decay": 1e-5,
        "max_epoch": 100,
        "batch_size": 4,
        "seed": 12345,

        "num_input_video": 2048,
        "num_input_text": 768,
        "num_hidden": 256,
        "num_layers": 2,

        "dropout_video": 0.1,
        "dropout_text": 0.1,
        "dropout_attn": 0.1,
        "dropout_fc": 0.5,

        "lambda_contrastive_inter": 0.0,
        "lambda_contrastive_intra": 0.0,
        "ratio": 0,
    },
} 

def build_args():
    parser = argparse.ArgumentParser("This script is used for the multimodal summarization task.")

    parser.add_argument('--dataset', type=str, default=None, choices=['TVSum', 'SumMe', 'BLiSS', 'Daily_Mail', 'CNN'])
    parser.add_argument('--data_root', type=str, default='data')

    # training & evaluation
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    parser.add_argument('--model_dir', type=str, default='logs')
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--nms_thresh', type=float, default=0.4)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--suffix', type=str, default='')

    # inference
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    parser.add_argument('--test', default=False, action='store_true', help='test mode')

    # common model config
    parser.add_argument('--num_input_video', type=int, default=1024)
    parser.add_argument('--num_input_text', type=int, default=768)
    parser.add_argument('--num_feature', type=int, default=512)
    parser.add_argument('--num_hidden', type=int, default=128)
    
    # transformer config
    parser.add_argument('--dropout_video', type=float, default=0.1, help='pre_drop for video')
    parser.add_argument('--dropout_text', type=float, default=0.1, help='pre_drop for text')
    parser.add_argument('--dropout_attn', type=float, default=0.1, help='dropout for attention operation in transformer')
    parser.add_argument('--dropout_fc', type=float, default=0.5, help='dropout for final classification')
    parser.add_argument('--num_layers', type=int, default=1)

    # contrastive loss
    parser.add_argument('--lambda_contrastive_inter', type=float, default=0.0)
    parser.add_argument('--lambda_contrastive_intra', type=float, default=0.0)
    parser.add_argument('--ratio', type=int, default=16)


    args = parser.parse_args()
    args.lr = _DATASET_HYPER_PARAMS[args.dataset]["lr"]
    args.weight_decay = _DATASET_HYPER_PARAMS[args.dataset]["weight_decay"]
    args.max_epoch = _DATASET_HYPER_PARAMS[args.dataset]["max_epoch"]
    args.batch_size = _DATASET_HYPER_PARAMS[args.dataset]["batch_size"]
    args.seed = _DATASET_HYPER_PARAMS[args.dataset]["seed"]
    
    args.num_input_video = _DATASET_HYPER_PARAMS[args.dataset]["num_input_video"]
    args.num_input_text = _DATASET_HYPER_PARAMS[args.dataset]["num_input_text"]
    args.num_hidden = _DATASET_HYPER_PARAMS[args.dataset]["num_hidden"]
    args.num_layers = _DATASET_HYPER_PARAMS[args.dataset]["num_layers"]

    args.dropout_video = _DATASET_HYPER_PARAMS[args.dataset]["dropout_video"]
    args.dropout_text = _DATASET_HYPER_PARAMS[args.dataset]["dropout_text"]
    args.dropout_attn = _DATASET_HYPER_PARAMS[args.dataset]["dropout_attn"]
    args.dropout_fc = _DATASET_HYPER_PARAMS[args.dataset]["dropout_fc"]
    
    args.lambda_contrastive_inter = _DATASET_HYPER_PARAMS[args.dataset]["lambda_contrastive_inter"]
    args.lambda_contrastive_intra = _DATASET_HYPER_PARAMS[args.dataset]["lambda_contrastive_intra"]
    args.ratio = _DATASET_HYPER_PARAMS[args.dataset]["ratio"]

    return args

def get_arguments() -> argparse.Namespace:
    args = build_args()

    args.model_dir = f'{args.model_dir}/{args.dataset}'
    if len(args.suffix) > 0:
        args.model_dir += f'_{args.suffix}'
    return args

