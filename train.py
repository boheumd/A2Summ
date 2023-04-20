import logging
import os

from torch.utils.tensorboard import SummaryWriter
from train_msmo import train_msmo
from train_videosumm import train_videosumm
from config import *
from utils import *

logger = logging.getLogger()

# For SumMe/TVSum datasets
def main_videosumm(args):
    init_logger(args.model_dir, args.log_file)
    set_random_seed(args.seed)
    dump_yaml(vars(args), '{}/args.yml'.format(args.model_dir))

    logger.info(vars(args))
    os.makedirs(args.model_dir, exist_ok=True)
    print(args.model_dir)

    args.writer = SummaryWriter(os.path.join(args.model_dir, 'tensorboard'))
    
    split_path = '{}/{}/splits.yml'.format(args.data_root, args.dataset)
    split_yaml = load_yaml(split_path)

    f1_results = {}
    stats = AverageMeter('fscore')

    for split_idx, split in enumerate(split_yaml):
        logger.info(f'Start training on {split_path}: split {split_idx}')
        max_val_fscore, best_val_epoch, max_train_fscore = train_videosumm(args, split, split_idx)
        stats.update(fscore=max_val_fscore)

        f1_results[f'split{split_idx}'] = float(max_val_fscore)

    logger.info(f'Training done on {split_path}.')
    logger.info(f'F1_results: {f1_results}')
    logger.info(f'F1-score: {stats.fscore:.4f}\n\n')

# For CNN/DailyMail/BLiSS datasets
def main_msmo(args):
    init_logger(args.model_dir, args.log_file)
    set_random_seed(args.seed)
    dump_yaml(vars(args), '{}/args.yml'.format(args.model_dir))

    logger.info(vars(args))
    os.makedirs(args.model_dir, exist_ok=True)
    print(args.model_dir)

    args.writer = SummaryWriter(os.path.join(args.model_dir, 'tensorboard'))

    max_val_R1, max_val_R2, max_val_RL, max_val_cos, best_val_epoch, \
        max_train_R1, max_train_R2, max_train_RL, max_train_cos = train_msmo(args)

    logger.info(f'Training done. Val R1: {max_val_R1:.4f}, R2: {max_val_R2:.4f}, RL: {max_val_RL:.4f}, Cos: {max_val_cos:.4f}, Best Epoch:{best_val_epoch}.')
    logger.info(f'             Train R1: {max_train_R1:.4f}, R2: {max_train_R2:.4f}, RL: {max_train_RL:.4f}, Cos: {max_train_cos:.4f}.\n\n')


if __name__ == '__main__':
    args = get_arguments()
    if args.dataset in ['Daily_Mail', 'CNN', 'BLiSS']:
        main_msmo(args)
    elif args.dataset in ['TVSum', 'SumMe']:
        main_videosumm(args)
    else:
        raise NotImplementedError

