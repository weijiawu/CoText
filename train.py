import torch
import numpy as np
import random
import shutil
import argparse
import os
import os.path as osp
import sys
import time
import json
from mmcv import Config
from utils.corrector import eval_img
from tqdm import tqdm
from dataset import build_data_loader
from models import build_model
from utils import AverageMeter
import copy


torch.manual_seed(23)
torch.cuda.manual_seed(23)
np.random.seed(23)
random.seed(22)


def train(train_loader, model, optimizer, epoch, start_iter, cfg, checkpoint_path, config_path):
    model.train()

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernels = AverageMeter()
    losses_emb = AverageMeter()
    losses_rec = AverageMeter()
    losses_desc =  AverageMeter()
    ious_text = AverageMeter()
    ious_kernel = AverageMeter()
    accs_rec = AverageMeter()


    with_rec = hasattr(cfg.model, 'recognition_head')
    max_iters = len(train_loader)
    cfg.update(dict(
            report_speed=False
        ))
    # start time
    start = time.time()
    for iter, data in enumerate(train_loader):
        # skip previous iterations
        if iter < start_iter:
            print('Skipping iter: %d' % iter)
            sys.stdout.flush()
            continue

        # time cost of data loader
        data_time.update(time.time() - start)

        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter, cfg)

        # prepare input
        data.update(dict(
            cfg=cfg
        ))

        # forward
        outputs = model(**data)
        
        # detection loss
        loss_text = torch.mean(outputs['loss_text'])
        losses_text.update(loss_text.item())

        loss_kernels = torch.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.item())
        if 'loss_emb' in outputs.keys():
            loss_emb = torch.mean(outputs['loss_emb'])
            losses_emb.update(loss_emb.item())

        iou_text = torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item())
        iou_kernel = torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item())

        # recognition loss
        if with_rec:
            loss_rec = outputs['loss_rec']
            valid = loss_rec > 0.5
            if torch.sum(valid) > 0:
                loss_rec = torch.mean(loss_rec[valid])
                losses_rec.update(loss_rec.item())

                acc_rec = outputs['acc_rec']
                acc_rec = torch.mean(acc_rec[valid])
                accs_rec.update(acc_rec.item(), torch.sum(valid).item())
            
            loss_desc = outputs['loss_desc']
            loss_desc = torch.mean(loss_desc)
            losses_desc.update(loss_desc.item())
        loss = torch.mean(outputs['loss'])
        losses.update(loss.item())
        # print(losses_desc.avg)
        # backward
        optimizer.zero_grad()
        if not torch.isnan(loss) and loss.item()>0:
            loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        # print log
        if iter % 10 == 0:
            output_log = f'({iter + 1}/{len(train_loader)}) LR: {optimizer.param_groups[0]["lr"]:.6f} | ' \
                         f'Batch: {batch_time.avg:.3f}s | Total: {batch_time.avg * iter / 60.0:.0f}min | ' \
                         f'ETA: {batch_time.avg * (len(train_loader) - iter) / 60.0:.0f}min | ' \
                         f'Loss: {losses.avg:.3f} | ' \
                         f'Loss(text/kernel/emb{"/rec/desc" if with_rec else ""}): {losses_text.avg:.3f}/{losses_kernels.avg:.3f}/' \
                         f'{losses_emb.avg:.3f}{"/" + format(losses_rec.avg, ".3f") if with_rec else ""}{"/" + format(losses_desc.avg, ".3f") if with_rec else ""}| ' \
                         f'IoU(text/kernel): {ious_text.avg:.3f}/{ious_kernel.avg:.3f}' \
                         f'{" | ACC rec: " + format(accs_rec.avg, ".3f") if with_rec else ""}'

            print(output_log)
            sys.stdout.flush()
        
        if iter % (max_iters//2) == 2 and epoch >= 1:
            model.eval()
            state = dict(
                epoch=epoch + 1,
                iter=iter,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict()
            )
            file_path = save_checkpoint(state, checkpoint_path, 0, 0, 0)
#             if cfg.train_cfg.isvalidate:
#                 p, r, f = validate(file_path, config_path)
#                 save_checkpoint(state, checkpoint_path, p, r, f)
#                 os.remove(file_path)
            model.train()


def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train_cfg.schedule

    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train_cfg.epoch * len(dataloader)
        lr = cfg.train_cfg.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, tuple):
        lr = cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, checkpoint_path, p, r, f):
    metric = "{}_{}_{}".format(round(p, 5), round(r, 5), round(f, 5))
    file_path = osp.join(checkpoint_path, '{}_{}_{}_checkpoint.pth.tar'.format(state['epoch'], state['iter'], metric))
    torch.save(state, file_path)
    return file_path

def main(args):
    cfg = Config.fromfile(args.config)
    print(json.dumps(cfg._cfg_dict, indent=4))

    if args.checkpoint is not None:
        checkpoint_path = osp.join('outputs', args.checkpoint)

    else:
        cfg_name, _ = osp.splitext(osp.basename(args.config))
        checkpoint_path = osp.join('outputs', cfg_name)
    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    print('Checkpoint path: %s.' % checkpoint_path)
    sys.stdout.flush()
    
    if hasattr(cfg.model, 'recognition_head') and 'CTC' in cfg.model.recognition_head.type:
        cfg.data.train.update(dict(
            use_ctc=True
        ))

    # data loader
    train_dataset = build_data_loader(cfg.data.train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    # model
    if hasattr(cfg.model, 'recognition_head'):
        cfg.model.recognition_head.update(dict(
            voc=train_dataset.voc,
            char2id=train_dataset.char2id,
            id2char=train_dataset.id2char,
        ))
    if hasattr(cfg.model, 'description_head'):
        cfg.model.description_head.update(dict(
            voc=train_dataset.voc
        ))
    
    model = build_model(cfg.model)
    # 冻结操作
    for name, param in model.named_parameters():
        if cfg.train_cfg.freeze_backbone and  name.split('.')[0] == 'backbone':
            param.requires_grad = False
            print('Freeze {}'.format(name))
        elif cfg.train_cfg.freeze_neck and  name.split('.')[0] in ['reduce_layer4', 
                                                            'reduce_layer3', 
                                                            'reduce_layer2',
                                                            'reduce_layer1',
                                                            'fpem1',
                                                            'fpem2']:
            param.requires_grad = False
            print('Freeze {}'.format(name))
        elif cfg.train_cfg.freeze_det and  name.split('.')[0] == 'det_head':
            param.requires_grad = False
            print('Freeze {}'.format(name))
        elif cfg.train_cfg.freeze_desc and  name.split('.')[0] == 'desc_head':
            param.requires_grad = False
            print('Freeze {}'.format(name))
        if cfg.train_cfg.freeze_rec and name.split('.')[0] == 'rec_head':
            param.requires_grad = False
            print('Freeze {}'.format(name))
            
    model = torch.nn.DataParallel(model).cuda()
    
    # Check if model has custom optimizer / loss
    if False: # hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        if cfg.train_cfg.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train_cfg.lr, momentum=0.99, weight_decay=5e-4)
        elif cfg.train_cfg.optimizer == 'Adam':
            # TODO 冻结识别部分参数
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_cfg.lr)

    start_epoch = 0
    start_iter = 0
    if args.resume:
        assert osp.isfile(args.resume), 'Error: no checkpoint directory found!'
        print('Resuming from checkpoint %s.' % args.resume)
        checkpoint = torch.load(args.resume)

        # start_epoch =  checkpoint['epoch']
        # start_iter = checkpoint['iter']
        if not cfg.train_cfg.load_desc_weights:
            print("removing desc weights!")
            checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'desc_head' not in k}
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])


    for epoch in range(start_epoch, cfg.train_cfg.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, cfg.train_cfg.epoch))

        train(train_loader, model, optimizer, epoch, start_iter, cfg, checkpoint_path, args.config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    args = parser.parse_args()

    main(args)
