import os
from EX_CONST import Const
from PIL import Image
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from detectors.datasets import *
from detectors.loss.gaussian_mse import GaussianMSE
from detectors.models.persp_trans_detector import PerspTransDetector
from detectors.utils.logger import Logger
from detectors.utils.draw_curve import draw_curve
from detectors.utils.image_utils import img_color_denormalize
from detectors.OFTTrainer import OFTtrainer
from detectors.RPNTrainer import RPNtrainer
import warnings
import itertools
from detectors.models.VGG16Head import VGG16RoIHead
from tensorboardX import SummaryWriter
import torch.nn as nn
warnings.filterwarnings("ignore")

def main(args):
    # seed
    writer = SummaryWriter('/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/tensorboard/log')

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    bright = T.ColorJitter(brightness = 0.5)
    contrast = T.ColorJitter(contrast=0.5)
    saturation = T.ColorJitter(saturation=0.5)

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # resize = T.Resize([384, 512]) # h, w
    # resize = T.Resize([240, 320]) # h, w
    train_trans = T.Compose([T.ToTensor(), bright, normalize])
    test_trans = T.Compose([T.ToTensor(), normalize])
    data_path = os.path.expanduser('/home/dzc/Data/%s' % Const.dataset)
    # data_path2 = os.path.expanduser('/home/dzc/Data/%s' % Const.dataset)
    base = Robomaster_1_dataset(data_path, args, worldgrid_shape=Const.grid_size)
    train_set = oftFrameDataset(base, train=True, transform=train_trans, grid_reduce=Const.reduce)
    test_set = oftFrameDataset(base , train=False, transform=test_trans, grid_reduce=Const.reduce)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # model
    model = PerspTransDetector(train_set)
    # classifier = model.classifier
    roi_head = VGG16RoIHead(Const.roi_classes + 1,  7, 1/Const.reduce)
    optimizer = optim.Adam(params=itertools.chain(model.parameters(), roi_head.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr': 1e-3},
    #                         {'params': filter(lambda p: p.requires_grad, model.rpn.parameters())},
    #                         {'params': filter(lambda p: p.requires_grad, roi_head.parameters())}], lr=args.lr, weight_decay=args.weight_decay)
    print('Settings:')
    print(vars(args))

    trainer = OFTtrainer(model, roi_head, denormalize)
    # trainer = RPNtrainer(model, roi_head, denormalize)

    # learn0.
    # model.load_state_dict(torch.load('%s/mvdet_rpn_%d.pth' % (Const.modelsavedir, 30)))
    # roi_head.load_state_dict(torch.load('%s/roi_rpn_head_%d.pth' % (Const.modelsavedir, 30)))
    print()
    # model.load_state_dict(torch.load("%s/mvdet_rpn_%d.pth" % (Const.modelsavedir, 4)))
    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        if not args.resume:
            print('Training...')
            # model.load_state_dict(torch.load("%s/mvdet_sep_rpn_%d.pth" % (Const.modelsavedir, 10)))
            # roi_head.load_state_dict(torch.load("%s/roi_rpn_head_%d.pth" % (Const.modelsavedir, 10)))
            loss = trainer.train(epoch, train_loader, optimizer, writer)
            torch.save(model.state_dict(), os.path.join('%s/mvdet_rpn_%d.pth' % (Const.modelsavedir, epoch)))
            torch.save(roi_head.state_dict(), os.path.join('%s/roi_rpn_head_%d.pth' % (Const.modelsavedir, epoch)))
            # trainer.test(epoch, test_loader, writer)
        else:
            print('Testing...')
            model.load_state_dict(torch.load("%s/mvdet_rpn_%d.pth" % (Const.modelsavedir, 30)))
            roi_head.load_state_dict(torch.load("%s/roi_rpn_head_%d.pth" % (Const.modelsavedir, 30)))
            trainer.test(epoch, test_loader, writer)
            break
    writer.close()

if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('-d', '--dataset', type=str, default='robo', choices=['wildtrack', 'multiviewx','robo'])
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=35, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.00015, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=71, help='random seed (default: None)')

    parser.add_argument('--resume', type=bool, default = True)
    args = parser.parse_args()

    main(args)