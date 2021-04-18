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
import warnings
from tensorboardX import SummaryWriter
warnings.filterwarnings("ignore")

def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = T.Resize([384, 512]) # h, w
    # resize = T.Resize([240, 320]) # h, w
    train_trans = T.Compose([resize, T.ToTensor(), normalize])

    data_path = os.path.expanduser('/home/dzc/Data/%s' % Const.dataset)
    base = Robomaster_1_dataset(data_path, args, worldgrid_shape=Const.grid_size)
    train_set = oftFrameDataset(base, train=True, transform=train_trans, grid_reduce=4)
    test_set = oftFrameDataset(base , train=False, transform=train_trans, grid_reduce=4)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # model
    model = PerspTransDetector(train_set, args.arch)


    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # loss

    logdir = f'logs/{args.dataset}_frame/{args.variant}/' + Const.dataset
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []
    test_moda_s = []

    trainer = OFTtrainer(model, logdir, denormalize, args.cls_thres, args.alpha)

    # learn
    if args.resume is None or args.resume == 1:
        print()
        print('Testing...')

        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')

            loss = trainer.train(epoch, train_loader, optimizer, args.log_interval)
            print('Testing...')
            # trainer.test(test_loader)
            # torch.save(model.state_dict(), os.path.join('/home/dzc/Desktop/CASIA/proj/mvdet/MVDet/finalModels/mvdet_model.pth'))
if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'small', 'large'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx','robo'])
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--numcam', type=int, default=2)
    parser.add_argument('--aux', type=bool, default=False)
    parser.add_argument('--info', type=str, default="no_extra")
    args = parser.parse_args()

    main(args)