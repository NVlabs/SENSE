"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# adapted from UPerNet https://github.com/CSAILVision/semantic-segmentation-pytorch
# System libs
import os
import datetime
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from PIL import Image
# Our libs
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion
from lib.utils import as_numpy, mark_volatile
import lib.utils.data as torchdata
import cv2
import pickle
import logging
# from UPerNet
import time
from models.models import UPerNet
# from SENSE
from sense.datasets.cityscapes_dataset import SegList, CITYSCAPE_PALETTE, TRIPLET_PALETTE
import sense.datasets.segmentation_data_transforms as transforms

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_input_images(filenames, output_dir):
    for ind in range(len(filenames)):
        im = Image.open(filenames[ind])
        im_dir, fn = os.path.split(filenames[ind])
        _, seq_name = os.path.split(im_dir)
        fn = os.path.join(output_dir, seq_name, fn[:-4] + '.png')
        out_dir = os.path.split(fn)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)

def save_raw_result(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im_dir, fn = os.path.split(filenames[ind])
        _, seq_name = os.path.split(im_dir)
        fn = os.path.join(output_dir, seq_name, fn[:-4] + '.pkl')
        # im = Image.fromarray(predictions[ind].astype(np.uint8))
        pred = predictions[ind].cpu().numpy()
        print('save_output_images', fn)
        out_dir = os.path.split(fn)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # im.save(fn)
        # np.savez_compressed(fn, pred=pred)
        with open(fn, 'wb') as f:
            pickle.dump(pred, f, pickle.HIGHEST_PROTOCOL)

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        im_dir, fn = os.path.split(filenames[ind])
        _, seq_name = os.path.split(im_dir)
        fn = os.path.join(output_dir, seq_name, fn[:-4] + '.png')
        print('save_output_images', fn)
        out_dir = os.path.split(fn)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       im_dir, fn = os.path.split(filenames[ind])
       _, seq_name = os.path.split(im_dir)
       fn = os.path.join(output_dir, seq_name, fn[:-4] + '.png')
       print('save_colorful_images', fn)
       out_dir = os.path.split(fn)[0]
       if not os.path.exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)

def pixel_acc(pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    correct = preds == label.long()
    acc_sum = torch.sum(valid * correct.long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc


def test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    pix_acc = 0
    for iter, (image, label, name) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        feed_dict = {'img_data': image.cuda(),
                     'seg_label': label.cuda()
                    }
        # print(image.min().item(), image.max().item())
        segSize = (label.shape[1], label.shape[2])
        with torch.no_grad():
            pred = model(feed_dict, segSize=segSize)
            final, raw_final = pred
            pix_acc += pixel_acc(final.cpu(), label)
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        if save_vis:
            # save_input_images(name, output_dir + '_input')
            save_raw_result(raw_final, name, output_dir + '_logits')
            # save_output_images(pred, name, output_dir)
            save_colorful_images(
                pred, name, output_dir + '_color',
                TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}, pixelAccuracy: {pix_acc:.2f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2),
                pix_acc=pix_acc/(iter+1)*100))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    # workers = [threading.Thread(target=resize_one, args=(i, j))
    #            for i in range(tensor.size(0)) for j in range(tensor.size(1))]

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # for i in range(tensor.size(0)):
    #     for j in range(tensor.size(1)):
    #         out[i, j] = np.array(
    #             Image.fromarray(tensor_cpu[i, j]).resize(
    #                 (w, h), Image.BILINEAR))
    # out = tensor.new().resize_(*out.shape).copy_(torch.from_numpy(out))
    return out


def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    num_scales = len(scales)
    for iter, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]
        h, w = input_data[0].size()[2:4]
        images = [input_data[0]]
        images.extend(input_data[-num_scales:])
        # pdb.set_trace()

        final = None
        with torch.no_grad():
            for image in images:
                feed_dict = {'img_data': image.cuda(),
                             'seg_label': label.cuda()
                            }
                segSize = (label.shape[1], label.shape[2])
                if final is None:
                    final = model(feed_dict, segSize=segSize)
                else:
                    final += model(feed_dict, segSize=segSize)
        pred = final.argmax(axis=1)
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color',
                                 CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def main(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    torch.cuda.set_device(args.gpu_id)

    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=255)

    model = SegmentationModule(net_encoder, net_decoder, crit)
    model.cuda()

    data_dir = args.data_dir
    if args.pix_val_01:
        normalize = transforms.Normalize(mean=[0, 0, 0], 
                                     std=[1., 1., 1.]
                                     )
    else:    
        normalize = transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], 
                                         std=[1., 1., 1.]
                                         )
    scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    if args.ms:
        dataset = SegListMS(data_dir, phase, transforms.Compose([
            transforms.ToTensor(convert_pix_range=args.pix_val_01),
            normalize,
        ]), scales, list_dir=args.data_dir)
    else:
        dataset = SegList(data_dir, phase, transforms.Compose([
            transforms.ToTensor(convert_pix_range=args.pix_val_01),
            normalize,
        ]), list_dir=args.data_dir, out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    # if len(args.test_suffix) > 0:
    #     out_dir += '_' + args.test_suffix
    out_dir = args.result
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.num_class, save_vis=False,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        mAP = test(test_loader, model, args.num_class, save_vis=args.save_vis,
                   has_gt=(phase == 'train' or phase == 'val'), 
                   output_dir=out_dir)

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--weights_encoder', default='',
                        help="weights to finetune net_encoder")
    parser.add_argument('--weights_decoder', default='',
                        help="weights to finetune net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')
    parser.add_argument('--suffix', default='_epoch_500.pt',
                        help='suffix of the saved model.')

    # Distillation model related arguments
    parser.add_argument('--distill-enc-arch', default=None,
                        help='encoder architecture of the distilled model')
    parser.add_argument('--distill-dec-arch', default=None,
                        help='decoder architecture of the distilled model')
    parser.add_argument('--distill-enc-weights', default='')
    parser.add_argument('--distill-dec-weights', default='')

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/train.odgt')
    parser.add_argument('--list_val',
                        default='./data/validation.odgt')
    parser.add_argument('--root_dataset',
                        default='./data/')
    parser.add_argument('--data-dir', type=str,
                        help='data directory of CityScapes',
                        default='/home/hzjiang/Data/CityScapes')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=8, type=int,
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                        help='input batch size')
    parser.add_argument('--gpu-id', default=0, type=int,
                        help='which GPU to use')
    parser.add_argument('--batch-size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_encoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_decoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--deep_sup_scale', default=0.4, type=float,
                        help='the weight of deep supervision loss')
    parser.add_argument('--fix_bn', default=0, type=int,
                        help='fix bn params')

    # Data related arguments
    parser.add_argument('--num_class', default=19, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=16, type=int,
                        help='number of data loading workers')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')
    parser.add_argument('--crop-size', type=int, default=840,
                        help='random crop size')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')
    parser.add_argument('--ms', action='store_true',
                        help='whether to do multi-scale testing.')
    parser.add_argument('--result', default=None,
                        help='result folder to save visualizations.')
    parser.add_argument('--phase', default='val')
    parser.add_argument('--pix-val-01', action='store_true',
        help='whether to normalize pixel value in [0, 1]')
    args = parser.parse_args()

    args.save_vis = args.result is not None

    # absolute paths of model weights
    if args.distill_enc_arch is None:
        args.weights_encoder = os.path.join(args.ckpt, args.id,
                                            'encoder' + args.suffix)
        args.weights_decoder = os.path.join(args.ckpt, args.id,
                                            'decoder' + args.suffix)
        assert os.path.exists(args.weights_encoder) and \
            os.path.exists(args.weights_encoder), 'checkpoint does not exitst! {}'.format(args.weights_encoder)
    else:
        args.distill_enc_weights = os.path.join(args.ckpt, args.id, 'encoder' + args.suffix)
        args.distill_dec_weights = os.path.join(args.ckpt, args.id, 'decoder' + args.suffix)
        print(args.distill_enc_weights)
        print(args.distill_dec_weights)
        assert os.path.exists(args.distill_enc_weights) and \
            os.path.exists(args.distill_dec_weights), 'checkpoint does not exitst!'

    if args.result is not None:
        args.result = os.path.join(args.result, args.id)
        if not os.path.isdir(args.result):
            os.makedirs(args.result)

    main(args)
