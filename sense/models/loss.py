"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flow_EPE(nn.Module):
  def __init__(self, isNorm=1):
    super(Flow_EPE, self).__init__()
    self.isNorm = isNorm

  def forward(self, input, target, mask=None):
    n_batch, c, h, w = input.size()
    epe = 0
    for i in range(n_batch):        
        cur_diff = 1.0 * (input[i] - target[i])
        cur_diff[target[i]!=target[i]] = 0
        cur_diff[target[i]>1000] = 0
        cur_diff[target[i]<-1000] = 0
        if mask is not None:
            cur_diff[mask[i]<1] = 0
        cur_val = 1.0 * torch.sum(torch.sqrt(torch.sum(cur_diff * cur_diff, 0)))
        if self.isNorm>0:
            n_bad = (target[i]!=target[i]).float().sum() + \
                (target[i]>1000).float().sum() + \
                (target[i]<-1000).float().sum()
            if mask is not None:
                n_bad += (1 - mask[i]).float().sum()
            n_bad = n_bad/2
            cur_val = 1.0 * cur_val/(h*w-n_bad)
        epe += cur_val
    
    if self.isNorm > 0:
        epe = 1.0 * epe/n_batch
    return epe


class Loss_Robust(nn.Module):
  def __init__(self, size_average=False, p_robust=0.4):
    super(Loss_Robust, self).__init__()
    self.size_average = size_average
    self.p_robust = p_robust
    print('Loss_Robust, p_robust: {}'.format(self.p_robust))

  def forward(self, input, target, mask=None):
    if self.size_average:
        raise NotImplementedError('size_average is not supported for Loss_Robust yet.')

    n_batch, c, h, w = input.size()
    loss_val = 0
    epsilon_robust = 0.01
    p_robust = self.p_robust
    for i in range(n_batch):        
        cur_diff = 1.0 * (input[i] - target[i])
        cur_diff[target[i]!=target[i]] = 0
        cur_diff[target[i]>1000] = 0
        cur_diff[target[i]<-1000] = 0
        if mask is not None:
            cur_diff[mask[i]<1] = 0
        cur_val = 1.0 * torch.sum(torch.pow(torch.sum(torch.abs(cur_diff), 0) + epsilon_robust, p_robust))
        loss_val += cur_val

    return loss_val

class MultiScaleLoss(nn.Module):

    def __init__(self, downsample_factors, weights=None, loss= 'l1', size_average=False, 
        class_weights=None, p_robust=0.4):
        super(MultiScaleLoss,self).__init__()
        self.downsample_factors = downsample_factors
        self.weights = torch.Tensor(downsample_factors).fill_(1) if weights is None else torch.Tensor(weights)
        self.weights = self.weights.cuda()
        assert(len(weights) == len(downsample_factors))
        self.loss = loss
        if loss=='l1':
            self.lossFunc = torch.nn.L1Loss(size_average=size_average)
        elif loss == 'smooth_l1':
            self.lossFunc = torch.nn.SmoothL1Loss(size_average=size_average)
        elif loss=='epe_loss':
            self.lossFunc = Flow_EPE(isNorm=size_average)
        elif loss=='epe':
            # self.lossFunc = torch.nn.MSELoss(size_average=True)
            self.lossFunc = Flow_EPE(isNorm=1)
        elif loss=='l1_robust':
            self.lossFunc = Loss_Robust(size_average=size_average, p_robust=p_robust)
        elif loss == 'xentropy_loss':
            self.lossFunc = torch.nn.CrossEntropyLoss(size_average=size_average, ignore_index=-1, weight=class_weights)
        else:
            print("Loss not implemented")
            pass
        self.multiScales = [nn.AvgPool2d(df, df) for df in downsample_factors]
        self.size_average = size_average

    def forward(self, input, target, mask=None):
        out = 0
        losses = []
        if type(input) is tuple:  
            for i,input_ in enumerate(input):
                downscale = int(self.downsample_factors[i])
                # target_ = self.multiScales[i](target)
                if len(target.size()) == 4:
                    target_ = target[:, :, ::downscale, ::downscale]
                else:
                    target_ = target[:, ::downscale, ::downscale]
                if mask is not None:
                    if len(mask.size()) == 4:
                        mask_ = mask[:, :, ::downscale, ::downscale]
                    else:
                        mask_ = mask[:, ::downscale, ::downscale]

                    if self.loss == 'epe_loss' or self.loss == 'epe' or self.loss == 'l1_robust':
                        tmp = self.lossFunc(input_, target_, mask_)
                    else:
                        tmp = self.lossFunc(input_[mask_], target_[mask_])
                else:
                    tmp = self.lossFunc(input_, target_)
                out += self.weights[i]*tmp
                losses.append(self.weights[i].data.cpu().item() * tmp.data.cpu().item())
                # tmp2 = self.weights[i] * tmp
                # print('\t***', i, target_.size(), self.weights[i].data.cpu(), tmp.data.cpu(), tmp2.data.cpu())
                # print self.weights[i], input_.size(), target_.size()
                # EPE_ = EPE(input_,target_, self.loss)
                # pdb.set_trace()
                # out += self.weights[i]*nn.L1Loss()(EPE_,EPE_.detach()*0) #Compare EPE_ with A Variable of the same size, filled with zeros)
                # out += self.weights[i]*torch.mean(EPE_)*1.0/2
        else:
            # pdb.set_trace()
            target_ = self.multiScales[0](target)
            if mask is None:
                if self.loss == 'epe_loss' or self.loss == 'epe' or self.loss == 'l1_robust':
                    out += self.weights[0] * self.lossFunc(input, target_, mask)
                else:
                    out += self.weights[0] * self.lossFunc(input[mask], target_[mask])
            else:
                out += self.weights[0]*self.lossFunc(input, target_)
            # EPE_ = EPE(input, self.multiScales[0](target), 'L2')
            # out += torch.mean(EPE_)
        # print out
        return out, losses

def multiscaleloss(downsample_factors, weights=None, loss='L1', size_average=False, 
    class_weights=None, p_robust=0.4):
    if weights is None:
        print("must specify the weights")
        pass
    if len(downsample_factors) ==1 and type(weights) is not tuple: #a single value needs a particular syntax to be considered as a tuple
        weights = (weights,)
    return MultiScaleLoss(downsample_factors,weights,loss, size_average, class_weights, p_robust)