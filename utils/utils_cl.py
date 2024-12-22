import os

import cv2
import kornia
import numpy
import numpy as np
from kornia.filters import SpatialGradient
from torch import Tensor
import torch.nn as nn
import torch
from nets.feature_s import GaussianBlurConv
from tqdm import tqdm
import random
from utils.batch_transformers import RGBToGray, YCbCrToRGB, RGBToYCbCr


def gradient(spatial, x: Tensor, eps: float = 1e-6) -> Tensor:
    s = spatial(x)
    dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
    u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + eps)
    return u


def fit_one_epoch(Generator_train, Generator, ir_Generator_train, ir_Generator, vi_Generator_train, vi_Generator,
                  vgg_loss, ema_1, ema_2, ema_3, opt_generator, opt_ir_generator, opt_vi_generator, epoch, epoch_step, gen, Epoch,
                  cuda, save_period, save_dir, local_rank=0):
    loss = 0
    loss_0, loss_1, loss_x = np.empty(1), np.empty(1), np.empty(1)
    loss_0[0], loss_1[0], loss_x[0] = 1, 1, 1
    
    vloss_ir, vloss_vi, vloss_en = np.empty(1), np.empty(1), np.empty(1)
    vloss_ir[0], vloss_vi[0], vloss_en[0] = 1, 1, 1
    
    v1loss_vi, v1loss_en = np.empty(1), np.empty(1)
    v1loss_vi[0], v1loss_en[0] = 1, 1
    
    spatial = SpatialGradient('diff')
    lF = nn.MSELoss(reduction='none')
    l1 = nn.L1Loss(reduction='none')
    l1.cuda()
    lF.cuda()
    spatial.cuda()
    gaussianBlurConv = GaussianBlurConv(channels=1)
    gaussianBlurConv = gaussianBlurConv.cuda()
    rgb2gray = RGBToGray()
    rgb2ycbcr = RGBToYCbCr()
    ycbcr2rgb = YCbCrToRGB()
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        ir_images, vi_images, enhanced_image, vi_mark, ir_mark = batch[0], batch[1], batch[2], batch[3], batch[4]
        with torch.no_grad():
            if cuda:
                ir_images = ir_images.cuda()
                vi_images = vi_images.cuda()
                enhanced_image = enhanced_image.cuda()
                vi_mark = vi_mark.cuda()
                ir_mark = ir_mark.cuda()
                max_mark = torch.max(vi_mark,ir_mark).cuda()
                '''
                ir_01 = ir_mark>0
                
                ir_01 = ir_01.float()
                #print(ir_01[0])
                #print(ir_01.sum())
                
                ir_01 = torch.max(torch.max(ir_01[:,0,:,:],ir_01[:,1,:,:]),ir_01[:,2,:,:]).unsqueeze(1)
                #print(ir_01[0])
                #print(ir_01.sum())
                
                vi_01 = vi_mark>0
                vi_01 = vi_01.float()
                vi_01 = torch.max(torch.max(vi_01[:,0,:,:],vi_01[:,1,:,:]),vi_01[:,2,:,:]).unsqueeze(1)
                
                min_01 = torch.min(ir_01, vi_01)
                
                a = vi_images.shape[0]
                enhanced_image = torch.zeros(vi_images.shape).cuda()
                for i in range(a):
                    enhanced_image[i:i+1, :, :, :] = en_net(vi_images[i:i+1, :, :, :], return_vals = False)
                
                ir_sep = ir_01 - min_01
                vi_sep = vi_01 - min_01
                same_01 = 1 - ir_sep - vi_sep
                
                ir_back = ir_images * same_01
                vi_back = vi_images * same_01
                en_back = enhanced_image * same_01
                
                ir_tar = ir_ycbcr * ir_sep
                vi_tar = vi_ycbcr * vi_sep
                '''
                ir_ycbcr = rgb2ycbcr(ir_images)
                vi_ycbcr = rgb2ycbcr(vi_images)
                en_ycbcr = rgb2ycbcr(enhanced_image)
                ir_y, ir_cb, ir_cr = ir_ycbcr.chunk(3, 1)
                vi_y, vi_cb, vi_cr = vi_ycbcr.chunk(3, 1)
                en_y, en_cb, en_cr = en_ycbcr.chunk(3, 1)
                alpha, beta = np.random.choice([0,-1,1]), -1
                
        Generator_train.train()
        vi_Generator_train.train()
        ir_Generator_train.train()
        g1_ir = gradient(spatial, ir_y)
        g1_vi = gradient(spatial, vi_y)
        
        for alpha in [0,-1,1]:
            opt_generator.zero_grad()
            opt_ir_generator.zero_grad()
            opt_vi_generator.zero_grad()
            
            if alpha == -1:
                alpha = random.random()
            
            fus_y, fus_cbcr, global_alpha = Generator_train(ir_images, vi_images, ir_mark, vi_mark, alpha, beta, cnt=iteration)
            fus_ycbcr = torch.cat([fus_y, fus_cbcr],dim=1)
            fus_images = ycbcr2rgb(fus_ycbcr)#torch.cat([fus_y, fus_y, fus_y],dim=1)
            fus_y, fus_cb, fus_cr = fus_ycbcr.chunk(3, 1)
            
            #fus_Y = torch.cat([fus_y,fus_y,fus_y],dim=1)
            #vi_Y = torch.cat([vi_y,vi_y,vi_y],dim=1)
            #en_Y = torch.cat([en_y,en_y,en_y],dim=1)
            #fus_images = fus_Y
            
            # 若出现色彩不准问题可尝试加大b1或者减小b4的系数，实验发现增加b4的权重可以有效的提升下游任务性能
            l_pixel, l_grad, l_target, l_ir_src, l_vi_src, l_vgg = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
            
            src_ir_images = ir_Generator_train(fus_images)
            src_vi_images = vi_Generator_train(fus_images)
            l_ir_src = l1(ir_images, src_ir_images)
            l_vi_src = l1(vi_images, src_vi_images)
            #print(l_ir_src.shape)
            
            g1_fus = gradient(spatial, fus_y)
            
            b1, b2, b3, b4, b5 = [3, 8, 0.1, 1, 1]
            if  alpha == 1:
                l_pixel = 2 * (l1(fus_y, vi_y) + l1(fus_cb, vi_cb) + l1(fus_cr, vi_cr))
                #l_pixel = l_pixel + torch.max(l1(fus_cbcr, torch.cat([vi_cb, vi_cr], dim=1)),l1(fus_cbcr, torch.cat([en_cb, en_cr], dim=1)))
                l_grad = l1(g1_fus, g1_vi)
                '''
                l_vgg_vi = vgg_loss(fus_images, vi_images).mean()
                l_vgg_en = vgg_loss(fus_images, enhanced_image).mean()
                np.append(v1loss_vi, l_vgg_vi.item())
                np.append(v1loss_en, l_vgg_en.item())
                tot = (v1loss_vi.mean() + v1loss_en.mean())
                k1, k2 = tot/v1loss_vi.mean(), tot/v1loss_en.mean()
                l_vgg = (k1*l_vgg_vi + k2*l_vgg_en)/2
                '''
                l_re = ((1-global_alpha) * l_ir_src + global_alpha * l_vi_src).mean()
                l_src = b1 * l_pixel.mean() + b2 * l_grad.mean() + b3 * l_vgg + l_re + b5 * l_target.mean() # #  # l_invary +  + b1 * l_ssim.mean()
                np.append(loss_1, l_src.mean().item())
                FF = (loss_x.mean() + loss_1.mean() + loss_0.mean()) / loss_1.mean()
                l_src = FF * l_src
            elif alpha == 0:
                l_pixel = 2 * (l1(fus_y, ir_y) + l1(fus_cb, ir_cb) + l1(fus_cr, ir_cr))
                l_grad = l1(g1_fus, g1_ir)
                l_target = lF(fus_y, ir_y) * gaussianBlurConv(g1_ir)
                #l_vgg = vgg_loss(fus_images, ir_images).mean()
                l_re = ((1-global_alpha) * l_ir_src + global_alpha * l_vi_src).mean()
                l_src = b1 * l_pixel.mean() + b2 * l_grad.mean() + b3 * l_vgg + l_re + b5 * l_target.mean() # #  # l_invary +  + b1 * l_ssim.mean()
                np.append(loss_0, l_src.mean().item())
                FF = (loss_x.mean() + loss_1.mean() + loss_0.mean()) / loss_0.mean()
                l_src = FF * l_src
            else:
                l_pixel = 2*((1-global_alpha) * l1(fus_y, ir_y) + global_alpha * l1(fus_y, vi_y) + global_alpha * (l1(fus_cb, vi_cb) + l1(fus_cr, vi_cr)))
                l_grad = l1(g1_fus, torch.max(g1_ir, g1_vi))
                l_target = lF(fus_y, ir_y) * gaussianBlurConv(g1_ir)
                l_re = ((1-global_alpha) * l_ir_src + global_alpha * l_vi_src).mean()
                l_src = b1 * l_pixel.mean() + b2 * l_grad.mean() + b3 * l_vgg + l_re + b5 * l_target.mean() # #  # l_invary +  + b1 * l_ssim.mean()
                np.append(loss_x, l_src.mean().item())
                FF = (loss_x.mean() + loss_1.mean() + loss_0.mean()) / loss_x.mean()
                l_src = FF * l_src
                
    
            #l_src = b1 * l_pixel.mean() + b2 * l_grad.mean() + b3 * l_vgg + b4 * l_ir_src + b4 * l_vi_src + b5 * l_target.mean() # #  # l_invary +  + b1 * l_ssim.mean()
            if iteration % 300 == 0:
                fus_img = kornia.utils.tensor_to_image(fus_images.cpu()) * 255.
                #fus_images = fus_images.astype(numpy.uint8)
                vi_img = kornia.utils.tensor_to_image(vi_images.cpu()) * 255.
                #vi_images = vi_images.astype(numpy.uint8)
                ir_img = kornia.utils.tensor_to_image(ir_images.cpu()) * 255.
                ir_mark_img = kornia.utils.tensor_to_image(ir_mark.cpu()) * 255.
                vi_mark_img = kornia.utils.tensor_to_image(vi_mark.cpu()) * 255.
                src_ir_images = kornia.utils.tensor_to_image(src_ir_images.cpu()) * 255.
                src_vi_images = kornia.utils.tensor_to_image(src_vi_images.cpu()) * 255.
                
                t = cv2.cvtColor(fus_img[0], cv2.COLOR_RGB2BGR)
                #t = fus_images[0]
                cv2.imwrite('logs/LLVIP/randn_alpha_2/img/%d_%02f_%02f.jpg' % (epoch,alpha,beta), t)
                #t = cv2.cvtColor(ir_images[0], cv2.COLOR_RGB2BGR)
                t = ir_img[0]
                cv2.imwrite('logs/LLVIP/randn_alpha_2/img/ir_%d_%02f_%02f.jpg' % (epoch,alpha,beta), t)
                t = cv2.cvtColor(vi_img[0], cv2.COLOR_RGB2BGR)
                #t = vi_images[0]
                cv2.imwrite('logs/LLVIP/randn_alpha_2/img/vi_%d_%02f_%02f.jpg' % (epoch,alpha,beta), t)
                
                #t = cv2.cvtColor(ir_mark[0], cv2.COLOR_RGB2BGR)
                t = ir_mark_img[0]
                cv2.imwrite('logs/LLVIP/randn_alpha_2/img/ir_mark_%d_%02f_%02f.jpg' % (epoch,alpha,beta), t)
                #t = cv2.cvtColor(vi_mark[0], cv2.COLOR_RGB2BGR)
                t = vi_mark_img[0]
                cv2.imwrite('logs/LLVIP/randn_alpha_2/img/vi_mark_%d_%02f_%02f.jpg' % (epoch,alpha,beta), t)
                #t = cv2.cvtColor(max_mark[0], cv2.COLOR_RGB2BGR)
                t = cv2.cvtColor(src_vi_images[0], cv2.COLOR_RGB2BGR)
                cv2.imwrite('logs/LLVIP/randn_alpha_2/img/revi_%d_%02f_%02f.jpg' % (epoch,alpha,beta), t)
                
                t = cv2.cvtColor(src_ir_images[0], cv2.COLOR_RGB2BGR)
                cv2.imwrite('logs/m3fd/randn_alpha_2/img/reir_%d_%02f_%02f.jpg' % (epoch,alpha,beta), t)
            if ema_1:
                ema_1.update(Generator_train)
    
            if ema_2:
                ema_2.update(ir_Generator_train)
    
            if ema_3:
                ema_3.update(vi_Generator_train)
    
            loss_value = l_src
    
            loss_value.backward()
            opt_generator.step()
            opt_ir_generator.step()
            opt_vi_generator.step()

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                # 'prior_kl_loss': 0.01 * prior_kl_loss.clone().detach().item(),
                                # 'info_loss': 1 * info_loss.clone().detach().item(),
                                # 'loc_info': 1 * loc_info.clone().detach().item(),
                                're': b4 * l_re.item(),
                                'vgg': b3 * l_vgg.item(),
                                'grad': b2 * l_grad.mean().item(),
                                'pixel': b1 * l_pixel.mean().item(),
                                'tar': b5 * l_target.mean().item(),
                                })
            pbar.update(1)

    if ema_1:
        save_state_dict_gen = ema_1.ema.state_dict()
    else:
        save_state_dict_gen = Generator.state_dict()

    # save_state_dict_gen = Generator.state_dict()

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(save_state_dict_gen, os.path.join(save_dir, "_generator_ep%03d-loss%.3f.pth" % (
            epoch + 1, loss / epoch_step)))

    torch.save(save_state_dict_gen, os.path.join(save_dir, "_generator_last_epoch_weights.pth"))