import argparse
import os.path
from os import mkdir
from os.path import join, exists
from argparse import Namespace
import numpy as np
import kornia
import numpy
import torch
from pathlib import Path
import cv2
import time
from nets.feature_s import Generator
from utils.batch_transformers import RGBToYCbCr, RGBToGray, YCbCrToRGB
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)


def parse_opt() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', default='test_imgs/m3fd/', type=str, help='fusion data root path')              # test_imgs/m3fd        test_imgs/LLVIP       test_imgs/MSRS
    parser.add_argument('--dst', default='results/', type=Path, help='fusion images save path')
    parser.add_argument('--weights', type=str, default="logs/m3fd/m3fd.pth", help='pretrained weights path')    # logs/m3fd/m3fd.pth    logs/LLVIP/LLVIP.pth  logs/MSRS/MSRS.pth

    return parser.parse_args()


def img_filter(x: Path) -> bool:
    return x.suffix in ['.png', '.bmp', '.jpg']


if __name__ == '__main__':
    config = parse_opt()
    if not exists(config.dst):
        mkdir(config.dst)

    # init model
    net = Generator()
    # load pretrained weights
    model_dict = net.state_dict()
    pretrained_dict = torch.load(config.weights, map_location='cpu')
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        t = np.shape(v)
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            print(k, np.shape(v))
    pretrained_dict = ''
    model_dict.update(temp_dict)
    temp_dict = ''
    net.load_state_dict(model_dict)
    model_dict = ''
    net.to('cuda:0')
    print('success load')
    ycbcr2rgb = YCbCrToRGB()
    rgb2ycbcr = RGBToYCbCr()
    root = Path(config.src)
    cnt = 0
    torch.cuda.empty_cache()
    List = os.listdir(root / 'Ir')
    A, B = [-100, 0, 100, 50, 40], [-100]
    #A, B = [-100], [-100] # -100 means using lightweight adaptation module
    for name in List:
        name = name.split('/')[-1]
        cnt = cnt + 1
        ir_path = root / 'Ir' / name
        vi_path = root / 'Vis' / name
        if os.path.exists(ir_path) is False:
            print(ir_path)
        if os.path.exists(vi_path) is False:
            print(vi_path)
        ir_c = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
        vi_c = cv2.imread(str(vi_path), cv2.IMREAD_COLOR)
        vi_mark = cv2.imread(str(vi_path).replace('Vis','Vis_annotation'), cv2.IMREAD_COLOR)
        ir_mark = cv2.imread(str(vi_path).replace('Vis','Ir_annotation'), cv2.IMREAD_COLOR)
        
        ir_t = kornia.utils.image_to_tensor(cv2.cvtColor(ir_c, cv2.COLOR_BGR2RGB) / 255.).float()
        vi_t = kornia.utils.image_to_tensor(cv2.cvtColor(vi_c, cv2.COLOR_BGR2RGB) / 255.).float()
        vi_mark = kornia.utils.image_to_tensor(cv2.cvtColor(vi_mark, cv2.COLOR_BGR2RGB) / 255.).float()
        ir_mark = kornia.utils.image_to_tensor(cv2.cvtColor(ir_mark, cv2.COLOR_BGR2RGB) / 255.).float()
        
        ir_c, vi_c = '', ''

        ir_t, vi_t, vi_mark, ir_mark = ir_t.unsqueeze(0).cuda(), vi_t.unsqueeze(0).cuda(), vi_mark.unsqueeze(0).cuda(), ir_mark.unsqueeze(0).cuda()
        vi_ycbcr = rgb2ycbcr(vi_t)
        for alpha in A:
            for beta in B:
                with torch.no_grad():
                    fus_y, fus_cbcr, global_alpha = net(ir=ir_t, vi=vi_t, vi_mark=vi_mark, ir_mark=ir_mark, alpha=alpha/100., beta=beta/100., cnt=1)
                fus = ycbcr2rgb(torch.cat([fus_y,fus_cbcr],dim=1))

                fus = kornia.utils.tensor_to_image(fus.cpu()) * 255.
                fus = cv2.cvtColor(fus, cv2.COLOR_RGB2BGR)
                save = config.dst / str(alpha)
                if not exists(save):
                    mkdir(save)
                cv2.imwrite(str(save / name), fus)
                fus = ''