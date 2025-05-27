
import os,sys,math
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils

from data_RGB import get_test_data, get_validation_data


from CRFamba import CRFamba

from skimage import img_as_ubyte


def test():
    parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

    parser.add_argument('--input_dir', default='/home/u1120210217/thick_cloud/datasets/T-Cloud/test/', type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='./tcloud/', type=str, help='Directory for results')
    parser.add_argument('--weights', default='/home/u1120210217/thick_cloud/test_crfamba_code/cloudremoval/CR-Famba_Model/model_best.pth', type=str,
                      help='Path to weights')

    parser.add_argument('--dataset', default='GoPr', type=str,
                        help='Test Dataset')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    model_restoration = CRFamba()
    print("Total number of param  is ", sum(x.numel() for x in model_restoration.parameters()))
    utils.load_checkpoint(model_restoration,args.weights)

    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration.eval()

    rgb_dir_test = args.input_dir
    print(rgb_dir_test)
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
                             pin_memory=True)


    result_dir = args.result_dir
    utils.mkdir(result_dir)

    for ii, data_test in enumerate(tqdm(test_loader), 0):

        input_    = data_test[0].cuda()
        filenames = data_test[2]
        
        with torch.no_grad():
            restored = model_restoration(input_)
            restored = torch.clamp(restored,0,1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            
            for batch in range(len(restored)):
            
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
            
            


if __name__=='__main__':
    test()



