from EncDec import EncDec
from OpticalFlow import OpticalFlow
from PoseReg import PoseReg
from Iter import Iter
from data_reader import *
from ssim import SSIM

import numpy as np
import torch.backends.cudnn as cudnn
import torch
import sys
import os

from imageio import imwrite


def eval():
    dataIndex = [15,36,65,80,85,87,108,118,148,149,156]

    dataset_dir = '../datasets'
    enc_checkpoint_path = sys.argv[1]
    outDir = sys.argv[2]
    gpu = int(sys.argv[3])

    os.makedirs(outDir, exist_ok=True)

    torch.cuda.set_device(gpu) # change allocation of current GPU
    encoder_decoder = EncDec().cuda()

    # load parameter
    enc_checkpoint = torch.load(enc_checkpoint_path)

    encoder_decoder.load_state_dict(enc_checkpoint['state_dict'])

    cudnn.benchmark = True

    batchSize = len(dataIndex)

    depth_error_result = np.empty([batchSize, 3])
    pose_error_result = np.empty([batchSize, 1])

    output_tmp = miniBatch_generate(dataset_dir, len(dataIndex), False, dataIndex=dataIndex)

    for i in range(len(dataIndex)):
        gtPath = os.path.join(outDir, 'depth_gt_' + str(i) + '.png')
        gtImagePath = os.path.join(outDir, 'image_gt_' + str(i) + '.png')
        gtImage2Path = os.path.join(outDir, 'image2_gt_' + str(i) + '.png')

        imwrite(gtPath, output_tmp['target_depth_first'][i, :, :, :].reshape(192, 256))
        imwrite(gtImagePath, output_tmp['input_image_first'][i, :, :, :].reshape(192, 256, 3))
        imwrite(gtImage2Path, output_tmp['input_image_second'][i, :, :, :].reshape(192, 256, 3))

        input_val = np.zeros([1, 6, 192, 256])
        input_val[0, 0:3, :, :] = output_tmp['input_image_first'][i, :, :, :].transpose(2, 0, 1)
        input_val[0, 3:6, :, :] = output_tmp['input_image_second'][i, :, :, :].transpose(2, 0, 1)
        input_val = torch.from_numpy(input_val).float().cuda()
        input_val = torch.autograd.Variable(input_val)

        # help doogie
        output_val, pose_val = encoder_decoder(input_val)

        outPath = os.path.join(outDir, 'depth_out_' + str(i) + '.png')
        img = output_val.data.cpu().numpy().reshape(192, 256)
        imwrite(outPath, img)

        gt_depth = torch.from_numpy(output_tmp['target_depth_first'][i, :, :, :].transpose(2, 0, 1)).float()
        gt_egomotion = torch.from_numpy(output_tmp['target_egomotion'][i, :]).float()

        depth = torch.autograd.Variable(gt_depth.cuda()).reshape(192, 256)
        egomotion = torch.autograd.Variable(gt_egomotion.cuda()).reshape(7)

        mse_loss = torch.nn.MSELoss()
        depth_loss = mse_loss(output_val, depth)
        pose_loss = torch.sqrt(mse_loss(pose_val, egomotion))

        depth_error_result[i, 0] = depth_loss.data.cpu().numpy()
        pose_error_result[i, 0] = pose_loss.data.cpu().numpy()

        l1_loss = torch.nn.L1Loss()
        depth_loss = l1_loss(output_val, depth)
        depth_error_result[i, 1] = depth_loss.data.cpu().numpy()

    print(depth_error_result)
    print(pose_error_result)

    print(np.mean(depth_error_result, axis=0))
    print(np.mean(pose_error_result, axis=0))

eval()
