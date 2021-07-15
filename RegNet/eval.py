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
    dataset_dir = '../datasets'
    enc_checkpoint_path = sys.argv[1]
    iter_checkpoint_path = sys.argv[2]
    outDir = sys.argv[3]
    gpu = int(sys.argv[4])

    os.makedirs(outDir, exist_ok=True)

    torch.cuda.set_device(gpu) # change allocation of current GPU
    encoder_decoder = EncDec().cuda()
    iterator = Iter().cuda()

    # load parameter
    enc_checkpoint = torch.load(enc_checkpoint_path)
    iter_checkpoint = torch.load(iter_checkpoint_path)

    encoder_decoder.load_state_dict(enc_checkpoint['state_dict'])
    iterator.load_state_dict(iter_checkpoint['state_dict'])

    cudnn.benchmark = True

    batchSize = 1705

    dataIndex = [x for x in range(batchSize)]
    depth_error_result = np.empty([batchSize, 3])
    pose_error_result = np.empty([batchSize, 1])

    for iterVal in dataIndex:
        output_tmp = miniBatch_generate(dataset_dir, 1, False, [iterVal])
        gtPath = os.path.join(outDir, 'depth_gt_' + str(iterVal) + '.png')
        gtImagePath = os.path.join(outDir, 'image_gt_' + str(iterVal) + '.png')
        gtImage2Path = os.path.join(outDir, 'image2_gt_' + str(iterVal) + '.png')

        imwrite(gtPath, output_tmp['target_depth_first'].reshape(192, 256))
        imwrite(gtImagePath, output_tmp['input_image_first'].reshape(192, 256, 3))
        imwrite(gtImage2Path, output_tmp['input_image_second'].reshape(192, 256, 3))

        input_val = np.zeros([1, 6, 192, 256])
        input_val[0, 0:3, :, :] = output_tmp['input_image_first'].transpose(0, 3, 1, 2)
        input_val[0, 3:6, :, :] = output_tmp['input_image_second'].transpose(0, 3, 1, 2)
        input_val = torch.from_numpy(input_val).float().cuda()
        input_val = torch.autograd.Variable(input_val)

        # help doogie
        output_val, _ = encoder_decoder(input_val)
        output_val = torch.cat((input_val, output_val), dim=1)
        output_val, pose_val = iterator(output_val)

        outPath = os.path.join(outDir, 'depth_out_' + str(iterVal) + '.png')
        img = output_val.data.cpu().numpy().reshape(192, 256)
        imwrite(outPath, img)

        gt_depth = torch.from_numpy(output_tmp['target_depth_first'].transpose(0, 3, 1, 2)).float()
        gt_egomotion = torch.from_numpy(output_tmp['target_egomotion']).float()

        depth = torch.autograd.Variable(gt_depth.cuda())
        egomotion = torch.autograd.Variable(gt_egomotion.cuda())

        mse_loss = torch.nn.MSELoss()
        depth_loss = mse_loss(output_val, depth)
        pose_loss = torch.sqrt(mse_loss(pose_val, egomotion))

        depth_error_result[iterVal, 0] = depth_loss.data.cpu().numpy()
        pose_error_result[iterVal, 0] = pose_loss.data.cpu().numpy()

        l1_loss = torch.nn.L1Loss()
        depth_loss = l1_loss(output_val, depth)
        depth_error_result[iterVal, 1] = depth_loss.data.cpu().numpy()

        log_loss = torch.nn.NLLLoss()
        depth_loss = log_loss(output_val, depth)
        depth_error_result[iterVal, 2] = depth_loss.data.cpu().numpy()

    print(depth_error_result)
    print(pose_error_result)

    print(np.mean(depth_error_result, axis=0))
    print(np.mean(pose_error_result, axis=0))

eval()
