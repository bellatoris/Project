from EncDec import EncDec
from OpticalFlow import OpticalFlow
from PoseReg import PoseReg
from data_reader import *
from ssim import SSIM

import numpy as np
import torch.backends.cudnn as cudnn
import torch
import time
import os
import sys
import re
import random

from imageio import imwrite


dataset_dir = '../datasets'
_dataIndex = [15,36,65,80,85,87,108,118,148,149,156]


def parsing_minibatch(input1, batch_size):
    input1['input_image_first'] = torch.from_numpy(input1['input_image_first'].transpose(0, 3, 1, 2)).float()
    input1['input_image_second'] = torch.from_numpy(input1['input_image_second'].transpose(0, 3, 1, 2)).float()
    input1['target_depth_first'] = torch.from_numpy(input1['target_depth_first'].transpose(0, 3, 1, 2)).float()
    input1['target_egomotion'] = torch.from_numpy(input1['target_egomotion']).float()

    return input1


def validateModel_simple(encoder_decoder, iteration, loss_mode):
    if loss_mode == 'l1':
        outDir = './validation_l2/'
    elif loss_mode == 'l1ssim':
        outDir = './validation_l1ssim/'
    else:
        outDir = './validation_l1/'

    os.makedirs(outDir, exist_ok=True)

    if len(_dataIndex) == 0:
        dataIndex = [random.randint(0, 1000) for _ in range(5)]
    else:
        dataIndex = _dataIndex

    output_tmp = miniBatch_generate(dataset_dir, len(dataIndex), False, dataIndex=dataIndex)

    for i in range(len(dataIndex)):
        gtPath = outDir + 'depth_gt_' + str(i) + '_' + str(iteration) + '.png'
        gtImagePath = outDir + 'image_gt_' + str(i) + '_' + str(iteration) + '.png'
        gtImage2Path = outDir + 'image2_gt_' + str(i) + '_' + str(iteration) + '.png'
        imwrite(gtPath, output_tmp['target_depth_first'][i, :, :, :].reshape(192, 256))
        imwrite(gtImagePath, output_tmp['input_image_first'][i, :, :, :].reshape(192, 256, 3))
        imwrite(gtImage2Path, output_tmp['input_image_second'][i, :, :, :].reshape(192, 256, 3))

        input_val = np.zeros([1, 6, 192, 256])
        input_val[0, 0:3, :, :] = output_tmp['input_image_first'][i, :, :, :].transpose(0, 3, 1, 2)
        input_val[0, 3:6, :, :] = output_tmp['input_image_second'][i, :, :, :].transpose(0, 3, 1, 2)
        input_val = torch.from_numpy(input_val).float().cuda()
        input_val = torch.autograd.Variable(input_val)

        # help doogie
        output_val, _ = encoder_decoder(input_val)

        outPath = outDir + 'depth_out_' + str(i) + '_' + str(iteration) + '.png'
        img = output_val.data.cpu().numpy().reshape(192, 256)
        imwrite(outPath, img)

    return 0


def main():
    loss_mode = sys.argv[1]

    if len(sys.argv) == 3:
        model_name = sys.argv[2]
    else:
        model_name = None

    if loss_mode == 'l1':
        outDir = './model_l2/'
        gpu = 0
    elif loss_mode == 'l1ssim':
        outDir = './model_l1ssim/'
        gpu = 1
    else:
        outDir = './model_l1/'
        gpu = 2

    os.makedirs(outDir, exist_ok=True)

    torch.cuda.set_device(gpu) # change allocation of current GPU
    encoder_decoder = EncDec().cuda()

    if model_name is not None:
        # load parameter
        enc_checkpoint = torch.load(os.path.join(outDir, model_name))
        encoder_decoder.load_state_dict(enc_checkpoint['state_dict'])
        i = re.findall(r'\d+', model_name)[0]
    else:
        i = 0

    cudnn.benchmark = True

    lr = 1e-4

    parameters_to_train = list(encoder_decoder.parameters())
    optimizer = torch.optim.Adam(parameters_to_train, lr=lr, weight_decay=2e-4)
    ssim = SSIM()

    # Call SUN3D directories
    batch_size = 64

    # Main training loop
    while (True):
        i += 1

        t = time.time()

        minibatch_tmp = miniBatch_generate(dataset_dir, batch_size, dataIndex=_dataIndex)
        input = parsing_minibatch(minibatch_tmp, batch_size)

        if (i % 10000 == 0):
            optimizer.lr = 1e-5

        (loss, depth_loss, pose_loss) = train(input, encoder_decoder, optimizer, ssim, loss_mode)

        if i % 10 == 0:
            elapse_t = time.time() - t
            print('Iteration: {0}\t'
                  'Elapse time {1:.4f}\t'
                  'Loss {2:.4f}\t'
                  'Depth Loss {3:.4f}\t'
                  'Pose Loss {4:.4f}\t'.format(
                      i, elapse_t, loss.item(),
                      depth_loss.item(),
                      pose_loss.item(),
                  ))

        if i % 100 == 0:
            torch.save({'state_dict': encoder_decoder.state_dict()}, os.path.join(outDir, 'encdec_{0}.pth'.format(i)))
            validateModel_simple(encoder_decoder, i, loss_mode)


def train(input, encoder_decoder, optimizer, ssim, loss_mode):
    image1 = torch.autograd.Variable(input['input_image_first'].cuda())
    image2 = torch.autograd.Variable(input['input_image_second'].cuda())
    egomotion = torch.autograd.Variable(input['target_egomotion'].cuda())
    depth = torch.autograd.Variable(input['target_depth_first'].cuda())

    image_pair = torch.cat((image1, image2), dim=1)

    if loss_mode == 'l2':
        loss_function = torch.nn.MSELoss()
    else:
        loss_function = torch.nn.L1Loss()

    # get depth
    depth_output, pose_output = encoder_decoder(image_pair)

    depth_loss = loss_function(depth_output, depth)
    pose_loss = loss_function(pose_output, egomotion)

    if loss_mode == 'l2' or loss_mode == 'l1':
        loss = depth_loss + pose_loss
    else:
        ssim_loss = ssim(depth_output, depth).mean()
        loss = ssim_loss + depth_loss + pose_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return (loss, depth_loss, pose_loss)


if __name__ == '__main__':
    main()
