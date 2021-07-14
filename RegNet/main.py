from EncDec import EncDec
from OpticalFlow import OpticalFlow
from PoseReg import PoseReg
from Iter import Iter
from data_reader import *
from ssim import SSIM

import numpy as np
import torch.backends.cudnn as cudnn
import torch
import time
import os
import sys

from imageio import imwrite


dataset_dir = '../datasets'


def parsing_minibatch(input1, batch_size):
    input1['input_image_first'] = torch.from_numpy(input1['input_image_first'].transpose(0, 3, 1, 2)).float()
    input1['input_image_second'] = torch.from_numpy(input1['input_image_second'].transpose(0, 3, 1, 2)).float()
    input1['target_depth_first'] = torch.from_numpy(input1['target_depth_first'].transpose(0, 3, 1, 2)).float()
    input1['target_egomotion'] = torch.from_numpy(input1['target_egomotion']).float()

    return input1


def validateModel_simple(encoder_decoder, iter, iteration, use_l2loss=False):
    if use_l2loss:
        outDir = './validation_l2/'
    else:
        outDir = './validation/'
    os.makedirs(outDir, exist_ok=True)

    for iterVal in range(5):
        output_tmp = miniBatch_generate(dataset_dir, 1, True)
        gtPath = outDir + 'depth_gt_' + str(iterVal) + '_' + str(iteration) + '.png'
        gtImagePath = outDir + 'image_gt_' + str(iterVal) + '_' + str(iteration) + '.png'
        gtImage2Path = outDir + 'image2_gt_' + str(iterVal) + '_' + str(iteration) + '.png'
        if not os.path.isfile(gtPath):
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
        output_val, _ = iter(output_val)

        outPath = outDir + 'depth_out_' + str(iterVal) + '_' + str(iteration) + '.png'
        img = output_val.data.cpu().numpy().reshape(192, 256)
        imwrite(outPath, img)

    return 0


def main():
    use_l2loss = sys.argv[1] == 'true'
        
    if use_l2loss:
        outDir = './model_l2/'
        gpu = 1
    else:
        outDir = './model/'
        gpu = 0
    os.makedirs(outDir, exist_ok=True)

    torch.cuda.set_device(gpu) # change allocation of current GPU
    encoder_decoder = EncDec().cuda()
    iterator = Iter().cuda()

    # load parameter
    # enc_checkpoint = torch.load('encdec.pth')
    # pose_checkpoint = torch.load('pose.pth')
    # opt_checkpoint = torch.load('optical.pth')
    # iter_checkpoint = torch.lead('iter.pth')

    # encoder_decoder.load_state_dict(enc_checkpoint['state_dict'])
    # pose_regressor.load_state_dict(pose_checkpoint['state_dict'])
    # optical_flow.load_state_dict(opt_checkpoint['state_dict'])
    # iter.load_state_dict(iter_checkpoint['state_dict'])

    cudnn.benchmark = True

    lr = 1e-4

    parameters_to_train = list(iterator.parameters())
    parameters_to_train += list(encoder_decoder.parameters())
    optimizer = torch.optim.Adam(parameters_to_train, lr=lr, weight_decay=2e-4)

    ssim = SSIM()

    # Call SUN3D directories
    batch_size = 64

    i = 0
    # Main training loop
    while (True):
        i += 1

        t = time.time()

        minibatch_tmp = miniBatch_generate(dataset_dir, batch_size)
        input = parsing_minibatch(minibatch_tmp, batch_size)

        if (i % 10000 == 0):
            optimizer.lr = 1e-5

        (loss, depth_loss, pose_loss) = train(input, encoder_decoder, iterator, optimizer, ssim, use_l2loss)

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

        if i % 1000 == 0:
            torch.save({'state_dict': encoder_decoder.state_dict()}, os.path.join(outDir, 'encdec_{0}.pth'.format(i)))
            torch.save({'state_dict': iterator.state_dict()}, os.path.join(outDir, 'iter_{0}.pth'.format(i)))
            validateModel_simple(encoder_decoder, iterator, i, use_l2loss)


def train(input, encoder_decoder, iterator, optimizer, ssim, use_l2loss=False):
    image1 = torch.autograd.Variable(input['input_image_first'].cuda())
    image2 = torch.autograd.Variable(input['input_image_second'].cuda())
    egomotion = torch.autograd.Variable(input['target_egomotion'].cuda())
    depth = torch.autograd.Variable(input['target_depth_first'].cuda())

    image_pair = torch.cat((image1, image2), dim=1)

    if use_l2loss:
        loss_function = torch.nn.MSELoss()
    else:
        loss_function = torch.nn.L1Loss()

    # get depth
    depth_output, pose_output = encoder_decoder(image_pair)

    # first_iter
    image_with_depth = torch.cat((image1, image2, depth_output), dim=1)
    depth_output, pose_output = iterator(image_with_depth)

    depth_loss = loss_function(depth_output, depth)
    pose_loss = loss_function(pose_output, egomotion)


    if use_l2loss:
        loss = depth_loss + pose_loss
    else:
        ##ssim
        # ssim_loss = ssim(depth_output, depth).mean()
        # loss = 0.85 * ssim_loss + 0.15 * depth_loss + pose_loss
        loss = depth_loss + pose_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return (loss, depth_loss, pose_loss)


if __name__ == '__main__':
    main()
