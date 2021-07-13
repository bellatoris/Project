from EncDec import EncDec
from OpticalFlow import OpticalFlow
from PoseReg import PoseReg
from Iter import Iter
from data_reader import *
from ssim import SSIM

import torch.backends.cudnn as cudnn
import torch
import time

import numpy as np
from imageio import imwrite
import os.path


def parsing_minibatch(input1, batch_size):
    input1['input_image_first'] = torch.from_numpy(input1['input_image_first'].transpose(0, 3, 1, 2)).float()
    input1['input_image_second'] = torch.from_numpy(input1['input_image_second'].transpose(0, 3, 1, 2)).float()
    input1['target_depth_first'] = torch.from_numpy(input1['target_depth_first'].transpose(0, 3, 1, 2)).float()
    input1['target_egomotion'] = torch.from_numpy(input1['target_egomotion']).float()

    return input1


def validateModel_simple(encoder_decoder, iter, iteration):
    dataDir = '/home/dongwoo/Project/dataset/SUN3D_validate/harvard_c11/hv_c11_2/'
    outDir = './validation/'

    for iterVal in range(5):
        frameID = 1 + iterVal*100
        frameIDAnother = 15 + iterVal*100
        output_tmp = dataLoad_SUN3D(dataDir, frameID, frameIDAnother, '', False)
        gtPath = outDir+'depth_gt_' + str(iterVal) + '.png'
        if not os.path.isfile(gtPath):
            imwrite(gtPath, output_tmp['depth_first'].reshape(192, 256))

        input_val = np.zeros([1, 6, 192, 256])
        input_val[:, 0:3, :, :] = output_tmp['image_first'].transpose(2, 0, 1)
        input_val[:, 3:6, :, :] = output_tmp['image_second'].transpose(2, 0, 1)
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

    lr = 1e-3

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

        minibatch_tmp = miniBatch_generate('', batch_size)
        input = parsing_minibatch(minibatch_tmp, batch_size)

        if (i % 10000 == 0):
            optimizer.lr = 1e-4

        (loss, depth_loss, pose_loss) = train(input, encoder_decoder, iterator, optimizer, ssim)

        if i % 10 == 0:
            elapse_t = time.time() - t
            print('Iteration: {0}\t'
                  'Elapse time {1:.4f}\t'
                  'Loss {2:.4f}\t'
                  'Depth Loss {3:.4f}\t'
                  'Pose Loss {4:.4f}\t'.format(
                      i, elapse_t, loss.data[0],
                      depth_loss.data[0],
                      pose_loss.data[0],
                  ))

        if i % 10000 == 0:
            torch.save({'state_dict': encoder_decoder.state_dict()}, 'encdec_{0}.pth'.format(i))
            torch.save({'state_ditct': iterator.state_dict()}, 'iter_{0}.pth'.format(i))
            validateModel_simple(encoder_decoder, iterator, i)


def train(input, encoder_decoder, iterator, optimizer, ssim):
    image1 = torch.autograd.Variable(input['input_image_first'].cuda())
    image2 = torch.autograd.Variable(input['input_image_second'].cuda())
    egomotion = torch.autograd.Variable(input['target_egomotion'].cuda())
    depth = torch.autograd.Variable(input['target_depth_first'].cuda())

    image_pair = torch.cat((image1, image2), dim=1)

    l1_loss = torch.nn.L1Loss()

    # get depth
    depth_output, pose_output = encoder_decoder(image_pair)

    # first_iter
    image_with_depth = torch.cat((image1, image2, depth_output), dim=1)
    depth_output, pose_output = iterator(image_with_depth)

    depth_loss = l1_loss(depth_output, depth) * 100
    pose_loss = l1_loss(pose_output, egomotion) * 100

    ##ssim
    #ssim_loss = ssim(depth_output, depth).mean(1, True)

    # loss = 0.85 * ssim_loss + 0.15 * depth_loss + pose_loss
    loss = depth_loss + pose_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return (loss, depth_loss, pose_loss)


if __name__ == '__main__':
    main()
