from EncDec import EncDec
from OpticalFlow import OpticalFlow
from PoseReg import PoseReg
from dataLoad_multiprocess import *
import torch.backends.cudnn as cudnn
import torch
import time

import numpy as np
from scipy.misc import imsave
import os.path


def parsing_minibatch(input1, batch_size):
    input1['input_image_first'] = torch.from_numpy(input1['input_image_first'].transpose(0, 3, 1, 2)).float()
    input1['input_image_second'] = torch.from_numpy(input1['input_image_second'].transpose(0, 3, 1, 2)).float()
    input1['target_depth_first'] = torch.from_numpy(input1['target_depth_first'].transpose(0, 3, 1, 2)).float()
    #input['depth_second'] = torch.from_numpy(input['depth_second']).float()
    input1['target_opticalFlow'] = torch.from_numpy(input1['target_opticalFlow'].transpose(0, 3, 1, 2)).float()
    input1['target_egomotion'] =  torch.from_numpy(input1['target_egomotion']).float()

    return input1

def validateModel_simple(encoder_decoder, iteration):
    #dataDir = '/home/dongwoo/Project/dataset/SUN3D_validate/harvard_c11/hv_c11_2/'
    dataDir = '/home/dongwoo/Project/dataset/SUN3D_validate/harvard_c11/hv_c11_2/'
    outDir = './validation/'
    
    for iterVal in range(5):

        frameID = 1 + iterVal*100
        frameIDAnother =15 + iterVal*100
#        frameID = 642
#        frameIDAnother =646
        output_tmp = dataLoad_SUN3D(dataDir,frameID,frameIDAnother,'',False)
        gtPath = outDir+'depth_gt_' + str(iterVal) + '.png'
        if not os.path.isfile(gtPath):
            imsave(gtPath, output_tmp['depth_first'].reshape(192,256))

        input_val = np.zeros([1,6,192,256])
        input_val[:,0:3,:,:] = output_tmp['image_first'].transpose(2,0,1); input_val[:,3:6,:,:] = output_tmp['image_second'].transpose(2,0,1)
        input_val = torch.from_numpy(input_val).float().cuda()
        
        # help doogie
        output_val, _ = encoder_decoder(torch.autograd.Variable(input_val))

        outPath = outDir + 'depth_out_' + str(iterVal) + '_' + str(iteration) + '.png'
        img = output_val.data.cpu().numpy().reshape(192,256)
        imsave(outPath, img)

    return 0


def main():
    encoder_decoder = EncDec().cuda()
    pose_regressor = PoseReg().cuda()
    optical_flow = OpticalFlow().cuda()

    # load parameter
    #enc_checkpoint = torch.load('encdec.pth')
    # pose_checkpoint = torch.load('pose.pth')
    # opt_checkpoint = torch.load('optical.pth')

    #encoder_decoder.load_state_dict(enc_checkpoint['state_dict'])
    # pose_regressor.load_state_dict(pose_checkpoint['state_dict'])
    # optical_flow.load_state_dict(opt_checkpoint['state_dict'])

    cudnn.benchmark = True

    lr = 1e-3

    optimizer = torch.optim.Adam(encoder_decoder.parameters(), lr=lr, weight_decay=2e-4)

    i = 1

    # Call SUN3D directories
    inputDir = listOfDir_generate('/home/dongwoo/Project/dataset/SUN3D')
    batch_size = 16


    # Main training loop
    while (True):
        t = time.time()
        minibatch_tmp = miniBatch_generate(inputDir,batch_size)
        input = parsing_minibatch(minibatch_tmp,batch_size)

        if (i % 10000 == 0):
            optimizer.lr = 1e-4

        (loss, depth_loss, pose_loss) = \
            train(input, encoder_decoder, pose_regressor, optical_flow, optimizer)
        
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
        i += 1
        if i % 100 == 0:
            torch.save({'state_dict': encoder_decoder.state_dict()}, 'encdec.pth') 
            # torch.save({'state_dict': pose_regressor.state_dict()}, 'pose.pth') 
            # torch.save({'state_dict': optical_flow.state_dict()}, 'optical.pth') 
            validateModel_simple(encoder_decoder,i)

def train(input, encoder_decoder, pose_regressor, optical_flow, optimizer):
    image1 = torch.autograd.Variable(input['input_image_first'].cuda())
    image2 = torch.autograd.Variable(input['input_image_second'].cuda())
    egomotion = torch.autograd.Variable(input['target_egomotion'].cuda())
    opticalflow = torch.autograd.Variable(input['target_opticalFlow'].cuda())
    depth = torch.autograd.Variable(input['target_depth_first'].cuda())

    image_pair = torch.cat((image1, image2), dim=1)

    # get depth
    depth_output, pose_output = encoder_decoder(image_pair)
    
    # image_with_depth = torch.cat((image1, image2, depth_output), dim=1)

    # get pose
    # pose_output = pose_regressor(image_with_depth)

    # get opticalflow
    # opticalflow_output = optical_flow(pose_output, depth_output)

    mse_loss = torch.nn.MSELoss()

    depth_loss = mse_loss(depth_output, depth) * 100
    pose_loss = mse_loss(pose_output, egomotion) * 100
    # optical_loss = mse_loss(opticalflow_output, opticalflow)

    loss = depth_loss + pose_loss
    # loss = depth_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    return (loss, depth_loss, pose_loss)
    


if __name__ == '__main__':
    main()
