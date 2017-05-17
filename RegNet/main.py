from EncDec import EncDec
from OpticalFlow import OpticalFlow
from PoseReg import PoseReg
from dataLoad_multiprocess import *
import torch.backends.cudnn as cudnn
import torch
import time


def parsing_minibatch(input1, batch_size):
    input1['input_image_first'] = torch.from_numpy(input1['input_image_first'].transpose(0, 3, 1, 2)).float()
    input1['input_image_second'] = torch.from_numpy(input1['input_image_second'].transpose(0, 3, 1, 2)).float()
    input1['target_depth_first'] = torch.from_numpy(input1['target_depth_first'].transpose(0, 3, 1, 2)).float()
    #input['depth_second'] = torch.from_numpy(input['depth_second']).float()
    input1['target_opticalFlow'] = torch.from_numpy(input1['target_opticalFlow'].transpose(0, 3, 1, 2)).float()
    input1['target_egomotion'] =  torch.from_numpy(input1['target_egomotion']).float()

    return input1

def main():
    encoder_decoder = EncDec().cuda()
    pose_regressor = PoseReg().cuda()
    optical_flow = OpticalFlow().cuda()

    enc_checkpoint = torch.load('encdec.pth')
    pose_checkpoint = torch.load('pose.pth')
    opt_checkpoint = torch.load('optical.pth')

    encoder_decoder.load_state_dict(enc_checkpoint['state_dict'])
    pose_regressor.load_state_dict(pose_checkpoint['state_dict'])
    optical_flow.load_state_dict(opt_checkpoint['state_dict'])

    cudnn.benchmark = True

    lr = 1e-3

    optimizer = torch.optim.Adam([{'params': encoder_decoder.parameters(), 'lr': lr},
                                  {'params': pose_regressor.parameters(), 'lr': lr},
                                  {'params': optical_flow.parameters(), 'lr': lr}],
                                 weight_decay=2e-4)

    i = 0 

    # Call SUN3D directories
    inputDir = listOfDir_generate('/home/dongwoo/Project/dataset/SUN3D')
    batch_size = 16


    # Main training loop
    while (True):
        minibatch_tmp = miniBatch_generate(inputDir,batch_size)
        input = parsing_minibatch(minibatch_tmp,batch_size)
        t = time.time()

        (loss, depth_loss, pose_loss, optical_loss) = \
            train(input, encoder_decoder, pose_regressor, optical_flow, optimizer)
        
        if i % 1 == 0:
            elapse_t = time.time() - t
            print('Iteration: {0}\t'
                  'Elapse time {1:.4f}\t'
                  'Loss {2:.4f}\t'
                  'Depth Loss {3:.4f}\t'
                  'Pose Loss {4:.4f}\t'
                  'Optical Loss {5:.4f}\t'.format(
                      i, elapse_t, loss.data[0],
                      depth_loss.data[0], 
                      pose_loss.data[0], optical_loss.data[0],  
                  ))
        i += 1
        if i % 100 == 0:
            torch.save({'state_dict': encoder_decoder.state_dict()}, 'encdec.pth') 
            torch.save({'state_dict': pose_regressor.state_dict()}, 'pose.pth') 
            torch.save({'state_dict': optical_flow.state_dict()}, 'optical.pth') 


def train(input, encoder_decoder, pose_regressor, optical_flow, optimizer):
    image1 = torch.autograd.Variable(input['input_image_first'].cuda())
    image2 = torch.autograd.Variable(input['input_image_second'].cuda())
    egomotion = torch.autograd.Variable(input['target_egomotion'].cuda())
    opticalflow = torch.autograd.Variable(input['target_opticalFlow'].cuda())
    depth = torch.autograd.Variable(input['target_depth_first'].cuda())

    image_pair = torch.cat((image1, image2), dim=1)

    # get depth
    depth_output = encoder_decoder(image_pair)
    
    image_with_depth = torch.cat((image1, image2, depth_output), dim=1)

    # get pose
    pose_output = pose_regressor(image_with_depth)

    # get opticalflow
    opticalflow_output = optical_flow(pose_output, depth_output)

    mse_loss = torch.nn.MSELoss()

    depth_loss = mse_loss(depth_output, depth) 
    pose_loss = mse_loss(pose_output, egomotion)
    optical_loss = mse_loss(opticalflow_output, opticalflow)

    loss = depth_loss + pose_loss + optical_loss 
    # loss = depth_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    return (loss, depth_loss, pose_loss, optical_loss)
    


if __name__ == '__main__':
    main()
