import torch
import random
import cv2
import os
import h5py

from dataset_tools import *

train_datasets = [
    'sun3d_train_0.1m_to_0.2m.h5',
    'sun3d_train_0.2m_to_0.4m.h5',
    'sun3d_train_0.8m_to_1.6m.h5',
]

validation_datasets = [
    'sun3d_train_0.4m_to_0.8m.h5',
]


def miniBatch_generate(directory, miniBatch_size=64, for_validation=False):
    # This is hyper-parameter for (image width and height, second frame number bias,
    #  image pair's thresholds)
    width = 256
    height = 192
    target_factor = 1
    width_resized = int(width/target_factor)
    height_resized = int(height/target_factor)

    # define pyTorch tensors of minibatch input/target
    # egomotion(1:3 = translation, 4:6 = rotation, 7 = scale)
    input_image_first = torch.Tensor(miniBatch_size, height, width, 3).numpy()
    input_image_second = torch.Tensor(miniBatch_size, height, width, 3).numpy()
    target_depth_first = torch.Tensor(miniBatch_size, height_resized, width_resized, 1).numpy()
    target_egomotion = torch.Tensor(miniBatch_size, 7).numpy()

    if for_validation == True:
        datasets = validation_datasets
    else:
        datasets = train_datasets

    datasetId = random.randint(0, len(datasets)-1)
    h5file = h5py.File(os.path.join(directory, datasets[datasetId]))

    for i in range(miniBatch_size):
        output_tmp =  readSUN3D_singleData(h5file)
        input_image_first[i, :, :, :] = output_tmp['image_first']
        input_image_second[i, :, :, :] = output_tmp['image_second']
        target_depth_first[i, :, :, :] = output_tmp['depth_first']
        target_egomotion[i, :] = output_tmp['egomotion']

    return {
            'input_image_first': input_image_first,
            'input_image_second': input_image_second,
            'target_depth_first': target_depth_first,
            'target_egomotion': target_egomotion
            }


def readSUN3D_singleData(h5file):
    keys = [k for k in h5file.keys()]
    frameId = random.randint(0, len(keys)-1)

    first_view = read_view(h5file[keys[frameId]].get('frames').get('t0').get('v0'))
    second_view = read_view(h5file[keys[frameId]].get('frames').get('t0').get('v1'))

    # get extrinsic
    first_extrinsic = np.concatenate((first_view.R, first_view.t.reshape(-1, 1)), axis=1)
    second_extrinsic = np.concatenate((second_view.R, second_view.t.reshape(-1, 1)), axis=1)

    extrinsicFirst = np.concatenate((first_extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    extrinsicSecond = np.concatenate((second_extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    extrinsicOut = np.dot(extrinsicSecond, np.linalg.inv(extrinsicFirst))

    motionOut = matToSe3(extrinsicOut)

    egomotion_extended = np.zeros(7)

    egomotion_tmp = motionOut

    scale_tmp = np.linalg.norm(egomotion_tmp[0:3])

    egomotion_tmp[0:3] = egomotion_tmp[0:3] / scale_tmp
    egomotion_extended[0:6] = egomotion_tmp
    egomotion_extended[6] = scale_tmp

    inverseDepth_first = first_view.depth
    inverseDepth_first[first_view.depth != 0] = 1 / inverseDepth_first[first_view.depth != 0]

    first_image = np.array(first_view.image) / 255
    second_image = np.array(second_view.image) / 255

    output_tmp = {}
    output_tmp['egomotion'] = egomotion_extended
    output_tmp['depth_first'] = cv2.resize(inverseDepth_first, (256, 192)).reshape(192, 256, 1)
    output_tmp['image_first'] = cv2.resize(first_image, (256, 192))
    output_tmp['image_second'] = cv2.resize(second_image, (256, 192))

    return output_tmp


def matToSe3(inputMtx):
    R = inputMtx[0:3, 0:3]
    theta = np.arccos((np.trace(R)-1)/2)

    if(theta != 0):
        lnR = theta/(2*np.sin(theta))*(R-R.transpose())
    else:
        lnR = 1/2*(R-R.transpose())

    outputSe3 = np.zeros(6)
    outputSe3[3] = lnR[2, 1]
    outputSe3[4] = lnR[0, 2]
    outputSe3[5] = lnR[1, 0]

    if(theta != 0):
        epsB = (1 - np.cos(theta)) / np.square(theta)
        epsC = (theta - np.sin(theta)) / np.power(theta, 3)
    else:
        epsB = 0.5
        epsC = 1/6

    V = np.eye(3) + epsB*lnR + epsC*lnR*lnR

    outputSe3[0:3] = np.dot(np.linalg.inv(V), inputMtx[0:3, 3])

    return outputSe3
