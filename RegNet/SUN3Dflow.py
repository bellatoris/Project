import numpy as np
import os
import re
from imageio import imread
from imageio import imwrite
from scipy.spatial import ConvexHull
import cv2


def SUN3Dflow_py(sequenceDir, sequenceName, frameID, frameIDAnother):
    SUN3Dpath = sequenceDir

    # read intrinsic
    intrinsic = os.path.join(SUN3Dpath, sequenceName, 'intrinsics.txt')
    intrinsic = np.loadtxt(intrinsic)

    K = intrinsic

    # file list
    imageFiles = sorted(os.listdir(os.path.join(SUN3Dpath,
                                                sequenceName,
                                                'image/')))
    depthFiles = sorted(os.listdir(os.path.join(SUN3Dpath,
                                                sequenceName,
                                                'depth/')))
    extrinsicFiles = sorted(os.listdir(os.path.join(SUN3Dpath,
                                                    sequenceName,
                                                    'extrinsics/')))

    # read the latest version of extrinsic parameters for cameras
    extrinsic = os.path.join(SUN3Dpath,
                             sequenceName,
                             'extrinsics',
                             extrinsicFiles[-1])
    extrinsic = np.loadtxt(extrinsic)
    extrinsic = np.reshape(extrinsic, (-1, 3, 4))
    extrinsicsC2W = np.transpose(extrinsic, (1, 2, 0))

    # read time stamp
    imageFrameID = np.zeros([len(imageFiles)])
    imageTimestamp = np.zeros([len(imageFiles)])
    for i in range(len(imageFiles)):
        id_time = re.findall('\d+', imageFiles[i])
        imageFrameID[i] = int(id_time[0])
        imageTimestamp[i] = int(id_time[1])

    depthFrameID = np.zeros([len(depthFiles)])
    depthTimestamp = np.zeros([len(depthFiles)])
    for i in range(len(depthFiles)):
        id_time = re.findall('\d+', depthFiles[i])
        depthFrameID[i] = int(id_time[0])
        depthTimestamp[i] = int(id_time[1])

    # synchronize: find a depth for each image
    frameCount = len(imageFiles)
    IDimage2depth = np.zeros([frameCount])
    for i in range(frameCount):
        IDimage2depth[i] = np.argmin(abs(depthTimestamp - imageTimestamp[i]))

    depth = os.path.join(SUN3Dpath,
                         sequenceName,
                         'depth',
                         depthFiles[int(IDimage2depth[frameID - 1])])
    depth = depthRead(depth)

    depthAnother = os.path.join(SUN3Dpath,
                                sequenceName,
                                'depth',
                                depthFiles[int(IDimage2depth[frameIDAnother - 1])])
    depthAnother = depthRead(depthAnother)

    # 3D warping using depth and camera parameters
    XYZcamera = depth2XYZcamera(K, depth)

    valid = XYZcamera[:, :, 3].transpose()
    valid = valid.reshape(valid.size)
    valid = valid.nonzero()[0]

    XYZcamera = XYZcamera.transpose(1, 0, 2).reshape(
        int(XYZcamera.size / 4), 4).transpose()
    XYZcamera = XYZcamera[0:3, valid]
#    XYZcamera = XYZcamera[XYZcamera[:, :, 3] != 0].transpose()
#    XYZcamera = XYZcamera[0:3,:]

    XYZworld = transformPointCloud(XYZcamera, extrinsicsC2W[:, :, frameID - 1])

    XYZcameraAnother = transformPointCloudInverse(
        XYZworld, extrinsicsC2W[:, :, frameIDAnother - 1])

    xyProject = np.dot(K, XYZcameraAnother)

    projectDepth = np.zeros([480, 640])
    projectDepth = projectDepth.transpose().reshape(projectDepth.size)
    projectDepth[valid] = xyProject[2, :]
    projectDepth = projectDepth.reshape(640, 480).transpose()

    good3D = (np.absolute(projectDepth - depthAnother) < 0.05) + 0  # double??

    good3D_vec = good3D.transpose().reshape(good3D.size)

    validgood = good3D_vec[valid].ravel().nonzero()
    valid = valid[validgood]

    gridX, gridY = np.meshgrid(
        np.array(range(0, 640)), np.array(range(0, 480)))

    gridX = gridX.transpose().reshape(gridX.size)
    gridY = gridY.transpose().reshape(gridY.size)

    gridX = gridX[valid]
    gridY = gridY[valid]

    xyProject = xyProject[0:2, validgood[0]] / \
        np.tile(xyProject[2, validgood[0]], (2, 1))

    u = xyProject[0, :] - gridX
    v = xyProject[1, :] - gridY

    F_gt = np.zeros([480, 640, 3])

    F_gt1 = np.zeros([480, 640]).reshape(-1)
    F_gt1[valid] = u
    F_gt1 = F_gt1.reshape(640, 480).transpose()

    F_gt2 = np.zeros([480, 640]).reshape(-1)
    F_gt2[valid] = v
    F_gt2 = F_gt2.reshape(640, 480).transpose()

    F_gt3 = np.zeros([480, 640]).reshape(-1)
    F_gt3[valid] = 1
    F_gt3 = F_gt3.reshape(640, 480).transpose()

    F_gt[:, :, 0] = F_gt1
    F_gt[:, :, 1] = F_gt2
    F_gt[:, :, 2] = F_gt3

    # camera distance
    extrinsicFirst = extrinsicsC2W[:, 4 - 1, frameID - 1]
    extrinsicSecond = extrinsicsC2W[:, 4 - 1, frameIDAnother - 1]
    camDistance = np.linalg.norm(extrinsicFirst - extrinsicSecond)

    # overlap ratio
    height = depth.shape[0]
    width = depth.shape[1]
    if (xyProject.shape[1] > 3):
        areaHull = ConvexHull(xyProject[0:2, :].transpose()).volume
    else:
        areaHull = 0
    overlapRatio = areaHull / (depth.shape[0] * depth.shape[1])

    # photo consistency error (PCE)
    gridX, gridY = np.meshgrid(
        np.array(range(0, 640)), np.array(range(0, 480)))
    image_first = imread(os.path.join(SUN3Dpath,
                                      sequenceName,
                                      'image',
                                      imageFiles[frameID - 1]))
    image_first = image_first / 255
    image_second = imread(os.path.join(SUN3Dpath,
                                       sequenceName,
                                       'image',
                                       imageFiles[frameIDAnother - 1]))
    image_second = image_second / 255
    image_warped = np.zeros([height, width, 3])
    """
    for iterChannel in range(3):

        point = np.concatenate((gridX.reshape(gridX.size,1),gridY.reshape(gridY.size,1)),axis=1)
        value =image_first[:, :, iterChannel].reshape(-1)



        image_tmp = scipy.interpolate.griddata(point, value,
                                               (gridX - F_gt[:, :, 0], gridY - F_gt[:, :, 1]),
                                               method='cubic')

        image_tmp[(F_gt[:, :, 2] == 0).nonzero()] = \
            image_second[:, :, iterChannel][(F_gt[:, :, 2] == 0).nonzero()]
        image_warped[:, :, iterChannel] = image_tmp

    PCE = np.sqrt(np.nansum(np.square(image_second[:] - image_warped[:]))) / (height * width)
    """
    PCE = 0
    # Output adjust
    Flow_gt = F_gt[:, :, 0:2]
    extrinsicFirst = np.concatenate((extrinsicsC2W[:, :, frameID - 1],
                                     np.array([0, 0, 0, 1]).reshape(1, 4)),
                                    axis=0)
    extrinsicSecond = np.concatenate((extrinsicsC2W[:, :, frameIDAnother - 1],
                                      np.array([0, 0, 0, 1]).reshape(1, 4)),
                                     axis=0)
    extrinsicOut = np.dot(np.linalg.inv(extrinsicSecond), extrinsicFirst)

    motionOut = matToSe3(extrinsicOut)

    inverseDepth_first = depth
    inverseDepth_first[depth != 0] = 1 / inverseDepth_first[depth != 0]

    # inverseDepth_first

    # Reshape to half-size
    resize_factor = 1/2.5

    output = {}
    """
    output['image_first'] = scipy.misc.imresize(image_first, resize_factor)
    output['image_second'] = scipy.misc.imresize(image_second, resize_factor)
    output['depth_first'] = scipy.misc.imresize(inverseDepth_first, resize_factor).reshape(192,256,1)
    Flow_gt0 = scipy.misc.imresize(Flow_gt[:,:,0], resize_factor).reshape(192,256,1)
    Flow_gt1 = scipy.misc.imresize(Flow_gt[:,:,1], resize_factor).reshape(192,256,1)
    output['opticalFlow'] = np.concatenate((Flow_gt0,Flow_gt1),axis=2)
    output['egomotion'] = motionOut
    """
    output['image_first'] = cv2.resize(
        image_first, (256, 192))  # (image_first, resize_factor)
    output['image_second'] = cv2.resize(image_second, (256, 192))
    output['depth_first'] = cv2.resize(
        inverseDepth_first, (256, 192)).reshape(192, 256, 1)
    Flow_gt0 = cv2.resize(Flow_gt[:, :, 0], (256, 192)).reshape(192, 256, 1)
    Flow_gt1 = cv2.resize(Flow_gt[:, :, 1], (256, 192)).reshape(192, 256, 1)
    output['opticalFlow'] = np.concatenate((Flow_gt0, Flow_gt1), axis=2)
    output['egomotion'] = motionOut

    output['pce'] = PCE
    output['overlapRatio'] = overlapRatio
    output['camDist'] = camDistance

    return output


def depthRead(filename):
    depth = imread(filename).astype('uint16')
    depth = np.bitwise_or(depth >> 3, depth << 13)
    depth = depth.astype(float) / 1000

    return depth


def depth2XYZcamera(K, depth):
    XYZcamera = np.zeros([480, 640, 4])
    x, y = np.meshgrid(np.array(range(0, 640)), np.array(range(0, 480)))
    XYZcamera[:, :, 0] = (x-K[0, 2])*(depth/K[0, 0])
    XYZcamera[:, :, 1] = (y-K[1, 2])*(depth/K[1, 1])
    XYZcamera[:, :, 2] = depth
    XYZcamera[:, :, 3] = (depth != 0) + 0

    return XYZcamera


def transformPointCloud(XYZ, Rt):
    XYZtransform = np.dot(Rt[0:3, 0:3], XYZ) + \
        np.tile(Rt[0:3, 3].reshape(3, 1), (1, XYZ.shape[1]))

    return XYZtransform


def transformPointCloudInverse(XYZ, Rt):
    XYZtransform = np.dot(np.linalg.inv(
        Rt[0:3, 0:3]), XYZ - np.tile(Rt[0:3, 3].reshape(3, 1), (1, XYZ.shape[1])))

    return XYZtransform


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


"""


output = SUN3Dflow_py('', '/home/dongwoo/Project/dataset/SUN3D/brown_cogsci_3/brown_cogsci_3', 1, 10)
print('success')
img_height = 192
img_width = 256
output['image_first'] =np.array(output['image_first'].reshape(img_height, img_width, 3))
output['image_second'] = np.array(output['image_second'].reshape(img_height,img_width,3))
output['opticalFlow'] = np.array(output['opticalFlow'].reshape(img_height,img_width,2))
output['egomotion'] = np.array(output['egomotion']).reshape(6)
output['depth_first'] = np.array(output['depth_first'].reshape(img_height,img_width,1))


imwrite('image_first.png',output['image_first'] )
imwrite('image_second.png',output['image_second'] )
imwrite('opticalFlow_u.png',output['opticalFlow'][:,:,0] )
imwrite('opticalFlow_v.png',output['opticalFlow'][:,:,1] )
imwrite('depth_first.png',output['depth_first'][:,:,0] )
"""
