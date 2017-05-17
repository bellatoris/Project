import torch
import numpy as np
from os import listdir
from glob import glob
import random
import scipy.misc
from scipy.misc import toimage

from SUN3Dflow import *
from multiprocessing import Process, Queue
import time
"""
 Function to load SUN3D Data pair
 Inputs : SUN3D dataset path, Name of sequence, index(number) of the first frame, index(number) of the second frame
 Outputs : Image of the first/second frame(H x W x 3), Optical Flow(H x W x 2), Egomotion(6-dim vector, se(3)), Inverse depth map of the first/second frame
           Camera Distance(in meter), Overlap ratio([0, 1)), Photo-cosistency Error (in RMS)
"""
def ypr2quat(ypr):
    print(ypr)
    quat = np.zeros(4)
    ypr_half_cos = np.cos(ypr/2)
    ypr_half_sin = np.sin(ypr/2)

    print(ypr_half_cos)
    quat[0] = ypr_half_cos[2]*ypr_half_cos[1]*ypr_half_cos[0] + ypr_half_sin[2]*ypr_half_sin[1]*ypr_half_sin[0]
    quat[1] = ypr_half_sin[2]*ypr_half_cos[1]*ypr_half_cos[0] + ypr_half_cos[2]*ypr_half_sin[1]*ypr_half_sin[0]
    quat[2] = ypr_half_cos[2]*ypr_half_sin[1]*ypr_half_cos[0] + ypr_half_sin[2]*ypr_half_cos[1]*ypr_half_sin[0]
    quat[3] = ypr_half_cos[2]*ypr_half_cos[1]*ypr_half_sin[0] + ypr_half_sin[2]*ypr_half_sin[1]*ypr_half_cos[0]

    return quat

def dataLoad_SUN3D(sequenceName, nFrame1, nFrame2,dataTopPath,verbose=False):
    
   # print(sequenceName, nFrame1, nFrame2,dataTopPath, matlabCodePath,verbose)
    # call matlab function (Input in single table, Output in single table "result")
    #SUN3D_output = mlab.run_func(matlabCodePath,{'sequenceDir':dataTopPath,'sequenceName':sequenceName,'frameID':str(nFrame1),'frameIDAnother':str(nFrame2)})
    
    output = SUN3Dflow_py(dataTopPath,sequenceName,nFrame1,nFrame2)
    #output = SUN3Dflow_py(dataTopPath,sequenceName,1,10)
  
    # Convert array type output to numpy.array type output
    # Somthing problem for load nparray by frombuffer & order = 'C' : Cannot convert to 3-dim Tensor
    # egomotion( 1:3 : translation, 4:6 : rotation )
   # output['image_first'] =np.array(output['image_first'].reshape(img_height, img_width, 3))
   # output['image_second'] = np.array(output['image_second'].reshape(img_height,img_width,3))
   # output['opticalFlow'] = np.array(output['opticalFlow'].reshape(img_height,img_width,2))
   # output['egomotion'] = np.array(output['egomotion']).reshape(6)
   # output['depth_first'] = np.array(output['depth_first'].reshape(img_height,img_width,1))
   # output['depth_second'] = np.array(output['depth_second'].reshape(img_height,img_width,1))

    # check whether the function provides wrong output
    if verbose:
        scipy.misc.imsave('image_first.png',output['image_first'] )
        scipy.misc.imsave('image_second.png',output['image_second'] )
        scipy.misc.imsave('opticalFlow_u.png',output['opticalFlow'][:,:,0] )
        scipy.misc.imsave('opticalFlow_v.png',output['opticalFlow'][:,:,1] )
        scipy.misc.imsave('depth_first.png',output['depth_first'] )
        #scipy.misc.imsave('depth_second.png',output['depth_second'] )

        print ('egomotion :' ,  output['egomotion'])
        print ('Camera distance :',  output['camDist'])
        print ('Overlap Ratio :', output['overlapRatio'])
        print ('Photo-consistency Error :',output['pce'] )

    return output


#output = dataLoad_SUN3D('harvard_c11/hv_c11_2',1,2,'../','./SUN3Dflow_py.m',True)
#output = dataLoad_SUN3D('harvard_c11/hv_c11_2',1,2,'../','./SUN3Dflow_py.m',True)

"""
 Function to load SUN3D Data path list
 Inputs : SUN3D dataset path
 Outputs : list of all sequence name and absolute directory
"""
def listOfDir_generate(dataPathTop):

    list_dir_top = listdir(dataPathTop)
    list_dir_total = []

    for dir_top in list_dir_top :
        dataPathBot = dataPathTop+'/'+dir_top
        list_dir_bot = listdir(dataPathBot)
        for dir_bot in list_dir_bot :
            list_dir_total.append(dataPathBot+'/'+dir_bot+'/')

    return list_dir_total

def readSUN3D_singleData(list_dir,threshold_PCE,threshold_camDist,threshold_overlap,threshold_iter,choose_secondFrame,output):
    
    dataDir = list_dir[random.randint(0,len(list_dir)-1)]
    #print(dataDir)
    
    minFrame = 1
    maxFrame = len(glob(dataDir+'image/*'))
    frameLength = maxFrame

    #print (minFrame,maxFrame)
    frameID = random.randint(minFrame,maxFrame)
    frameIDAnother = -1
    count = 1 
    while(True):
        
        if (frameID==frameIDAnother):
            frameID = random.randint(1,frameLength)
        else:
            while(True):
                frameIDAnother = random.randint(max(minFrame,frameID-choose_secondFrame),min(maxFrame, frameID+choose_secondFrame))    
                if (frameID != frameIDAnother):
                    break

            #frameID = 10
            #frameIDAnother = 20
            output_tmp = dataLoad_SUN3D(dataDir,frameID,frameIDAnother,'',False)
            #output_tmp = dataLoad_SUN3D(dataDir,642,646,'',False)
           # print(output_tmp['camDist'], output_tmp['pce'] ,output_tmp['overlapRatio'])
            if (output_tmp['camDist'] > threshold_camDist and output_tmp['pce'] < threshold_PCE and output_tmp['overlapRatio']> threshold_overlap and output_tmp['overlapRatio']<1):
                
                egomotion_extended = np.zeros(7)

                egomotion_tmp = output_tmp['egomotion']        
                
                scale_tmp = np.linalg.norm(egomotion_tmp[0:3])   
                
                """
                quat_tmp = ypr2quat(egomotion_tmp[3:6])
                egomotion_extended[0:3] = egomotion_tmp[0:3]/scale_tmp
                
                egomotion_extended[3:7] = quat_tmp
                egomotion_extended[7] = scale_tmp
                """
                
                egomotion_tmp[0:3] = egomotion_tmp[0:3]/scale_tmp
                egomotion_extended[0:6] = egomotion_tmp
                egomotion_extended[6] = scale_tmp
                output_tmp['egomotion'] = egomotion_extended
                
                #print('finished')
                
                output.put(output_tmp)
                
                break
   
                """
                scale_tmp = np.linalg.norm(egomotion_tmp[0:3])
                target_egomotion[iterBatch,0:3] = egomotion_tmp[0:3]/scale_tmp
                target_egomotion[iterBatch,3:3] = egomotion_tmp[3:3]
                target_egomotion[iterBatch,6] = scale_tmp
                """
        
            else:
                #print(count)
                count = count + 1
                if (output_tmp['overlapRatio']<= threshold_overlap):
                    minFrame = min(frameID,frameIDAnother)
                    maxFrame = max(frameID,frameIDAnother)
                
                if (count == threshold_iter):
                    count = 1
                    frameIDAnother = frameID
                    minFrame = 1
                    maxFrame = frameLength
                    
                    #print('Check the sequence :'+dataDir)
        

def miniBatch_generate(list_dir, miniBatch_size=8):
    # This is hyper-parameter for ( image width and height, second frame number bias,
    #  image pair's thresholds)
    width = 256
    height = 192
    target_factor = 1
    width_resized = int(width/target_factor)
    height_resized = int(height/target_factor)
    choose_secondFrame = 15

    threshold_PCE = 0.1
    threshold_camDist = 0.02
    threshold_overlap = 0.5
    threshold_iter =5

    # define pyTorch tensors of minibatch input/target
    # egomotion( 1:3 : translation, 4:6 : rotation, 7 : scale )
    input_image_first = torch.Tensor(miniBatch_size,height,width,3 ).numpy()
    input_image_second = torch.Tensor(miniBatch_size,height,width,3 ).numpy()
    target_depth_first = torch.Tensor(miniBatch_size,height_resized,width_resized,1 ).numpy()
    #target_depth_second = torch.Tensor(miniBatch_size,height_resized,width_resized,1 ).numpy()
    target_opticalFlow = torch.Tensor(miniBatch_size,height_resized,width_resized,2 ).numpy()
    target_egomotion = torch.Tensor(miniBatch_size,7).numpy()

    

    # parallel loop for iteration of minibatch
    numProcess =miniBatch_size
    
    # Process allocate
    # .append and pre-allocation are same??
     
    for iterProcess in range(int(miniBatch_size/numProcess)):
        processBias = numProcess*(iterProcess-1)

        procs = []#miniBatch_size*[None]
        output_multiProcess = numProcess*[None]
        for iterBatch in range(numProcess):
            output_multiProcess[iterBatch] = Queue()

        for iterBatch in range(numProcess):
            procs.append(Process(target=readSUN3D_singleData, args=(list_dir,threshold_PCE,threshold_camDist,threshold_overlap,threshold_iter,choose_secondFrame,output_multiProcess[iterBatch])))

        # Process start
        for p in procs:
            p.start()

        # Save output to local variable
        for iterBatch in range(numProcess):
            output_tmp = output_multiProcess[iterBatch].get()
            output_multiProcess[iterBatch].close()

            input_image_first[iterBatch+processBias,:,:,:] = output_tmp['image_first']
            input_image_second[iterBatch+processBias,:,:,:] = output_tmp['image_second']
            target_depth_first[iterBatch+processBias,:,:,:] = output_tmp['depth_first']
            #target_depth_second[iterBatch+processBias,:,:,:] = output_tmp['depth_second']
            target_opticalFlow[iterBatch+processBias,:,:,:] = output_tmp['opticalFlow']
            target_egomotion[iterBatch+processBias,:] = output_tmp['egomotion']

        for p in procs:
            p.join()

    return {'input_image_first':input_image_first,'input_image_second': input_image_second,'target_depth_first': target_depth_first,'target_opticalFlow': target_opticalFlow,'target_egomotion':target_egomotion}

"""
outDir = listOfDir_generate('/home/dongwoo/Project/dataset/SUN3D')
#print(outDir)
#a = mlab.run_func('./SUN3Dflow_py.m',{'sequenceDir':'','sequenceName':'/home/dongwoo/Project/dataset/SUN3D/hotel_sf/scan2/','frameID':str(30),'frameIDAnother':str(50)})
t = time.time()
output = miniBatch_generate(outDir,1)
elapsed = time.time() - t
print(output['target_egomotion'])
print('Success, time :',elapsed)

print('good')
"""