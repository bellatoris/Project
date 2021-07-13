from ctypes import *
from PIL import Image
from io import BytesIO
from collections import namedtuple

import numpy as np
import os

_lib_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], 'liblz4.so'))
liblz4 = CDLL(_lib_path)

# depth always stores the absolute depth values (not inverse depth)
# image is a PIL.Image with the same dimensions as depth
# depth_metric should always be 'camera_z'
# K corresponds to the width and height of image/depth (intrinsic)
# R, t is the world to camera transform (extrinsic)
View = namedtuple('View',['R','t','K','image','depth','depth_metric'])


def lz4_uncompress(input_data, expected_decompressed_size):
    """decompresses the LZ4 compressed data

    input_data: bytes
        byte string of the input data
    expected_decompressed_size: int
        size of the decompressed output data
    returns the decompressed data as bytes or None on error
    """
    assert isinstance(input_data,bytes), "input_data must be of type bytes"
    assert isinstance(expected_decompressed_size,int), "expected_decompressed_size must be of type int"

    dst_buf = create_string_buffer(expected_decompressed_size)
    status = liblz4.LZ4_decompress_safe(input_data,dst_buf,len(input_data),expected_decompressed_size)
    if status != expected_decompressed_size:
        return None
    else:
        return dst_buf.raw


def read_webp_image(h5_dataset):
    """Reads a dataset that stores an image compressed as webp

    h5_dataset : hdf5 dataset object
    Returns the image as PIL Image
    """
    data = h5_dataset[:].tobytes()
    img_bytesio = BytesIO(data)
    pil_img = Image.open(img_bytesio, 'r')
    return pil_img


def read_lz4half_depth(h5_dataset):
    """Reads a dataset that stores a depth map in lz4 compressed float16 format

    h5_dataset : hdf5 dataset object
    Returns the depth map as numpy array with float32
    """
    extents = h5_dataset.attrs['extents']
    num_pixel = extents[0]*extents[1]
    expected_size = 2*num_pixel
    data = h5_dataset[:].tobytes()
    depth_raw_data = lz4_uncompress(data, int(expected_size))
    depth = np.fromstring(depth_raw_data, dtype=np.float16)
    depth = depth.astype(np.float32)
    depth = depth.reshape((extents[0], extents[1]))
    return depth


def read_camera_params(h5_dataset):
    """Reads a dataset that stores camera params in float64

    h5_dataset : hdf5 dataset object
    Returns K,R,t as numpy array with float64
    """
    fx = h5_dataset[0]
    fy = h5_dataset[1]
    skew = h5_dataset[2]
    cx = h5_dataset[3]
    cy = h5_dataset[4]
    K = np.array([[fx, skew, cx],
                 [0, fy, cy],
                 [0, 0, 1]], dtype=np.float64)
    R = np.array([[h5_dataset[5], h5_dataset[8], h5_dataset[11]],
                  [h5_dataset[6], h5_dataset[9], h5_dataset[12]],
                  [h5_dataset[7], h5_dataset[10], h5_dataset[13]]], dtype=np.float64)
    t = np.array([h5_dataset[14], h5_dataset[15], h5_dataset[16]], dtype=np.float64)
    return K, R, t


def read_view(h5_group):
    """Reads the view group and returns it as a View tuple

    h5_group: hdf5 group
        The group for reading the view
    Returns the View tuple
    """
    img = read_webp_image(h5_group['image'])
    depth = read_lz4half_depth(h5_group['depth'])
    depth_metric = h5_group['depth'].attrs['depth_metric'].decode('ascii')
    K_arr, R_arr, t_arr = read_camera_params(h5_group['camera'])
    return View(image=img, depth=depth, depth_metric=depth_metric, K=K_arr, R=R_arr, t=t_arr)
