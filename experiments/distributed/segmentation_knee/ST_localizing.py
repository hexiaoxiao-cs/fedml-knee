import argparse
from glob import glob
import logging
import os
import random
import socket
import sys
import datetime
from multiprocessing import Pool
import numpy as np
import psutil
import setproctitle
import torch
import tqdm
import wandb
from torchinfo import summary
# add the FedML root directory to the python path
# from model.segmentation.Unet_3D import Unet_3D
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML")))

from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from FedML.fedml_api.distributed.fedseg.FedSegAPI import FedML_init, FedML_FedSeg_distributed
from FedML.fedml_api.distributed.fedseg.utils import count_parameters

#from data_preprocessing.coco.segmentation.data_loader.py import load_partition_data_distributed_coco_segmentation, load_partition_data_coco_segmentation
from data_preprocessing.pascal_voc_augmented.data_loader import load_partition_data_distributed_pascal_voc, \
    load_partition_data_pascal_voc
from data_preprocessing.cityscapes.data_loader import load_partition_data_distributed_cityscapes, \
    load_partition_data_cityscapes
from data_preprocessing.sensetime.data_loader import load_partition_data_sensetime, load_partition_data_sensetime_semi
from model.segmentation.deeplabV3_plus import DeepLabV3_plus
from model.segmentation.unet import UNet
from model.segmentation.Vnet_3D import VNet
from data_preprocessing.sensetime.utils import *
#create data_loader for validation data
import os, sys, time, copy, shutil

# plasma
# import sitktools

# numpy, scipy, scikit-learn
import numpy as np
from numpy import random
# import cPickle as pickle
import pickle
import gzip
import SimpleITK as sitk
from PIL import Image
# scipy
import scipy
# from scipy.misc import imsave, imread
import scipy.io as sio
import scipy.ndimage as ndimage
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
# skimage
from skimage.morphology import label

# dice: single label
def dice(seg, gt, val_lb=1):
    """
    ## init
    """
    if seg.shape != gt.shape:
        raise ValueError("Shape mismatch: seg and gt must have the same shape.")
    #
    if val_lb < 0:
        seg = seg > 0
        gt = gt > 0
    else:
        seg = seg == val_lb
        gt = gt == val_lb
    # Compute Dice coefficient
    intersection = np.logical_and(seg, gt)
    return 2. * intersection.sum() / (seg.sum() + gt.sum())


# voe: single label
def voe(seg, gt, val_lb=1):
    """
    ## init
    """
    if seg.shape != gt.shape:
        raise ValueError("Shape mismatch: seg and gt must have the same shape.")
    #
    if val_lb < 0:
        seg = seg > 0
        gt = gt > 0
    else:
        seg = seg == val_lb
        gt = gt == val_lb
    # Compute voe coefficient
    intersection = np.logical_and(seg, gt)
    union = np.logical_or(seg, gt)
    return 100.0 * (1.0 - np.float32(intersection.sum()) / np.float32(union.sum()))


# vd: single label
def vd(seg, gt, val_lb=1):
    """
    ## init
    """
    if seg.shape != gt.shape:
        raise ValueError("Shape mismatch: seg and gt must have the same shape.")
    #
    if val_lb < 0:
        seg = seg > 0
        gt = gt > 0
    else:
        seg = seg == val_lb
        gt = gt == val_lb
    # Compute vd coefficient
    gt = np.int8(gt)
    wori = np.int8(seg - gt)
    return 100.0 * (wori.sum() / gt.sum())


"""
## medpy for dists
"""


## basic: surface errors/distances
def surface_distances(result, reference, voxelspacing=None, connectivity=1, iterations=1, ret_all=False):
    """
    # The distances between the surface voxel of binary objects in result and their
    # nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_3d(result.astype(np.bool))
    reference = np.atleast_3d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')
    # extract only 1-pixel border line of objects
    result_border = np.logical_xor(result, binary_erosion(result, structure=footprint, iterations=iterations))
    reference_border = np.logical_xor(reference, binary_erosion(reference, structure=footprint, iterations=iterations))
    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    ##
    if ret_all:
        return sds, dt, result_border, reference_border
    else:
        return sds


## ausdorff Distance.
def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    ## Hausdorff Distance.
    # Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    # images. It is defined as the maximum surface distance between the objects.
    ## Parameters
    # ----------
    # result : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # reference : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # voxelspacing : float or sequence of floats, optional
    #     The voxelspacing in a distance unit i.e. spacing of elements
    #     along each dimension. If a sequence, must be of length equal to
    #     the input rank; if a single number, this is used for all axes. If
    #     not specified, a grid spacing of unity is implied.
    # connectivity : int
    #     The neighbourhood/connectivity considered when determining the surface
    #     of the binary objects. This value is passed to
    #     `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
    #     Note that the connectivity influences the result in the case of the Hausdorff distance.
    # Returns
    # -------
    # hd : float
    #     The symmetric Hausdorff Distance between the object(s) in ```result``` and the
    #     object(s) in ```reference```. The distance unit is the same as for the spacing of
    #     elements along each dimension, which is usually given in mm.
    #
    # See also
    # --------
    # :func:`assd`
    # :func:`asd`
    # Notes
    # -----
    # This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


## Average surface distance metric.
def asd(result, reference, voxelspacing=None, connectivity=1):
    """
    # Average surface distance metric.
    # Computes the average surface distance (ASD) between the binary objects in two images.
    # Parameters
    # ----------
    # result : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # reference : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # voxelspacing : float or sequence of floats, optional
    #     The voxelspacing in a distance unit i.e. spacing of elements
    #     along each dimension. If a sequence, must be of length equal to
    #     the input rank; if a single number, this is used for all axes. If
    #     not specified, a grid spacing of unity is implied.
    # connectivity : int
    #     The neighbourhood/connectivity considered when determining the surface
    #     of the binary objects. This value is passed to
    #     `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
    #     The decision on the connectivity is important, as it can influence the results
    #     strongly. If in doubt, leave it as it is.
    # Returns
    # -------
    # asd : float
    #     The average surface distance between the object(s) in ``result`` and the
    #     object(s) in ``reference``. The distance unit is the same as for the spacing
    #     of elements along each dimension, which is usually given in mm.
    # See also
    # --------
    # :func:`assd`
    # :func:`hd`
    # Notes
    # -----
    # This is not a real metric, as it is directed. See `assd` for a real metric of this.
    # The method is implemented making use of distance images and simple binary morphology
    # to achieve high computational speed.
    # Examples
    # --------
    # The `connectivity` determines what pixels/voxels are considered the surface of a
    # binary object. Take the following binary image showing a cross
    #
    # from scipy.ndimage.morphology import generate_binary_structure
    # cross = generate_binary_structure(2, 1)
    # array([[0, 1, 0],
    #        [1, 1, 1],
    #        [0, 1, 0]])
    # With `connectivity` set to `1` a 4-neighbourhood is considered when determining the
    # object surface, resulting in the surface
    # .. code-block:: python
    #
    #     array([[0, 1, 0],
    #            [1, 0, 1],
    #            [0, 1, 0]])
    # Changing `connectivity` to `2`, a 8-neighbourhood is considered and we get:
    # .. code-block:: python
    #
    #     array([[0, 1, 0],
    #            [1, 1, 1],
    #            [0, 1, 0]])
    #
    # , as a diagonal connection does no longer qualifies as valid object surface.
    #
    # This influences the  results `asd` returns. Imagine we want to compute the surface
    # distance of our cross to a cube-like object:
    #
    # cube = generate_binary_structure(2, 1)
    # array([[1, 1, 1],
    #        [1, 1, 1],
    #        [1, 1, 1]])
    #
    # , which surface is, independent of the `connectivity` value set, always
    #
    # .. code-block:: python
    #
    #     array([[1, 1, 1],
    #            [1, 0, 1],
    #            [1, 1, 1]])
    #
    # Using a `connectivity` of `1` we get
    #
    # asd(cross, cube, connectivity=1)
    # 0.0
    #
    # while a value of `2` returns us
    #
    # asd(cross, cube, connectivity=2)
    # 0.20000000000000001
    #
    # due to the center of the cross being considered surface as well.
    """
    sds = surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd


## Average symmetric surface distance.
def assd(result, reference, voxelspacing=None, connectivity=1):
    """
    # Average symmetric surface distance.
    #
    # Computes the average symmetric surface distance (ASD) between the binary objects in
    # two images.
    #
    # Parameters
    # ----------
    # result : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # reference : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # voxelspacing : float or sequence of floats, optional
    #     The voxelspacing in a distance unit i.e. spacing of elements
    #     along each dimension. If a sequence, must be of length equal to
    #     the input rank; if a single number, this is used for all axes. If
    #     not specified, a grid spacing of unity is implied.
    # connectivity : int
    #     The neighbourhood/connectivity considered when determining the surface
    #     of the binary objects. This value is passed to
    #     `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
    #     The decision on the connectivity is important, as it can influence the results
    #     strongly. If in doubt, leave it as it is.
    #
    # Returns
    # -------
    # assd : float
    #     The average symmetric surface distance between the object(s) in ``result`` and the
    #     object(s) in ``reference``. The distance unit is the same as for the spacing of
    #     elements along each dimension, which is usually given in mm.
    #
    # See also
    # --------
    # :func:`asd`
    # :func:`hd`
    #
    # Notes
    # -----
    # This is a real metric, obtained by calling and averaging
    #
    # >>> asd(result, reference)
    #
    # and
    #
    # >>> asd(reference, result)
    #
    # The binary images can therefore be supplied in any order.
    """
    assd = np.mean(
        (asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)))
    return assd


"""
## connected comp
"""


## max connected
def max_connected_comp(tmp_buff, lb_num=-1, neighbors=4):
    if lb_num == -1:
        binary_buff = np.uint8(tmp_buff > 0)
    else:
        binary_buff = np.uint8(tmp_buff == lb_num)
    # connected_comp
    connected_group, connected_num = label(binary_buff, neighbors=neighbors, return_num=True)
    comp_sum = []
    for i in range(connected_num + 1):
        if i == 0:
            comp_sum.insert(i, 0)
            continue
        comp_sum.insert(i, np.sum(connected_group == i))
    max_comp_ind = np.argmax(comp_sum)
    if lb_num == -1:
        max_comp = np.uint8(connected_group == max_comp_ind)
    else:
        max_comp = np.uint8(connected_group == max_comp_ind) * np.uint8(lb_num)
    #
    return max_comp

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--process_name', type=str, default='FedSeg-Knee-distributed:',
                        help='Machine process names')

    parser.add_argument('--model', type=str, default='semi', metavar='N', required=True,
                        choices=['Vnet_3D', "bone_semi", "bone_semi_transform", "fine", "bd", "semi"],
                        help='neural network used in training')

    # parser.add_argument('--backbone', type=str, default='resnet',
    #                     help='employ with backbone (default: xception)')
    #
    # parser.add_argument('--backbone_pretrained', type=str2bool, nargs='?', const=True, default=True,
    #                     help='pretrained backbone (default: True)')
    #
    # parser.add_argument('--backbone_freezed', type=str2bool, nargs='?', const=True, default=False,
    #                     help='Freeze backbone to extract features only once (default: False)')

    # parser.add_argument('--extract_feat', type=str2bool, nargs='?', const=True, default=False,
    #                     help='Extract Feature Maps of (default: False) NOTE: --backbone_freezed has to be True for this argument to be considered')

    # parser.add_argument('--outstride', type=int, default=16,
    #                     help='network output stride (default: 16)')

    parser.add_argument('--dataset', type=str, default='sensetime', metavar='N',
                        choices=['sensetime', 'sensetime_semi', 'OAI'],
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='../../../data/sensetime',
                        help='data directory (default = ../../../data/sensetime)')

    parser.add_argument('--checkname', type=str, default='Vnet3D-sensetime-hetero', help='set the checkpoint name')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=1, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--batch_size_lb', type=int, default=1, metavar='N',
                        help='input batch size of labeled data for training (default 1)')

    parser.add_argument('--client_num_per_round', type=int, default=1, metavar='NN',
                        help='number of workers')

    parser.add_argument('--save_client_model', type=str2bool, nargs='?', const=True, default=True,
                        help='whether to save locally trained model by clients (default: False')

    parser.add_argument('--save_model', type=str2bool, nargs='?', const=True, default=True,
                        help='whether to save best averaged model (default: False')

    parser.add_argument('--load_model', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to load pre-trained model weights (default: False')

    parser.add_argument('--model_path', type=str, default=None,
                        help='Pre-trained saved model path  NOTE: --load has to be True for this argument to be considered')

    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 4)')

    parser.add_argument('--sync_bn', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to use sync bn (default: False)')

    parser.add_argument('--freeze_bn', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to freeze bn parameters (default: False)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')

    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--evaluation_frequency', type=int, default=5,
                        help='Frequency of model evaluation on training dataset (Default: every 5th round)')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=2,
                        help='gpu_num_per_server')

    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str, default="mapping_config",
                        help='the key in gpu utilization file')

    parser.add_argument('--image_size', type=int, default=32,
                        help='Specify the size of the image (along sagittal direction)')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    parser.add_argument('--net_channel', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()

    return args

img_sitk_l=[]
lbl_sitk_l=[]
# 0.Read image files
path_case = '../../../data/OAI/test_changed.txt'
base_path= '../../../data/OAI'
file_path=[]
image_path=[]
label_path=[]
sagittal_num=32
image=[]
label=[]
path=[]
idd=[]
net_channel=8
# spacing_c_a_s = [0.3125, 0.3125, 3.3]
Model_path="/research/cbim/vast/xh172/FedCV/experiments/distributed/segmentation_knee/run/sensetime_oai_semi/semi_OAI/ST_full_reso/model_best.pth.tar"
def bbox2_3d(img):
    r=np.any(img,axis=(1,2))
    c=np.any(img,axis=(0,2))
    z=np.any(img,axis=(0,1))
    rmin,rmax=np.where(r)[0][[0,-1]]
    cmin,cmax=np.where(c)[0][[0,-1]]
    zmin,zmax=np.where(z)[0][[0,-1]]
    return rmin,rmax,cmin,cmax,zmin,zmax

# with open(path_case, 'r') as f:
#     for i in f.readlines():
#         path_i = i.split("\n")[0]
#         file_path.append(os.path.join(base_path, path_i))
# for i in file_path:
#     if "bl" in i:
#         mode = 0
#     else:
#         mode = 1
#     name = os.path.basename(i)
#     names = name.split("_")
#     subject_name = names[0] + "_" + names[1]
#     if mode == 0:
#         path_i = os.path.join(base_path, "img_with_mask", "bl_sag_3d_dess_mhd", subject_name + ".mha")
#     else:
#         path_i = os.path.join(base_path, "img_with_mask", "12m_sag_3d_dess_mhd", subject_name + ".mhd")
#     id = subject_name
#     img_sitk = sitk.ReadImage(path_i)
#     img_sitk_l.append(img_sitk)
#     label_sitk = sitk.ReadImage(i)
#     lbl_sitk_l.append(label_sitk)
#     resampled_img_itk, resampled_label_itk = itk_resample(img_sitk, label_sitk, out_spacing=spacing_c_a_s,
#                                                           interpolation=sitk.sitkLinear,
#                                                           dtype=sitk.sitkFloat32)
#     resample_img = sitk.GetArrayFromImage(resampled_img_itk)
#     resampled_label = sitk.GetArrayFromImage(resampled_label_itk)

#     #### pad or crop to fixed size [30, 512, 512]
#     # self.slices_list.append(self.slices_list[-1]+resample_img.shape[0])
#     sagital_num_slices = sagittal_num  # resample_img.shape[0]  ###################### notice
#     # if isSWIN==True: sagital_num_slices = 32
#     fixed_size_img = pad_crop(resample_img, (sagital_num_slices, 512, 512), mode='constant', value=0)
#     fixed_size_label = pad_crop(resampled_label, (sagital_num_slices, 512, 512), mode='constant', value=0)
#     #### move the menis labels(3,4) to (7,8), not necessary
#     # tongyi label value
#     # fixed_size_label[fixed_size_label == 3] = 9
#     # fixed_size_label[fixed_size_label == 4] = 10
#     # fixed_size_label[fixed_size_label == 5] = 3
#     # fixed_size_label[fixed_size_label == 6] = 4
#     # fixed_size_label[fixed_size_label == 7] = 5
#     # fixed_size_label[fixed_size_label == 8] = 6
#     # ## used for menius
#     # fixed_size_label[fixed_size_label == 9] = 7
#     # fixed_size_label[fixed_size_label == 10] = 8

#     #### z-score norm
#     fixed_size_img = (fixed_size_img - fixed_size_img.mean()) / fixed_size_img.std()

#     # if you need use bd-loss
#     # if self.bd:
#     #     #### calculate dist
#     #     label_onehot = np.stack([fixed_size_label == c for c in range(9)], axis=0).astype(np.int32)
#     #     label_dist = one_hot2dist(label_onehot).astype(np.float32)
#     #     self.dist.append(label_dist)

#     #### append arrays and ids to list for dataloader
#     # print(fixed_size_img.shape)
#     # print(fixed_size_label.shape)
#     image.append(fixed_size_img)
#     label.append(fixed_size_label.astype(np.uint8))
#     path.append(path_i)
#     idd.append(id)
#     #print(path_i)
# input_file="../../../data/OAI/0_list_unlabeled.pkl"
image=None
device=0
# with open(input_file,"rb") as f:
#     image=pickle.load(f)
# list_bl="../../../data/OAI/bl_sag_3d_dess_with_did.txt"
# list_12m="../../../data/OAI/12m_sag_3d_dess_with_did.txt"
root_dir="../../../data/sensetime/"
# unlabel={0:[],1:[],2:[],3:[],4:[],5:[]}
# with open(list_bl,"r") as f:
#     for line in f.readlines():
#         terms=line.split()
#         # if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
#         #     continue
#         # else:
#         unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "bl", terms[0] + "_" + terms[1] + "_" + terms[2] + ".mha")))
# with open(list_12m,"r") as f:
#     for line in f.readlines():
#         terms=line.split()
#         # if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
#         #     continue
#         # else:
#         unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "12m", terms[0] + "_" + terms[1] + "_" + terms[2] + ".mha")))
unlabel=[]
labeled=[]
for i in glob(os.path.join(root_dir,"ST_full_reso_nolbl","*.nii.gz")):
    unlabel.append((os.path.basename(i),i))
for i in glob(os.path.join(root_dir,"ST_full_reso","*","org.mha")):
    labeled.append((i.split("/")[-2],i))
    print(i.split("/")[-2])
# 1. pre-processing
# resample to fixed spacing

# 2. Vnet prediction
# input_ = torch.from_numpy(fixed_size_img[np.newaxis, :][np.newaxis, :])

model = VNet(model_channel=2,
                     out_class=4)
model.load_state_dict(torch.load(Model_path)['state_dict'])
model.to(device)
model.eval()
out=pd.DataFrame()
save_path="unlabeled_cropped"
# save_path = 'OAI_pred_corrected_cross_existing_2/'

for (name,file_place) in tqdm.tqdm(unlabel):
    # if name!="00072.nii.gz":
    #     continue
    img=sitk.ReadImage(file_place)
    x=sitk.GetArrayFromImage(img)
    # if "RIGHT" in file_place:
    #     x=np.flip(x,0)
    x = pad_crop(x, (128,400,400), mode='constant', value=0)
    x = (x - x.mean()) / x.std()
    # print(x)
    x=torch.from_numpy(x[np.newaxis, :][np.newaxis, :])
    pred=model(x.float().to(device))
    pred=torch.argmax(torch.nn.functional.softmax(pred,dim=1),dim=1)
    pred = pred[0].cpu().numpy()
    # 3. post-processing
    region_num = [ 1,2, 1]
    try:
        pred = multi_find_largest_k_region(pred.astype(np.uint8), region_num)
    except AssertionError:
        out=out.append([[name,"failed"]])
        continue
    #test output
    # lbl=sitk.GetImageFromArray(pred)
    # sitk.WriteImage(lbl,"00072_lbl.nii.gz")
    # break

    try:
        rmin1,rmax1,cmin1,cmax1,zmin1,zmax1=bbox2_3d(pred==1)
        rmin2,rmax2,cmin2,cmax2,zmin2,zmax2=bbox2_3d(pred==2)
        rmin3,rmax3,cmin3,cmax3,zmin3,zmax3=bbox2_3d(pred==3)
        rmin,rmax,cmin,cmax,zmin,zmax=bbox2_3d(pred)
        out=out.append([[name,"0",rmin,cmin,zmin,rmax,cmax,zmax]])#all
        out=out.append([[name,"1",rmin1,cmin1,zmin1,rmax1,cmax1,zmax1]])#all
        out=out.append([[name,"2",rmin2,cmin2,zmin2,rmax2,cmax2,zmax2]])#all
        out=out.append([[name,"3",rmin3,cmin3,zmin3,rmax3,cmax3,zmax3]])#all
    except:
        out=out.append([[name,"failed"]])
        continue
    # print([name,"0",rmin,cmin,zmin,rmax,cmax,zmax])
    #saving cropped
    # to_save=sitk.GetArrayFromImage(img)
    # to_save=to_save[rmin:rmax,cmin:cmax,zmin:zmax]
    # save_itk=sitk.GetImageFromArray(to_save)
out.to_csv("./bbox_ST_lbl.csv")
    # print(rmin,rmax,cmin,cmax,zmin,zmax)
    # print(bbox2_3d(y))
    # # Femur Bone: 1 Femur Cartilage: 2
    # # Tibia Bone: 5 Tibia Cartilage: 6
    # # Patella Bone: 7 Patella Cartilage: 8
    # # meniscus: 3 4
    # # pred[pred == 8] = 10
    # # pred[pred == 7] = 9
    # # pred[pred == 6] = 8
    # # pred[pred == 5] = 7
    # # pred[pred == 4] = 6
    # # pred[pred == 3] = 5
    # # pred[pred == 10] = 4
    # # pred[pred == 9] = 3
    # pic_res = list()
    # truth = label[i]
    # # prediction_truth = prediction > 0
    # # prediction_1_truth = prediction_1 == 1
    # # prediction_2_truth = prediction_2 == 2
    # # truth_1 = truth == 1
    # # truth_2 = truth == 2
    # # truth_truth = truth > 0
    # voxelspacing = spacing_c_a_s.reverse()
    # line = list()
    # line.append(dice(pred, truth, -1))
    # line.append(voe(pred, truth, -1))
    # line.append(vd(pred, truth, -1))
    # line.append(asd(pred, truth, voxelspacing=voxelspacing))
    # line.append(hd(pred, truth, voxelspacing=voxelspacing))
    # line.append(assd(pred, truth, voxelspacing=voxelspacing))
    # print(line)
    # pic_res.append(line)
    # for j in range(3):
    #     line = list()
    #     # pred_temp=pred== j+1
    #     # truth_temp=truth==j+1
    #     line.append(dice(pred, truth, j + 1))
    #     line.append(voe(pred, truth, j + 1))
    #     line.append(vd(pred, truth, j + 1))
    #     pred_temp = pred == j + 1
    #     truth_temp = truth == j + 1
    #     line.append(asd(pred_temp, truth_temp, voxelspacing=voxelspacing))
    #     line.append(hd(pred_temp, truth_temp, voxelspacing=voxelspacing))
    #     line.append(assd(pred_temp, truth_temp, voxelspacing=voxelspacing))
    #     pic_res.append(line)
    #     print(line)

    # scores = scores.append(pic_res)
    # 4. save to files
'''
    
    pred_itk = sitk.GetImageFromArray(pred)
    pred_itk.SetOrigin(img_sitk_l[i].GetOrigin())
    pred_itk.SetSpacing(spacing_c_a_s)
    pred_itk.SetDirection(img_sitk_l[i].GetDirection())
    # resample pred to origin spacing
    resampled_label_itk = itk_resample_only_label(pred_itk, out_spacing=img_sitk_l[i].GetSpacing())

    pred = sitk.GetArrayFromImage(resampled_label_itk)

    # pad or crop to origin size
    pred = pad_crop(pred, img_sitk_l[i].GetSize()[::-1], mode='constant',
                    value=0)
    pred_itk = sitk.GetImageFromArray(pred)
    pred_itk.SetOrigin(img_sitk_l[i].GetOrigin())
    pred_itk.SetSpacing(spacing_c_a_s)
    pred_itk.SetDirection(img_sitk_l[i].GetDirection())

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sitk.WriteImage(pred_itk, save_path + idd[i] + '_pred.mha')
    sitk.WriteImage(img_sitk_l[i],save_path+idd[i]+"_img.mha")
    sitk.WriteImage(lbl_sitk_l[i],save_path+idd[i]+"_lbl.mha")

scores.to_csv(os.path.join(save_path,"summary.csv"))
'''