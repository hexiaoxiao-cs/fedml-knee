import os
import numpy as np
import SimpleITK as sitk
import time
import operator
import torch
import torch.nn as nn
from skimage.measure import label
import itertools
from torch.utils.data.sampler import Sampler
import random
from torch.nn import functional as F
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
import copy
import matplotlib.pyplot as plt

timer_verbose = False
import cv2
import math
from scipy.ndimage import distance_transform_edt as distance
import numba
import logging

def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


class timer(object):
    def __init__(self, verbose=1):
        self.verbose = verbose

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            if self.verbose:
                print('Function "%s" costs %.2f s' % (fn.__name__, time.time() - start))
            return result

        return wrapper


def seed_torch(seed=10):
    #     random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dice_score_single(m1, m2, smooth=1):  # , is_average=True

    num = m1.size(0)
    m1 = m1.view(num, -1)
    m2 = m2.view(num, -1)
    intersection = (m1 * m2)
    ss = (2. * intersection.sum() + smooth) / (m1.sum() + m2.sum() + smooth)
    return ss.data.cpu().numpy()


@timer(timer_verbose)
def dice_score(m1, m2, label_num=7, smooth=1.):  # , is_average=True
    '''

    :param m1:  shape [N,L,W,H]
    :param m2: shape [N,L,W,H]
    :param smooth:
    :return:
    '''
    scores = []
    # print('channel_num',label_num)
    for i in range(1, label_num):
        m1_ = m1.clone()
        m2_ = m2.clone()
        m1_ = m1_ == i
        m2_ = m2_ == i

        num = m1.size(0)
        m1_ = m1_.view(num, -1)
        m2_ = m2_.view(num, -1)
        intersection = (m1_ * m2_)
        ss = (2. * intersection.sum(1) + smooth) / (m1_.sum(1) + m2_.sum(1) + smooth)
        scores.append(ss.mean().data.cpu().numpy())
    return scores


@timer(timer_verbose)
def voe_score(m1, m2, label_num=7, smooth=1.):
    scores = []
    # channel_num = m1.size(1)
    for i in range(1, label_num):
        m1_ = m1.clone()
        m2_ = m2.clone()
        m1_ = m1_ == i
        m2_ = m2_ == i

        num = m1.size(0)
        m1_ = m1_.view(num, -1)
        m2_ = m2_.view(num, -1)
        intersection = (m1_ * m2_)
        ss = 100. * (1. - (intersection.sum(1) + smooth) / (m1_.sum(1) + m2_.sum(1) - intersection.sum(1) + smooth))
        scores.append(ss.mean().data.cpu().numpy())
    return scores


@timer(timer_verbose)
def vd_score(m1, m2, label_num=7, smooth=1.):
    scores = []
    # channel_num = m1.size(1)
    for i in range(1, label_num):
        m1_ = m1.clone()
        m2_ = m2.clone()
        m1_ = m1_ == i
        m2_ = m2_ == i

        num = m1.size(0)
        m1_ = m1_.view(num, -1)
        m2_ = m2_.view(num, -1)
        ss = 100. * ((m1_.sum(1) - m2_.sum(1) + smooth) / (m2_.sum(1) + smooth))
        scores.append(ss.mean().data.cpu().numpy())
    return scores


@timer(timer_verbose)
def hd_score(m1, m2, label_num=7, spacing=[1, 1, 1]):
    scores = []
    # channel_num = m1.size(1)
    for i in range(1, label_num):
        m1_ = copy.deepcopy(m1)
        m2_ = copy.deepcopy(m2)
        m1_ = m1_ == i
        m2_ = m2_ == i

        score_hd = hd(m1_, m2_, spacing)
        scores.append(score_hd)
    return scores


@timer(timer_verbose)
def asd_score(m1, m2, label_num=7, spacing=[1, 1, 1]):
    scores = []
    # channel_num = m1.size(1)
    for i in range(1, label_num):
        m1_ = copy.deepcopy(m1)
        m2_ = copy.deepcopy(m2)
        m1_ = m1_ == i
        m2_ = m2_ == i

        score_hd = asd(m1_, m2_, spacing)
        scores.append(score_hd)
    return scores


@timer(timer_verbose)
def assd_score(m1, m2, label_num=7, spacing=[1, 1, 1]):
    scores = []
    # channel_num = m1.size(1)
    for i in range(1, label_num):
        m1_ = copy.deepcopy(m1)
        m2_ = copy.deepcopy(m2)
        m1_ = m1_ == i
        m2_ = m2_ == i

        score_hd = assd(m1_, m2_, spacing)
        scores.append(score_hd)
    return scores


@timer(timer_verbose)
def rsd_score(m1, m2, label_num=7, spacing=[1, 1, 1]):
    scores = []
    # channel_num = m1.size(1)
    for i in range(1, label_num):
        m1_ = copy.deepcopy(m1)
        m2_ = copy.deepcopy(m2)
        m1_ = m1_ == i
        m2_ = m2_ == i

        score_hd = rsd(m1_, m2_, spacing)
        scores.append(score_hd)
    return scores


@timer(timer_verbose)
def surface_score(m1, m2, label_num=7, spacing=[1, 1, 1]):
    scores_hd = []
    scores_asd = []
    scores_rsd = []
    # channel_num = m1.size(1)
    for i in range(1, label_num):
        m1_ = copy.deepcopy(m1)
        m2_ = copy.deepcopy(m2)
        m1_ = m1_ == i
        m2_ = m2_ == i

        hd, asd, rsd = surface_3(m1_, m2_, spacing)
        scores_hd.append(hd)
        scores_asd.append(asd)
        scores_rsd.append(rsd)
    return scores_hd, scores_asd, scores_rsd


### from chaowei, only for debug
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
    # num = m1.size(0)
    # m1  = m1.view(num,-1)
    # m2  = m2.view(num,-1)
    # intersection = (m1 * m2)
    # scores = (2. * intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    # if is_average:
    #     score = scores.sum()/num
    #     return score
    # else:
    #     return -scores


# class WeightedDiceLoss():
#     def __init__(self,w_1_4_10):
#         super(WeightedDiceLoss, self).__init__()
#         self.w_1_4_10 = w_1_4_10
#     def forward(self, m1, m2,smooth=1):
#         '''
#         :param
#         m1: pred
#         :param
#         m2: label
#         :param
#         w_1_4_10:
#         :return:
#         '''
#         loss = 0.
#         channel = m1.size(1)
#         for i in range(channel):
#
#             m1_ = m1.clone()[:,i,:,:,:]
#             m2_ = m2.clone()
#             if i==0:
#                 m2_[m2_!=0]=2
#                 m2_[m2_==0]=1
#                 m2_[m2_==2]=0
#             else:
#                 m2_[m2_ != i] = 0
#                 m2_[m2_ == i] = 1
#
#             num = m1.size(0)
#             m1_ = m1_.view(num, -1)
#             m2_ = m2_.view(num, -1)
#             intersection = (m1_ * m2_)
#             ss = 1- (2. * intersection.sum()+smooth) / (m1_.sum() + m2_.sum()+smooth)
#             loss+= self.w_1_4_10[i]/self.w_1_4_10.sum()* ss
#         return loss
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
    # print('ref',reference_border.shape)
    # print('rf_bd',reference_border.shape)
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    # print('dt',dt.shape)
    # print('rs', result_border.shape)
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


## Average surface distance metric. (only oone side)
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


def asd2(result, reference, voxelspacing=None, connectivity=1):
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
    sds = surface_distances(result, reference, voxelspacing, connectivity) ** 2
    asd2 = sds.mean()
    return asd2


## Root Mean Square Distance
def rsd(result, reference, voxelspacing=None, connectivity=1):
    rsd = np.mean((asd2(result, reference, voxelspacing, connectivity),
                   asd2(reference, result, voxelspacing, connectivity))) ** 0.5
    return rsd


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


def surface_3(result, reference, voxelspacing=None, connectivity=1):
    s12 = surface_distances(result, reference, voxelspacing, connectivity)
    s21 = surface_distances(reference, result, voxelspacing, connectivity)

    hd = max(s12.max(), s21.max())
    # assd = np.mean((s12.mean(), s21.mean()))
    # rsd = np.mean(((s12**2).mean(), (s21**2).mean()))**0.5
    assd = np.mean(np.hstack((s12, s21)))
    rsd = np.mean(np.hstack((s12 ** 2, s21 ** 2))) ** 0.5

    return hd, assd, rsd


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1.

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2. * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


# def WeightedDiceLoss_roi(m1, m2, weight, smooth=1):
#
#     '''
#     :param
#     m1: pred
#     :param
#     m2: label
#     :param
#     w_1_4_10:
#     :return:
#     '''
#     loss = 0.
#     channel = m1.size(1)
#     for i in range(channel):
#
#         m1_ = m1.clone()[:, i, :, :, :]
#         m2_ = m2.clone()
#         # if i == 0:
#         #     m2_[m2_ != 0] = 2
#         #     m2_[m2_ == 0] = 1
#         #     m2_[m2_ == 2] = 0
#         # else:
#         m2_[m2_ != i+1] = 0
#         m2_[m2_ == i+1] = 1
#
#         num = m1.size(0)
#         m1_ = m1_.view(num, -1)
#         m2_ = m2_.view(num, -1)
#         intersection = (m1_ * m2_)
#
#         # print('inter',intersection.sum(1).shape)
#         # print('m1',m1_.sum(1).shape)
#         ss = 1 - (2. * intersection.sum(1) + smooth) / (m1_.sum(1) + m2_.sum(1) + smooth)
#         # print('ss',ss.shape)
#         # ss = 1- (2. * intersection.sum()+smooth) / (m1_.sum() + m2_.sum()+smooth)
#         loss += weight[i] / weight.sum() * ss.mean()
#     return loss
def WeightedDiceLoss(m1, m2, weight, smooth=1):
    '''
    :param
    m1: pred
    :param
    m2: label
    :param
    w_1_4_10:
    :return:
    '''
    loss = 0.
    channel = m1.size(1)
    for i in range(channel):

        m1_ = m1.clone()[:, i, :]  # [:,i,:,:,:]
        m2_ = m2.clone()
        if i == 0:
            m2_[m2_ != 0] = 2
            m2_[m2_ == 0] = 1
            m2_[m2_ == 2] = 0
        else:
            m2_[m2_ != i] = 0
            m2_[m2_ == i] = 1

        num = m1.size(0)
        # print('m1,m2  ',m1_.shape,m2_.shape)
        m1_ = m1_.view(num, -1)
        m2_ = m2_.view(num, -1)
        intersection = (m1_ * m2_)
        ss = 1 - (2. * intersection.sum(1) + smooth) / (m1_.sum(1) + m2_.sum(1) + smooth)
        loss += weight[i] / weight.sum() * ss.mean()
    return loss


# def WeightedDiceLoss(m1,m2,weight,smooth=1.):
#
#         '''
#         :param
#         m1: pred
#         :param
#         m2: label
#         :param
#         w_1_4_10:
#         :return:
#         '''
#         loss = 0.
#         channel = m1.size(1)
#         for i in range(channel):
#
#             m1_ = m1.clone()[:,i,:] #[:,i,:,:,:]
#             m2_ = m2.clone()
#             if i==0:
#                 m2_[m2_!=0]=2
#                 m2_[m2_==0]=1
#                 m2_[m2_==2]=0
#             else:
#                 m2_[m2_ != i] = 0
#                 m2_[m2_ == i] = 1
#
#             num = m1.size(0)
#             # print('m1,m2  ',m1_.shape,m2_.shape)
#             m1_ = m1_.view(num, -1)
#             m2_ = m2_.view(num, -1)
#             intersection = (m1_ * m2_).sum(1).float()
#             sum_ = (m1_.sum(1) + m2_.sum(1)).float()
#             ss = 1. - (2. * intersection + smooth) / (sum_ + smooth)
#
#             loss+= weight[i]/weight.sum()* ss.mean()
#             if loss<0 or loss>1:
#                 print('loss',loss,'\n','inter', intersection,'sum',sum_,sep=' ')
#                 print('m1',m1.max(),m1.min(),m1.sum())
#                 print('m2', m2.max(), m2.min(),m2.sum())
#                 mm= (m1*m2)
#                 print('mm', mm.max(), mm.min(),mm.sum())
#         return loss
# a little bit different from itk_change_spacing
def itk_resample(itk_img, label_sitk, out_spacing=[1.0, 1.0, 1.0], interpolation=sitk.sitkNearestNeighbor,
                 dtype=sitk.sitkUInt16):
    itk_size, itk_spacing, itk_origin, itk_direction = itk_img.GetSize(), itk_img.GetSpacing(), itk_img.GetOrigin(), itk_img.GetDirection()
    resample_scale = np.float32(out_spacing) / np.float32(itk_spacing)
    out_size = [int(x) for x in np.round(itk_size / resample_scale).astype(np.int32)]
    if itk_img.GetSize() == tuple(out_size):
        return itk_img, label_sitk
    t = sitk.Transform(3, sitk.sitkScale)
    t.SetParameters((1.0, 1.0, 1.0))
    resampled_img_itk = sitk.Resample(itk_img, out_size, t, interpolation, itk_origin, out_spacing, itk_direction, 0.0,
                                      dtype)
    resampled_label_itk = sitk.Resample(label_sitk, out_size, t, sitk.sitkNearestNeighbor, itk_origin, out_spacing,
                                        itk_direction, 0.0,
                                        sitk.sitkUInt8)
    return resampled_img_itk, resampled_label_itk


def itk_resample_only(itk_img, out_spacing=[1.0, 1.0, 1.0], interpolation=sitk.sitkNearestNeighbor,
                      dtype=sitk.sitkUInt16):
    itk_size, itk_spacing, itk_origin, itk_direction = itk_img.GetSize(), itk_img.GetSpacing(), itk_img.GetOrigin(), itk_img.GetDirection()
    resample_scale = np.float32(out_spacing) / np.float32(itk_spacing)
    out_size = [int(x) for x in np.round(itk_size / resample_scale).astype(np.int32)]
    if itk_img.GetSize() == tuple(out_size):
        return itk_img
    t = sitk.Transform(3, sitk.sitkScale)
    t.SetParameters((1.0, 1.0, 1.0))
    resampled_img_itk = sitk.Resample(itk_img, out_size, t, interpolation, itk_origin, out_spacing, itk_direction, 0.0,
                                      dtype)
    return resampled_img_itk


def itk_resample_only_label(label_sitk, out_spacing=[1.0, 1.0, 1.0], interpolation=sitk.sitkNearestNeighbor,
                            dtype=sitk.sitkUInt16):
    itk_size, itk_spacing, itk_origin, itk_direction = label_sitk.GetSize(), label_sitk.GetSpacing(), label_sitk.GetOrigin(), label_sitk.GetDirection()
    resample_scale = np.float32(out_spacing) / np.float32(itk_spacing)
    out_size = [int(x) for x in np.round(itk_size / resample_scale).astype(np.int32)]
    if label_sitk.GetSize() == tuple(out_size):
        return label_sitk
    t = sitk.Transform(3, sitk.sitkScale)
    t.SetParameters((1.0, 1.0, 1.0))
    resampled_img_itk = sitk.Resample(label_sitk, out_size, t, sitk.sitkNearestNeighbor, itk_origin, out_spacing,
                                      itk_direction, 0.0,
                                      sitk.sitkUInt8)
    return resampled_img_itk


def is_num(a):
    return isinstance(a, int) or isinstance(a, float)


def delta(x1, x2):
    delta_ = x2 - x1
    return delta_ // 2, delta_ - delta_ // 2


def get_padding_width(o_shape, d_shape):
    if is_num(o_shape):
        o_shape, d_shape = [o_shape], [d_shape]
    assert len(o_shape) == len(d_shape), 'Length mismatched!'
    borders = []
    for o, d in zip(o_shape, d_shape):
        borders.extend(delta(o, d))
    return borders


def get_crop_width(o_shape, d_shape):
    return get_padding_width(d_shape, o_shape)


def get_padding_shape_with_stride(o_shape, stride):
    assert isinstance(o_shape, list) or isinstance(o_shape, tuple) or isinstance(o_shape, np.ndarray)
    o_shape = np.array(o_shape)
    d_shape = np.ceil(o_shape / stride) * stride
    return d_shape.astype(np.int32)


def pad(arr, d_shape, mode='constant', value=0, strict=True):
    """
    pad numpy array, tested!
    :param arr: numpy array
    :param d_shape: array shape after padding or minimum shape
    :param mode: padding mode,
    :param value: padding value
    :param strict: if True, d_shape must be greater than arr shape and output shape is d_shape. if False, d_shape is minimum shape and output shape is np.maximum(arr.shape, d_shape)
    :return: padded arr with expected shape
    """
    assert arr.ndim == len(d_shape), 'Dimension mismatched!'
    if not strict:
        d_shape = np.maximum(arr.shape, d_shape)
    else:
        assert np.all(np.array(d_shape) >= np.array(arr.shape)), 'Padding shape must be greater than arr shape'
    borders = np.array(get_padding_width(arr.shape, d_shape))
    before = borders[list(range(0, len(borders), 2))]
    after = borders[list(range(1, len(borders), 2))]
    padding_borders = tuple(zip([int(x) for x in before], [int(x) for x in after]))
    # print(padding_borders)
    if mode == 'constant':
        return np.pad(arr, padding_borders, mode=mode, constant_values=value)
    else:
        return np.pad(arr, padding_borders, mode=mode)


def crop(arr, d_shape, strict=True):
    """
    central  crop numpy array, tested!
    :param arr: numpy array
    :param d_shape: expected shape
    :return: cropped array with expected array
    """
    assert arr.ndim == len(d_shape), 'Dimension mismatched!'
    if not strict:
        d_shape = np.minimum(arr.shape, d_shape)
    else:
        assert np.all(np.array(d_shape) <= np.array(arr.shape)), 'Crop shape must be smaller than arr shape'
    borders = np.array(get_crop_width(arr.shape, d_shape))
    start = borders[list(range(0, len(borders), 2))]
    # end = - borders[list(range(1, len(borders), 2))]
    end = map(operator.add, start, d_shape)
    slices = tuple(map(slice, start, end))
    return arr[slices]


def pad_crop(arr, d_shape, mode='constant', value=0):
    """
    pad or crop numpy array to expected shape, tested!
    :param arr: numpy array
    :param d_shape: expected shape
    :param mode: padding mode,
    :param value: padding value
    :return: padded and cropped array
    """
    assert arr.ndim == len(d_shape), 'Dimension mismatched!'
    arr = pad(arr, d_shape, mode, value, strict=False)
    return crop(arr, d_shape)


def get_roi_bbox(coarse_seg, roi_size_whole):
    # for getting ROI

    has_label = np.where(coarse_seg)
    bbox = [has_label[0].min(), has_label[0].max(), has_label[1].min(), has_label[1].max(), has_label[2].min(),
            has_label[2].max()]
    return bbox
    # centroid = [(bbox[i] + bbox[i + 1]) / 2 for i in range(0, len(bbox), 2)]
    # print('bbox_1',bbox)
    # # centroid = np.array(centroid) * (np.array(coarse_spacing) / np.array(fine_spacing))[::-1]
    # # return np.round(bbox).astype(np.int32) + 1#bbox
    # bbox = [[center - bbox_width / 2, center + bbox_width / 2] for center, bbox_width in zip(centroid, roi_size_whole)]
    # print('bbox_2', np.round(bbox).astype(np.int32) + 1 )
    # return np.round(bbox).astype(np.int32) + 1  # + 1 or not has a slight impact on the final segmentation result


def itk_right2left(itk):
    """
    flip itk from right to left, tested
    :param itk:
    :return:
    """
    resampler = sitk.FlipImageFilter()
    flipAxes = [False, False, True]
    resampler.SetFlipAxes(flipAxes)
    return resampler.Execute(itk)


def binary_find_largest_k_region(arr, top_k=1):
    """
    binary find the largest k region(s) of numpy array
    :param arr:
    :param top_k:
    :return:
    """
    arr = label(arr, connectivity=1)
    labels, counts = np.unique(arr, return_counts=True)
    counts = counts[labels > 0];
    labels = labels[labels > 0]
    top_k_label = labels[np.argsort(counts)[-top_k:]]
    return np.isin(arr, top_k_label).astype(np.uint8)


def multi_find_largest_k_region(arr, top_ks=None):
    """
    multiple class find the largest k region(s) of numpy array
    :param arr:
    :param top_ks:
    :return:
    """
    labels = np.unique(arr)
    labels = np.sort(labels[labels > 0])
    if top_ks is None:
        top_ks = [1] * len(labels)
    else:
        # if len(top_ks) != len(labels), just return the origin image
        # return arr
        assert len(top_ks) == len(labels), 'got %d labels and %d top_k(s)' % (len(labels), len(top_ks))
    multi_largest_k = np.zeros_like(arr)
    for cls, k in zip(labels, top_ks):
        cls_mask = arr == cls
        multi_largest_k += binary_find_largest_k_region(cls_mask, k) * cls
    return multi_largest_k


# ## original link: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/03_Image_Details.html
# def resamplingImage_Sp2(image_sitk, sp2=[1.0, 1.0, 1.0], sampling_way=sitk.sitkNearestNeighbor, sampling_pixel_t=sitk.sitkUInt16):
#     ## get org info
#     buff_sz1, sp1, origin = image_sitk.GetSize(), image_sitk.GetSpacing(), image_sitk.GetOrigin()
#     direction = image_sitk.GetDirection()
#     # rate
#     fScales = np.zeros(len(sp1), np.float32)
#     for i in range(len(sp1)):
#         fScales[i] = np.float32(sp2[i]) / np.float32(sp1[i])
#     # change buff size
#     buff_sz2 = list()
#     for i in range(len(buff_sz1)):
#         buff_sz2.append(int(np.round(buff_sz1[i]/fScales[i])))
#     # resampled info
#     print("Orig Size ", buff_sz1, "\nNew Size ", buff_sz2)
#     print("Orig Sp ", sp1, "\nNew Sp ", sp2)
#     print(origin)
#     ## resample
#     t = sitk.Transform(3, sitk.sitkScale)
#     t.SetParameters((1.0, 1.0, 1.0))
#     resampled_image_sitk = sitk.Resample(image_sitk, buff_sz2, t, sampling_way, origin, sp2, direction, 0.0, sampling_pixel_t)
#     print("New Image size:", resampled_image_sitk.GetSize())
#     return resampled_image_sitk
#################################################3  from semi
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        # print('two sample iter_____')
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    # print('unlabel__ iter')
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size(), "INPUT_LOGITS TARGET_LOGITS MISMATCH"
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    #logging.info("input_softmax"+str(input_softmax.shape))
    #logging.info("target_softmax"+str(target_softmax.shape))
    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    consistency = 0.1
    consistency_rampup = 40.0
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha) #None Moded
def update_ema_variables_mod(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(1-alpha).add_(param.data, alpha=alpha) #modified 
def update_ema_variables_less_freq(model, ema_model, alpha, global_step,max_epoch):
    # Use the true average until the exponential average is more correct
    alpha = min(global_step /max_epoch, alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(1-alpha).add_(param.data, alpha=alpha) #modified  #Remodified
def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


class PercentileClippingAndToFloat:
    """Change the histogram of image by doing global contrast normalization."""

    def __init__(self, cut_min=0.5, cut_max=99.5):
        """
        cut_min - lowest percentile which is used to cut the image histogram
        cut_max - highest percentile
        """
        self.cut_min = cut_min
        self.cut_max = cut_max

    def __call__(self, img, mask=None):
        img = img.astype(np.float32)
        lim_low, lim_high = np.percentile(img, [self.cut_min, self.cut_max])
        img = np.clip(img, lim_low, lim_high)

        img -= lim_low
        img /= img.max()

        img = img.astype(np.float32)
        if mask is not None:
            mask = mask.astype(np.float32)

        return img, mask


class CenterCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        c, h, w = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2

        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = np.ascontiguousarray(img[:, y1:y2, x1:x2])
        if mask is not None:
            mask = np.ascontiguousarray(mask[:, y1:y2, x1:x2])

        return img, mask


class HorizontalFlip(object):
    def __init__(self, prob=.5):
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            img = np.flip(img, axis=2)
            if mask is not None:
                mask = np.flip(mask, axis=2)
        return img, mask

    def randomize(self):
        self.state['p'] = random.random()


class GammaCorrection(object):
    def __init__(self, gamma_range=(0.5, 2), prob=0.5):
        self.gamma_range = gamma_range
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, image, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            image = image ** (1 / self.state['gamma'])
            # TODO: implement also for integers
            image = np.clip(image, 0, 1)
        return image, mask

    def randomize(self):
        self.state['p'] = random.random()
        self.state['gamma'] = random.uniform(*self.gamma_range)


class BilateralFilter(object):
    def __init__(self, d, sigma_color, sigma_space, prob=.5):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        if self.state['p'] < self.prob:
            img = np.squeeze(img)
            img = cv2.bilateralFilter(img, self.d, self.sigma_color, self.sigma_space)
            img = img[None, ...]
        return img, mask

    def randomize(self):
        self.state['p'] = random.random()


class DualCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask=None):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class OneOf(object):
    def __init__(self, transforms, prob=.5):
        self.transforms = transforms
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        if self.state['p'] < self.prob:
            img, mask = self.state['t'](img, mask)
        return img, mask

    def randomize(self):
        self.state['p'] = random.random()
        self.state['t'] = random.choice(self.transforms)
        self.state['t'].prob = 1.


class Scale(object):
    def __init__(self, ratio_range=(0.7, 1.2), prob=.5):
        self.ratio_range = ratio_range
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            ch, d0_i, d1_i = img.shape
            d0_o = math.floor(d1_i * self.state['r'])
            d0_o = d0_o + d0_o % 2
            d1_o = math.floor(d1_i * self.state['r'])
            d1_o = d1_o + d1_o % 2

            # img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit,
            #                           borderType=cv2.BORDER_REFLECT_101)
            img = np.squeeze(img)
            img = cv2.resize(img, (d1_o, d0_o), interpolation=cv2.INTER_LINEAR)
            img = img[None, ...]

            if mask is not None:
                # msk1 = cv2.copyMakeBorder(mask, limit, limit, limit, limit,
                #                           borderType=cv2.BORDER_REFLECT_101)
                tmp = np.empty((mask.shape[0], d1_o, d0_o), dtype=mask.dtype)
                for idx_ch, mask_ch in enumerate(mask):
                    tmp[idx_ch] = cv2.resize(mask_ch, (d1_o, d0_o),
                                             interpolation=cv2.INTER_NEAREST)
                mask = tmp
        return img, mask

    def randomize(self):
        self.state['p'] = random.random()
        self.state['r'] = round(random.uniform(*self.ratio_range), 2)


class Crop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            self.output_size = output_size
        else:
            raise ValueError('Incorrect value')
        # self.keep_size = keep_size
        # self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        rows_in, cols_in = img.shape[1:]
        rows_out, cols_out = self.output_size
        rows_out = min(rows_in, rows_out)
        cols_out = min(cols_in, cols_out)

        r0 = math.floor(self.state['r0f'] * (rows_in - rows_out))
        c0 = math.floor(self.state['c0f'] * (cols_in - cols_out))
        r1 = r0 + rows_out
        c1 = c0 + cols_out

        img = np.ascontiguousarray(img[:, r0:r1, c0:c1])
        if mask is not None:
            mask = np.ascontiguousarray(mask[:, r0:r1, c0:c1])
        return img, mask

    def randomize(self):
        # self.state['p'] = random.random()
        self.state['r0f'] = random.random()
        self.state['c0f'] = random.random()


class CenterCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        c, h, w = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2

        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = np.ascontiguousarray(img[:, y1:y2, x1:x2])
        if mask is not None:
            mask = np.ascontiguousarray(mask[:, y1:y2, x1:x2])

        return img, mask


class NoTransform(object):
    def __call__(self, *args):
        return args


class ToTensor(object):
    def __call__(self, *args):
        return [torch.from_numpy(e) for e in args]


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask=None):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std

        if mask is not None:
            mask = mask.astype(np.float32)
        return img, mask


def mixup_data(x, y, alpha=1.0, device='cpu'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


##visualization


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """

    def __init__(self, model):
        self.model = model

    def generate_gradients(self, input_image):
        # Put model in evaluation mode
        self.model.eval()

        x = input_image.clone()

        x.requires_grad = True

        with torch.enable_grad():
            # Forward
            model_output = self.model(x)
            # Zero grads
            self.model.zero_grad()

            grad = torch.autograd.grad(model_output, x, grad_outputs=model_output,
                                       only_inputs=True)[0]

            self.model.train()

        return grad


@numba.jit(nopython=True, nogil=True)
def gen_oracle_map(feat, ind, w, h):
    # feat: B x maxN x featDim
    # ind: B x maxN
    batch_size = feat.shape[0]
    max_objs = feat.shape[1]
    feat_dim = feat.shape[2]
    out = np.zeros((batch_size, feat_dim, h, w), dtype=np.float32)
    vis = np.zeros((batch_size, h, w), dtype=np.uint8)
    ds = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for i in range(batch_size):
        queue_ind = np.zeros((h * w * 2, 2), dtype=np.int32)
        queue_feat = np.zeros((h * w * 2, feat_dim), dtype=np.float32)
        head, tail = 0, 0
        for j in range(max_objs):
            if ind[i][j] > 0:
                x, y = ind[i][j] % w, ind[i][j] // w
                out[i, :, y, x] = feat[i][j]
                vis[i, y, x] = 1
                queue_ind[tail] = x, y
                queue_feat[tail] = feat[i][j]
                tail += 1
        while tail - head > 0:
            x, y = queue_ind[head]
            f = queue_feat[head]
            head += 1
            for (dx, dy) in ds:
                xx, yy = x + dx, y + dy
                if xx >= 0 and yy >= 0 and xx < w and yy < h and vis[i, yy, xx] < 1:
                    out[i, :, yy, xx] = f
                    vis[i, yy, xx] = 1
                    queue_ind[tail] = xx, yy
                    queue_feat[tail] = f
                    tail += 1
    return out


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian3D(shape, sigma=1):
    m, n, l = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-m:m + 1, -n:n + 1, -l:l + 1]

    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian3D((diameter, diameter, diameter), sigma=diameter / 6)

    x, y, z = int(center[0]), int(center[1]), int(center[2])

    height, width, length = heatmap.shape[0:3]

    left, right = min(x, radius), min(length - x, radius + 1)
    top, bottom = min(y, radius), min(width - y, radius + 1)
    head, tail = min(z, radius), min(height - z, radius + 1)

    masked_heatmap = heatmap[z - head:z + tail, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - head:radius + tail, radius - top:radius + bottom, radius - left:radius + right]
    # print("masked_gaussian: ",masked_gaussian.shape)
    # plt.imshow(masked_gaussian)
    # plt.savefig("hm.png")
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width, length = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    # print('feat',feat.shape,ind.shape)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # print('ind',  ind.shape)
    feat = feat.gather(1, ind)
    # print('feat', feat.shape)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    ## 取出对应index的 feat值
    feat = feat.permute(0, 2, 3, 4, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(4))
    feat = _gather_feat(feat, ind)
    return feat


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        # print(pred.shape,'pred...')
        # print(output.shape,'output')
        # print(target.shape,'target')
        loss = F.l1_loss(pred, target, size_average=False)
        loss = loss / (8 * ind.size()[0] + 1e-4)
        return loss


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool3d(
        heat, (kernel, kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width, length = scores.size()
    ## 找到hm 每个类别的 topk
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    ### 推出topk的坐标
    topk_inds = topk_inds % (height * width * length)
    topk_zs = (topk_inds // height * width).int().float()
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()
    ## 找到 80 * K 中的 topk
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=8):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    # 局部做一个简单的max
    heat = _nms(heat)
    ## 取出 top K 个 hm坐标，返回：topk的 hm值，对应index，对应所属类别，对应y坐标，对应x坐标
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 3)
        ## reg 是对 center 进行调整
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        zs = ys.view(batch, K, 1) + reg[:, :, 2:3]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 3)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        zs - wh[..., 2:3] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2,
                        zs + wh[..., 2:3] / 2], dim=3)
    detections = torch.cat([bboxes, scores, clses], dim=3)

    return detections


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        ## 从heatmap 坐标 转换成 原图坐标
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret

class DiceLoss_2(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_2, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

if __name__ == '__main__':
    WeightedDiceLoss()
    # torch.manual_seed(1)
    # # while(1):
    # x = torch.zeros(4,3,12,12)
    # x = torch.zeros(4, 3, 12, 12)
    # #gen = VNet().cuda()
    # # y = gen(x)
    # # print(y.shape)
    #
    # print('    Total params: %.2fMB' % (sum(p.numel() for p in gen.parameters()) / (1024.0 * 1024) * 4))
