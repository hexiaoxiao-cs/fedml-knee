import numpy as np
from scipy.ndimage import distance_transform_edt as distance
import SimpleITK as sitk
from skimage.measure import label
import operator
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


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res
