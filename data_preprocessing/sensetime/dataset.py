import os.path
import logging
from torch.utils.data import Subset
import nibabel as nib
from torch.utils import data
import numpy as np
import glob
import SimpleITK as sitk
from .utils import *
from os.path import basename, join
# import torchio
import cv2
from tqdm import tqdm
import math
# torchio.Subject
# import  matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)


class MyData(data.Dataset):
    def __init__(self,list_file,sagittal_num=32, transforms=None, bd=False):
        self.image = []
        self.label = []
        self.path = []
        self.idd = []
        self.transforms = transforms
        self.ret = []
        self.dist = []
        self.bd = bd
        self.image_path=[]
        self.label_path=[]
        self.file_path=[]
        cnt = 0
        # self.inform = []
        # self.norm_empty_slice_list = []
        # self.slices_list = [0]
        spacing_c_a_s = [0.3125, 0.3125, 3.3]  # [0.30303,0.30303,3.3]
        self.root_dir='../../../data/sensetime'

        with open(list_file, 'r') as f:
            for i in f.readlines():
                path_i = i.split("\n")[0]
                self.file_path.append(os.path.join(self.root_dir,path_i))
        for i in self.file_path:
            # cnt+=1
            # if cnt>1:
            #     continue
            #### read img and label and id from files
            path_i = i
            self.image_path.append(path_i)
            path_label = path_i.replace('org', 'seg')
            self.label_path.append(path_label)
            id = path_i.split('/')[-2].split('.')[0]
            img_sitk = sitk.ReadImage(path_i)
            label_sitk = sitk.ReadImage(path_label)
            print(path_i)
            #### resample to fixed spacing
            resampled_img_itk, resampled_label_itk = itk_resample(img_sitk, label_sitk, out_spacing=spacing_c_a_s,
                                                                  interpolation=sitk.sitkLinear,
                                                                  dtype=sitk.sitkFloat32)
            resample_img = sitk.GetArrayFromImage(resampled_img_itk)
            resampled_label = sitk.GetArrayFromImage(resampled_label_itk)

            #### pad or crop to fixed size [30, 512, 512]
            # self.slices_list.append(self.slices_list[-1]+resample_img.shape[0])
            sagital_num_slices = sagittal_num # resample_img.shape[0]  ###################### notice
            # if isSWIN==True: sagital_num_slices = 32
            fixed_size_img = pad_crop(resample_img, (sagital_num_slices, 512, 512), mode='constant', value=0)
            fixed_size_label = pad_crop(resampled_label, (sagital_num_slices, 512, 512), mode='constant', value=0)
            #### move the menis labels(3,4) to (7,8), not necessary
            # tongyi label value
            fixed_size_label[fixed_size_label == 3] = 9
            fixed_size_label[fixed_size_label == 4] = 10
            fixed_size_label[fixed_size_label == 5] = 3
            fixed_size_label[fixed_size_label == 6] = 4
            fixed_size_label[fixed_size_label == 7] = 5
            fixed_size_label[fixed_size_label == 8] = 6
            ## used for menius
            fixed_size_label[fixed_size_label == 9] = 7
            fixed_size_label[fixed_size_label == 10] = 8

            #### z-score norm
            fixed_size_img = (fixed_size_img - fixed_size_img.mean()) / fixed_size_img.std()

            # if you need use bd-loss
            # if self.bd:
            #     #### calculate dist
            #     label_onehot = np.stack([fixed_size_label == c for c in range(9)], axis=0).astype(np.int32)
            #     label_dist = one_hot2dist(label_onehot).astype(np.float32)
            #     self.dist.append(label_dist)

            #### append arrays and ids to list for dataloader
            #print(fixed_size_img.shape)
            #print(fixed_size_label.shape)
            self.image.append(fixed_size_img)
            self.label.append(fixed_size_label.astype(np.uint8))
            self.path.append(path_i)
            self.idd.append(id)
            # self.ret.append(ret)
            #### if you need to save the resampled imgs and labels
            # if 0:
            #     img_itk = sitk.GetImageFromArray(fixed_size_img)
            #     img_itk.SetSpacing(resampled_img_itk.GetSpacing())
            #     img_itk.SetOrigin(resampled_img_itk.GetOrigin())
            #     img_itk.SetDirection(resampled_img_itk.GetDirection())
            #
            #     label_itk = sitk.GetImageFromArray(fixed_size_label)
            #     label_itk.SetSpacing(resampled_img_itk.GetSpacing())
            #     label_itk.SetOrigin(resampled_img_itk.GetOrigin())
            #     label_itk.SetDirection(resampled_img_itk.GetDirection())
            #
            #     sitk.WriteImage(img_itk, 'output/resampled/' + id + '_img.mha')
            #     sitk.WriteImage(label_itk, 'output/resampled/' + id + '_label.mha')
        self.len = len(self.image)

    def __getitem__(self, index):
        image = self.image[index]
        label = self.label[index]
        return image[np.newaxis, :], label

    def __len__(self):
        return self.len

    def __repr__(self):
        return 'cases: ' + str(len(self.image))

class MyData_test(MyData):
    '''
    used when save the prediction
    '''

    def __getitem__(self, index):
        image = self.image[index]
        label = self.label[index]
        if 0:
            dist = self.dist[index]
        path = self.path[index]
        return image[np.newaxis, :], label  # , dist #,id,ret  #,path,id,right
class MyData_SEMI(data.Dataset):
    def __init__(self,sagittal_num,isOAI=False):

        self.image = []
        self.label = []
        self.path = []
        self.right_flag = []
        self.idd = []
        self.sagittal_num=sagittal_num
        self.file_path = []
        self.temp_zero = np.zeros((144, 224, 272)).astype(np.uint8)
        self.isOAI=isOAI
        ## load labeled dataset
        self.root_dir='../../../data/sensetime'
        file_labeled = os.path.join(self.root_dir,'list_train_ST_lbl.txt')
        with open(file_labeled, 'r') as f:
            for i in f.readlines():
                path_i = i.split("\n")[0]
                self.file_path.append(os.path.join(self.root_dir,path_i))
        self.num_labeled = len(self.file_path)
        ## load unlabeled dataset
        # file_unlabeled = os.path.join(self.root_dir,'list_train_unlabel_new.txt')
        for i in glob.glob(os.path.join(self.root_dir,"st_format_unlabeled","*.mha")):
            path_i=i
            self.file_path.append(path_i)
        # with open(file_unlabeled, 'r') as f:
        #     for i in f.readlines():
        #         path_i = i.split("\n")[0]
        #         self.file_path.append(os.path.join(self.root_dir,path_i))
        self.num_unlabeled = len(self.file_path) - self.num_labeled
        #print('_data_len: ',len(self.file_path))

    def __getitem__(self, index):
        path_i = self.file_path[index]
        if 'org' in path_i:
            path_label = path_i.replace('org', 'seg')
            id = path_i.split('/')[-2].split('.')[0]
        else:
            path_label=None
        ####
        # spacing_c_a_s = [0.3125, 0.3125, 3.3]
        # sagital_num_slices = self.sagittal_num
        ####
        img_sitk = sitk.ReadImage(path_i)
        if path_label == None:
            # resampled_img_itk = itk_resample_only(img_sitk, out_spacing=spacing_c_a_s,
            #                                       interpolation=sitk.sitkLinear, dtype=sitk.sitkFloat32)
            resampled_img_itk=img_sitk
            fixed_size_label = self.temp_zero
        else:
            label_sitk = sitk.ReadImage(path_label)
            # resampled_img_itk, resampled_label_itk = itk_resample(img_sitk, label_sitk, out_spacing=spacing_c_a_s,
            #                                                       interpolation=sitk.sitkLinear, dtype=sitk.sitkFloat32)
            fixed_size_label = sitk.GetArrayFromImage(label_sitk)
            # fixed_size_label = pad_crop(resampled_label, (sagital_num_slices, 512, 512), mode='constant', value=0)

            # if (self.isOAI == True):
            #     fixed_size_label[fixed_size_label == 2] = 1
            #     fixed_size_label[fixed_size_label == 6] = 2
            #     fixed_size_label[fixed_size_label == 8] = 3
            #     fixed_size_label[fixed_size_label > 3] = 0
            # else:
            #     fixed_size_label[fixed_size_label == 3] = 9
            #     fixed_size_label[fixed_size_label == 4] = 10
            #     fixed_size_label[fixed_size_label == 5] = 3
            #     fixed_size_label[fixed_size_label == 6] = 4
            #     fixed_size_label[fixed_size_label == 7] = 5
            #     fixed_size_label[fixed_size_label == 8] = 6
            #     ## used for menius
            #     fixed_size_label[fixed_size_label == 9] = 7
            #     fixed_size_label[fixed_size_label == 10] = 8

        fixed_size_img = sitk.GetArrayFromImage(img_sitk)
        # fixed_size_img = pad_crop(resample_img, (sagital_num_slices, 512, 512), mode='constant', value=0)
        fixed_size_img = (fixed_size_img - fixed_size_img.mean()) / fixed_size_img.std()

        # print(fixed_size_img.shape,fixed_size_label.dtype)

        return fixed_size_img[np.newaxis, :], fixed_size_label.astype(np.uint8)

    def __len__(self):
        return len(self.file_path)

    def __repr__(self):
        return 'train_data_len: labeled cases: ' + str(self.num_labeled) + '  unlabeled cases: ' + str(
            self.num_unlabeled)

class MyData_SEMI_new(data.Dataset):
    def __init__(self,sagittal_num,isOAI=False):

        self.image = []
        self.label = []
        self.path = []
        self.right_flag = []
        self.idd = []
        self.sagittal_num=sagittal_num
        self.file_path = []
        self.temp_zero = np.zeros((16, 288, 352)).astype(np.uint8)
        self.isOAI=isOAI
        ## load labeled dataset
        self.root_dir='../../../data/sensetime'
        file_labeled = os.path.join(self.root_dir,'list_train_ST_lbl.txt')
        with open(file_labeled, 'r') as f:
            for i in f.readlines():
                path_i = i.split("\n")[0]
                self.file_path.append(os.path.join(self.root_dir,path_i))
        self.num_labeled = len(self.file_path)
        ## load unlabeled dataset
        # file_unlabeled = os.path.join(self.root_dir,'list_train_unlabel_new.txt')
        for i in glob.glob(os.path.join(self.root_dir,"st_format_unlabeled","*.nii.gz")):
            path_i=i
            self.file_path.append(path_i)
        # with open(file_unlabeled, 'r') as f:
        #     for i in f.readlines():
        #         path_i = i.split("\n")[0]
        #         self.file_path.append(os.path.join(self.root_dir,path_i))
        self.num_unlabeled = len(self.file_path) - self.num_labeled
        #print('_data_len: ',len(self.file_path))

    def __getitem__(self, index):
        path_i = self.file_path[index]
        if 'org' in path_i:
            path_label = path_i.replace('org', 'seg')
            id = path_i.split('/')[-2].split('.')[0]
        else:
            path_label=None
        ####
        # spacing_c_a_s = [0.3125, 0.3125, 3.3]
        # sagital_num_slices = self.sagittal_num
        ####
        img_sitk = sitk.ReadImage(path_i)
        if path_label == None:
            # resampled_img_itk = itk_resample_only(img_sitk, out_spacing=spacing_c_a_s,
            #                                       interpolation=sitk.sitkLinear, dtype=sitk.sitkFloat32)
            resampled_img_itk=img_sitk
            fixed_size_label = self.temp_zero
        else:
            label_sitk = sitk.ReadImage(path_label)
            # resampled_img_itk, resampled_label_itk = itk_resample(img_sitk, label_sitk, out_spacing=spacing_c_a_s,
            #                                                       interpolation=sitk.sitkLinear, dtype=sitk.sitkFloat32)
            fixed_size_label = sitk.GetArrayFromImage(label_sitk)
            # fixed_size_label = pad_crop(resampled_label, (sagital_num_slices, 512, 512), mode='constant', value=0)

            # if (self.isOAI == True):
            #     fixed_size_label[fixed_size_label == 2] = 1
            #     fixed_size_label[fixed_size_label == 6] = 2
            #     fixed_size_label[fixed_size_label == 8] = 3
            #     fixed_size_label[fixed_size_label > 3] = 0
            # else:
            #     fixed_size_label[fixed_size_label == 3] = 9
            #     fixed_size_label[fixed_size_label == 4] = 10
            #     fixed_size_label[fixed_size_label == 5] = 3
            #     fixed_size_label[fixed_size_label == 6] = 4
            #     fixed_size_label[fixed_size_label == 7] = 5
            #     fixed_size_label[fixed_size_label == 8] = 6
            #     ## used for menius
            #     fixed_size_label[fixed_size_label == 9] = 7
            #     fixed_size_label[fixed_size_label == 10] = 8

        fixed_size_img = sitk.GetArrayFromImage(img_sitk)
        # fixed_size_img = pad_crop(resample_img, (sagital_num_slices, 512, 512), mode='constant', value=0)
        fixed_size_img = (fixed_size_img - fixed_size_img.mean()) / fixed_size_img.std()

        # print(fixed_size_img.shape,fixed_size_label.dtype)

        return fixed_size_img[np.newaxis, :], fixed_size_label.astype(np.uint8)

    def __len__(self):
        return len(self.file_path)

    def __repr__(self):
        return 'train_data_len: labeled cases: ' + str(self.num_labeled) + '  unlabeled cases: ' + str(
            self.num_unlabeled)

class MyData_val_semi(data.Dataset):
    def __init__(self, file, sagital_num_slices, isOAI=False, transforms=None):
        self.istrain = True if 'train' in file else False
        self.image = []
        self.label = []
        self.path = []
        self.idd = []
        self.transforms = transforms
        self.ret = []
        self.dist = []
        self.root_dir="../../../data/sensetime/"
        cnt = 0
        # self.inform = []
        # self.norm_empty_slice_list = []
        # self.slices_list = [0]
        # spacing_c_a_s = [0.3125, 0.3125, 3.3]  # [0.30303,0.30303,3.3]
        with open(file, 'r') as f:
            for i in f.readlines():
                # cnt+=1
                # if cnt>1:
                #     continue

                #### read img and label and id from files
                path_i = i.split("\n")[0]
                path_i=os.path.join(self.root_dir,path_i)
                path_label = path_i.replace('org', 'seg')
                id = path_i.split('/')[-2].split('.')[0]
                img_sitk = sitk.ReadImage(path_i)
                label_sitk = sitk.ReadImage(path_label)

                #### resample to fixed spacing
                # resampled_img_itk, resampled_label_itk = img_sitk,label_sitk
                fixed_size_img = sitk.GetArrayFromImage(img_sitk)
                fixed_size_label = sitk.GetArrayFromImage(label_sitk)

                #### pad or crop to fixed size [30, 512, 512]
                # self.slices_list.append(self.slices_list[-1]+resample_img.shape[0])
                # fixed_size_img = pad_crop(resample_img, (sagital_num_slices, 512, 512), mode='constant', value=0)
                # fixed_size_label = pad_crop(resampled_label, (sagital_num_slices, 512, 512), mode='constant', value=0)
                #### move the menis labels(3,4) to (7,8), not necessary
                # tongyi label value
                # if(isOAI==True):
                #     fixed_size_label[fixed_size_label == 2] = 1
                #     fixed_size_label[fixed_size_label == 6] = 2
                #     fixed_size_label[fixed_size_label == 8] = 3
                #     fixed_size_label[fixed_size_label > 3] = 0
                # else:
                #     fixed_size_label[fixed_size_label == 3] = 9
                #     fixed_size_label[fixed_size_label == 4] = 10
                #     fixed_size_label[fixed_size_label == 5] = 3
                #     fixed_size_label[fixed_size_label == 6] = 4
                #     fixed_size_label[fixed_size_label == 7] = 5
                #     fixed_size_label[fixed_size_label == 8] = 6
                #     ## used for menius
                #     fixed_size_label[fixed_size_label == 9] = 7
                #     fixed_size_label[fixed_size_label == 10] = 8

                #### z-score norm
                fixed_size_img = (fixed_size_img - fixed_size_img.mean()) / fixed_size_img.std()

                # if you need use bd-loss
                # if self.bd:
                #     #### calculate dist
                #     label_onehot = np.stack([fixed_size_label == c for c in range(9)], axis=0).astype(np.int32)
                #     label_dist = one_hot2dist(label_onehot).astype(np.float32)
                #     self.dist.append(label_dist)

                #### append arrays and ids to list for dataloader
                #print(fixed_size_img.shape)
                #print(fixed_size_label.shape)
                self.image.append(fixed_size_img)
                self.label.append(fixed_size_label.astype(np.uint8))
                self.path.append(path_i)
                self.idd.append(id)
                # self.ret.append(ret)
                #### if you need to save the resampled imgs and labels
                # if 0:
                #     img_itk = sitk.GetImageFromArray(fixed_size_img)
                #     img_itk.SetSpacing(resampled_img_itk.GetSpacing())
                #     img_itk.SetOrigin(resampled_img_itk.GetOrigin())
                #     img_itk.SetDirection(resampled_img_itk.GetDirection())
                #
                #     label_itk = sitk.GetImageFromArray(fixed_size_label)
                #     label_itk.SetSpacing(resampled_img_itk.GetSpacing())
                #     label_itk.SetOrigin(resampled_img_itk.GetOrigin())
                #     label_itk.SetDirection(resampled_img_itk.GetDirection())
                #
                #     sitk.WriteImage(img_itk, 'output/resampled/' + id + '_img.mha')
                #     sitk.WriteImage(label_itk, 'output/resampled/' + id + '_label.mha')
        self.len = len(self.image)

    def __getitem__(self, index):
        image = self.image[index]
        label = self.label[index]
        label[label == 7] = 0
        label[label == 8] = 0

        return image[np.newaxis, :], label  # , dist #,id,ret  #,path,id,right

    def __len__(self):
        return self.len

    def __repr__(self):
        return 'cases: ' + str(len(self.image))

class MyData_val_semi_OAI(data.Dataset):
    def __init__(self, file, sagital_num_slices, transforms=None):
        self.istrain = True if 'train' in file else False
        self.image = []
        self.label = []
        self.path = []
        self.idd = []
        self.transforms = transforms
        self.ret = []
        self.dist = []
        self.root_dir="../../../data/sensetime/"
        cnt = 0
        # self.inform = []
        # self.norm_empty_slice_list = []
        # self.slices_list = [0]
        spacing_c_a_s = [0.3125, 0.3125, 3.3]  # [0.30303,0.30303,3.3]
        with open(file, 'r') as f:
            for i in f.readlines():
                # cnt+=1
                # if cnt>1:
                #     continue

                #### read img and label and id from files
                path_i = i.split("\n")[0]
                path_i=os.path.join(self.root_dir,path_i)
                path_label = path_i.replace('org', 'seg')
                id = path_i.split('/')[-2].split('.')[0]
                img_sitk = sitk.ReadImage(path_i)
                label_sitk = sitk.ReadImage(path_label)

                #### resample to fixed spacing
                resampled_img_itk, resampled_label_itk = itk_resample(img_sitk, label_sitk, out_spacing=spacing_c_a_s,
                                                                      interpolation=sitk.sitkLinear,
                                                                      dtype=sitk.sitkFloat32)
                resample_img = sitk.GetArrayFromImage(resampled_img_itk)
                resampled_label = sitk.GetArrayFromImage(resampled_label_itk)

                #### pad or crop to fixed size [30, 512, 512]
                # self.slices_list.append(self.slices_list[-1]+resample_img.shape[0])
                fixed_size_img = pad_crop(resample_img, (sagital_num_slices, 512, 512), mode='constant', value=0)
                fixed_size_label = pad_crop(resampled_label, (sagital_num_slices, 512, 512), mode='constant', value=0)
                #### move the menis labels(3,4) to (7,8), not necessary
                # tongyi label value
                # fixed_size_label[fixed_size_label == 3] = 9
                # fixed_size_label[fixed_size_label == 4] = 10
                # fixed_size_label[fixed_size_label == 5] = 3
                # fixed_size_label[fixed_size_label == 6] = 4
                # fixed_size_label[fixed_size_label == 7] = 5
                # fixed_size_label[fixed_size_label == 8] = 6
                # ## used for menius
                # fixed_size_label[fixed_size_label == 9] = 7
                # fixed_size_label[fixed_size_label == 10] = 8


                #### z-score norm
                fixed_size_img = (fixed_size_img - fixed_size_img.mean()) / fixed_size_img.std()

                # if you need use bd-loss
                # if self.bd:
                #     #### calculate dist
                #     label_onehot = np.stack([fixed_size_label == c for c in range(9)], axis=0).astype(np.int32)
                #     label_dist = one_hot2dist(label_onehot).astype(np.float32)
                #     self.dist.append(label_dist)

                #### append arrays and ids to list for dataloader
                #print(fixed_size_img.shape)
                #print(fixed_size_label.shape)
                self.image.append(fixed_size_img)
                self.label.append(fixed_size_label.astype(np.uint8))
                self.path.append(path_i)
                self.idd.append(id)
                # self.ret.append(ret)
                #### if you need to save the resampled imgs and labels
                # if 0:
                #     img_itk = sitk.GetImageFromArray(fixed_size_img)
                #     img_itk.SetSpacing(resampled_img_itk.GetSpacing())
                #     img_itk.SetOrigin(resampled_img_itk.GetOrigin())
                #     img_itk.SetDirection(resampled_img_itk.GetDirection())
                #
                #     label_itk = sitk.GetImageFromArray(fixed_size_label)
                #     label_itk.SetSpacing(resampled_img_itk.GetSpacing())
                #     label_itk.SetOrigin(resampled_img_itk.GetOrigin())
                #     label_itk.SetDirection(resampled_img_itk.GetDirection())
                #
                #     sitk.WriteImage(img_itk, 'output/resampled/' + id + '_img.mha')
                #     sitk.WriteImage(label_itk, 'output/resampled/' + id + '_label.mha')
        self.len = len(self.image)

    def __getitem__(self, index):
        image = self.image[index]
        label = self.label[index]
        # label[label == 7] = 0
        # label[label == 8] = 0

        return image[np.newaxis, :], label  # , dist #,id,ret  #,path,id,right

    def __len__(self):
        return self.len

    def __repr__(self):
        return 'cases: ' + str(len(self.image))
if __name__=='__main__':
    train_data = MyData_SEMI()
    n_train = len(train_data.file_path)  # Number of training samples

    #
    # # if partition == "homo":
    total_num = n_train
    idxs = np.random.permutation(total_num)
    batch_idxs = np.array_split(idxs, 10)  # As many splits as n_nets = number of clients
    net_data_idx_map = {i: batch_idxs[i] for i in range(10)}
    data_subset=[Subset(train_data, batch) for batch in batch_idxs]
