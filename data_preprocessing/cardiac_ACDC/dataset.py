from operator import sub
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


class OAI_data(data.Dataset):
    def __init__(self,list_file,root_dir='../../../data/oai',sagittal_num=64,other_num=256, transforms=None, bd=False):
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
        self.other_num=other_num
        cnt = 0
        # self.inform = []
        # self.norm_empty_slice_list = []
        # self.slices_list = [0]
        #spacing_c_a_s = [0.3125, 0.3125, 3.3]  # [0.30303,0.30303,3.3]
        spacing_c_a_s=[0.417753906,0.417753906,1.5859375]
        #1.5859375	0.39921875	0.39921875
        self.root_dir=root_dir
        #New Spacing 384,384,30
        with open(list_file, 'r') as f:
            for i in f.readlines():
                path_i = i.split("\n")[0]
                self.file_path.append(os.path.join(self.root_dir,path_i))
        for i in self.file_path:
            if "bl" in i:
                mode = 0
            else:
                mode = 1
            name=os.path.basename(i)
            names=name.split("_")
            subject_name=names[0]+"_"+names[1]
            if mode ==0:
                path_i=os.path.join(self.root_dir,"img_with_mask","bl_sag_3d_dess_mhd",subject_name+".mha")
            else:
                path_i=os.path.join(self.root_dir,"img_with_mask","12m_sag_3d_dess_mhd",subject_name+".mhd")
            self.image_path.append(path_i)
            self.label_path.append(i)
            id = subject_name
            # print(path_i)
            img_sitk = sitk.ReadImage(path_i)
            label_sitk = sitk.ReadImage(i)
            #### resample to fixed spacing
            resampled_img_itk, resampled_label_itk = itk_resample(img_sitk, label_sitk, out_spacing=spacing_c_a_s,
                                                                  interpolation=sitk.sitkLinear,
                                                                  dtype=sitk.sitkFloat32)
            # print(resampled_img_itk.GetSize())
            resample_img = sitk.GetArrayFromImage(resampled_img_itk)
            resampled_label = sitk.GetArrayFromImage(resampled_label_itk)

            #### pad or crop to fixed size [30, 512, 512]
            # self.slices_list.append(self.slices_list[-1]+resample_img.shape[0])
            sagital_num_slices = sagittal_num # resample_img.shape[0]  ###################### notice
            # if isSWIN==True: sagital_num_slices = 32
            fixed_size_img = pad_crop(resample_img, (sagital_num_slices, self.other_num, self.other_num), mode='constant', value=0)
            fixed_size_label = pad_crop(resampled_label, (sagital_num_slices, self.other_num, self.other_num), mode='constant', value=0)
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
        return image[np.newaxis, :], label

    def __len__(self):
        return self.len

    def __repr__(self):
        return 'cases: ' + str(len(self.image))

class MyData_test(OAI_data):
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


class OAI_Data_non_label(data.Dataset):
    def __init__(self,list_file,root_dir='../../../data/oai',data_root='bl',sagittal_num=64,other_num=256, transforms=None, bd=False):
        self.image = []
        self.path = []
        self.idd = []
        self.transforms = transforms
        self.ret = []
        self.dist = []
        self.bd = bd
        self.image_path=[]
        self.label_path=[]
        self.file_path=[]
        self.other_num=other_num
        self.temp_zero = np.zeros((sagittal_num, self.other_num, self.other_num)).astype(np.uint8)
        cnt = 0
        # self.inform = []
        # self.norm_empty_slice_list = []
        # self.slices_list = [0]
        #spacing_c_a_s = [0.3125, 0.3125, 3.3]  # [0.30303,0.30303,3.3]
        spacing_c_a_s=[0.417753906,0.417753906,1.5859375]
        self.root_dir=root_dir

        with open(list_file, 'r') as f:
            for i in f.readlines():
                stuff=i.split()
                self.file_path.append(os.path.join(self.root_dir,data_root,stuff[0]+"_"+stuff[1]+"_"+stuff[2]+".mha"))
                self.idd.append(stuff[0])
        for i in self.file_path:
            path_i = i
            self.image_path.append(path_i)
            img_sitk = sitk.ReadImage(path_i)
            print(path_i)
            #### resample to fixed spacing
            resampled_img_itk = itk_resample_only(img_sitk, out_spacing=spacing_c_a_s,
                                                                  interpolation=sitk.sitkLinear,
                                                                  dtype=sitk.sitkFloat32)
            resample_img = sitk.GetArrayFromImage(resampled_img_itk)

            #### pad or crop to fixed size [30, 512, 512]
            # self.slices_list.append(self.slices_list[-1]+resample_img.shape[0])
            sagital_num_slices = sagittal_num # resample_img.shape[0]  ###################### notice
            # if isSWIN==True: sagital_num_slices = 32
            fixed_size_img = pad_crop(resample_img, (sagital_num_slices, self.other_num, self.other_num), mode='constant', value=0)            #### move the menis labels(3,4) to (7,8), not necessary
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
            self.path.append(path_i)
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
        fixed_size_label = self.temp_zero
        return image[np.newaxis, :], fixed_size_label.astype(np.uint8)

    def __len__(self):
        return self.len

    def __repr__(self):
        return 'cases: ' + str(len(self.image))

class MyData_SEMI(data.Dataset):
    def __init__(self,list_label,list_bl,list_12m,root_dir='../../../data/oai',sagittal_num=32,other_num=512):

        self.image = []
        self.label = []
        self.path = []
        self.right_flag = []
        self.idd = []
        self.sagittal_num=sagittal_num
        self.file_path = []
        self.file_path_labeled=[]
        self.file_path_labeled_label=[]
        self.file_path_bl=[]
        self.file_path_12m=[]
        self.filpper=sitk.FlipImageFilter()
        self.other_num=other_num
        self.filpper.SetFlipAxes([False,False,True])
        self.temp_zero = np.zeros((sagittal_num, self.other_num, self.other_num)).astype(np.uint8)
        ## load labeled dataset
        self.root_dir=root_dir
        to_exclude=[]
        with open(list_label, 'r') as f:
            for i in f.readlines():
                path_i = i.split("\n")[0]
                if "bl" in i:
                    mode = 0
                else:
                    mode = 1
                self.file_path_labeled_label.append(os.path.join(self.root_dir,path_i))
                name=os.path.basename(path_i)
                names=name.split("_")
                subject_name=names[0]+"_"+names[1]
                if mode == 0:
                    path_i = os.path.join(self.root_dir, "img_with_mask", "bl_sag_3d_dess_mhd", subject_name + ".mha")
                else:
                    path_i = os.path.join(self.root_dir, "img_with_mask", "12m_sag_3d_dess_mhd", subject_name + ".mhd")
                id = subject_name
                self.idd.append(id)
                self.file_path_labeled.append(path_i)
                self.path.append(path_i)
                to_exclude.append(subject_name)
        self.num_labeled = len(self.file_path_labeled_label)
        with open(list_bl, 'r') as f:
            for i in f.readlines():
                stuff = i.split()
                if stuff[0]+"_"+stuff[1] in to_exclude:
                    continue
                self.file_path_bl.append(
                    os.path.join(self.root_dir, "bl", stuff[0] + "_" + stuff[1] + "_" + stuff[2] + ".mha"))
                self.idd.append(stuff[0])
        self.num_bl=len(self.file_path_bl)
        with open(list_12m, 'r') as f:
            for i in f.readlines():
                stuff = i.split()
                if stuff[0]+"_"+stuff[1] in to_exclude:
                    continue
                self.file_path_bl.append(
                    os.path.join(self.root_dir, "12m", stuff[0] + "_" + stuff[1] + "_" + stuff[2] + ".mha"))
                self.idd.append(stuff[0])
        self.num_12m=len(self.file_path_12m)
        ## load unlabeled dataset
        # file_unlabeled = os.path.join(self.root_dir,'list_train_unlabel.txt')
        # with open(file_unlabeled, 'r') as f:
        #     for i in f.readlines():
        #         path_i = i.split("\n")[0]
        #         self.file_path.append(os.path.join(self.root_dir,path_i))

        self.num_unlabeled = self.num_12m+self.num_bl
        #print('_data_len: ',len(self.file_path))

    def __getitem__(self, index):
        if index<self.num_labeled:
            mode=0
            path_i=self.file_path_labeled[index]
            path_label=self.file_path_labeled_label[index]
        elif index-self.num_labeled<self.num_bl:
            mode=1
            path_i=self.file_path_bl[index-self.num_labeled]
            path_label=None
        else:
            mode=2
            path_i=self.file_path_12m[index-self.num_labeled-self.num_bl]
            path_label=None
        # path_i = self.file_path[index]
        # if 'org' in path_i:
        #     path_label = path_i.replace('org', 'seg')
        #     id = path_i.split('/')[-2].split('.')[0]
        # elif 'unlabel' in path_i:
        #     path_label = None
        ####
        #spacing_c_a_s = [0.3125, 0.3125, 3.3]
        #spacing_c_a_s=[0.36458333,0.36458333,3.3]
        spacing_c_a_s=[0.417753906,0.417753906,1.5859375]
        logging.info("Using Mydata Semi")
        sagital_num_slices = self.sagittal_num
        ####
        img_sitk = sitk.ReadImage(path_i)
        #print(path_i)
        if path_label == None:
            resampled_img_itk = itk_resample_only(img_sitk, out_spacing=spacing_c_a_s,
                                                  interpolation=sitk.sitkLinear, dtype=sitk.sitkFloat32)
            resample_img = sitk.GetArrayFromImage(resampled_img_itk)
            if "RIGHT" in path_i:
                resample_img=np.flip(resample_img,0)
            fixed_size_label = self.temp_zero
            # if "RIGHT" in path_i:
                #do something for right stuff
        else:
            label_sitk = sitk.ReadImage(path_label)
            #print(path_label)
            # logging.info()
            resampled_img_itk, resampled_label_itk = itk_resample(img_sitk, label_sitk, out_spacing=spacing_c_a_s,
                                                                  interpolation=sitk.sitkLinear, dtype=sitk.sitkFloat32)
            resampled_label = sitk.GetArrayFromImage(resampled_label_itk)
            fixed_size_label = pad_crop(resampled_label, (sagital_num_slices, self.other_num, self.other_num), mode='constant', value=0)
            resample_img = sitk.GetArrayFromImage(resampled_img_itk)


        fixed_size_img = pad_crop(resample_img, (sagital_num_slices, self.other_num, self.other_num), mode='constant', value=0)
        fixed_size_img = (fixed_size_img - fixed_size_img.mean()) / fixed_size_img.std()

        # print(fixed_size_img.shape,fixed_size_label.dtype)

        return fixed_size_img[np.newaxis, :], fixed_size_label.astype(np.uint8)

    def __len__(self):
        return self.num_12m+self.num_bl+self.num_labeled

    def __repr__(self):
        return 'train_data_len: labeled cases: ' + str(self.num_labeled) + '  unlabeled cases: ' + str(
            self.num_unlabeled)

class MyData_SEMI_did_opt(data.Dataset):
    def __init__(self,list_label,list_unlabel,root_dir='../../../data/oai',sagittal_num=32,other_num=512):
        #TODO:
        #1. 数据集输入为有label（img，label，left）tuple和无label的path to image
        #2. 
        self.image = []
        self.label = []
        self.path = []
        self.right_flag = []
        self.idd = []
        self.sagittal_num=sagittal_num
        self.file_path = []
        self.file_path_labeled=[]
        self.file_path_labeled_label=[]
        self.other_num=other_num
        # self.file_path_bl=[]
        # self.file_path_12m=[]
        self.file_path_unlabeled=[]
        self.filpper=sitk.FlipImageFilter()
        self.filpper.SetFlipAxes([False,False,True])
        self.temp_zero = np.zeros((sagittal_num, self.other_num, self.other_num)).astype(np.uint8)
        ## load labeled dataset
        self.root_dir=root_dir

        for subject_name,img,lbl in list_label:
            self.file_path_labeled.append(img)
            self.file_path_labeled_label.append(lbl)
            self.idd.append(subject_name)
        if list_unlabel==None:
            self.num_labeled=len(self.file_path_labeled)
            self.num_unlabeled=0
            return
        for subject_name,img in list_unlabel:
            self.file_path_unlabeled.append(img)
            self.idd.append(subject_name)
        
        self.num_labeled=len(self.file_path_labeled)
        self.num_unlabeled=len(self.file_path_unlabeled)
        logging.info("Using MyData_Semi_DID")

        # with open(list_label, 'r') as f:
        #     for i in f.readlines():
        #         path_i = i.split("\n")[0]
        #         if "bl" in i:
        #             mode = 0
        #         else:
        #             mode = 1
        #         self.file_path_labeled_label.append(os.path.join(self.root_dir,path_i))
        #         name=os.path.basename(path_i)
        #         names=name.split("_")
        #         subject_name=names[0]+"_"+names[1]
        #         if mode == 0:
        #             path_i = os.path.join(self.root_dir, "img_with_mask", "bl_sag_3d_dess_mhd", subject_name + ".mha")
        #         else:
        #             path_i = os.path.join(self.root_dir, "img_with_mask", "12m_sag_3d_dess_mhd", subject_name + ".mhd")
        #         id = subject_name
        #         self.idd.append(id)
        #         self.file_path_labeled.append(path_i)
        #         self.path.append(path_i)
        # self.num_labeled = len(self.file_path_labeled_label)
        # with open(list_bl, 'r') as f:
        #     for i in f.readlines():
        #         stuff = i.split()
        #         self.file_path_bl.append(
        #             os.path.join(self.root_dir, "bl", stuff[0] + "_" + stuff[1] + "_" + stuff[2] + ".mha"))
        #         self.idd.append(stuff[0])
        # self.num_bl=len(self.file_path_bl)
        # with open(list_12m, 'r') as f:
        #     for i in f.readlines():
        #         stuff = i.split()
        #         self.file_path_bl.append(
        #             os.path.join(self.root_dir, "12m", stuff[0] + "_" + stuff[1] + "_" + stuff[2] + ".mha"))
        #         self.idd.append(stuff[0])
        # self.num_12m=len(self.file_path_12m)
        ## load unlabeled dataset
        # file_unlabeled = os.path.join(self.root_dir,'list_train_unlabel.txt')
        # with open(file_unlabeled, 'r') as f:
        #     for i in f.readlines():
        #         path_i = i.split("\n")[0]
        #         self.file_path.append(os.path.join(self.root_dir,path_i))

        # self.num_unlabeled = self.num_12m+self.num_bl
        #print('_data_len: ',len(self.file_path))

    def __getitem__(self, index):
        if index<self.num_labeled:
            mode=0
            path_i=self.file_path_labeled[index]
            path_label=self.file_path_labeled_label[index]
        else:
            mode=1
            path_i=self.file_path_unlabeled[index-self.num_labeled]
            path_label=None
        # path_i = self.file_path[index]
        # if 'org' in path_i:
        #     path_label = path_i.replace('org', 'seg')
        #     id = path_i.split('/')[-2].split('.')[0]
        # elif 'unlabel' in path_i:
        #     path_label = None
        ####
        #spacing_c_a_s = [0.3125, 0.3125, 3.3]
        #spacing_c_a_s=[0.36458333,0.36458333,3.3]
        # spacing_c_a_s=[0.417753906,0.417753906,1.5859375]
        sagital_num_slices = self.sagittal_num
        ####
        # crop_filter=sitk.CropImageFilter()
        # # crop_filter.SetUpperBoundaryCropSize([64,64,0])
        # # crop_filter.SetLowerBoundaryCropSize([16,16,0])
        # crop_filter.SetUpperBoundaryCropSize([65,82,0])
        # crop_filter.SetLowerBoundaryCropSize([31,14,0])
        img_sitk = sitk.ReadImage(path_i)
        # img_sitk=crop_filter.Execute(img_sitk)
        #print(path_i)
        if path_label == None:
            
            # resampled_img_itk = itk_resample_only(img_sitk, out_spacing=spacing_c_a_s,
            #                                       interpolation=sitk.sitkLinear, dtype=sitk.sitkFloat32)
            resample_img = sitk.GetArrayFromImage(img_sitk)
            if "RIGHT" in path_i:
                resample_img=np.flip(resample_img,0)
            fixed_size_label = self.temp_zero
            # if "RIGHT" in path_i:
                #do something for right stuff
        else:
            label_sitk = sitk.ReadImage(path_label)
            # label_sitk=crop_filter.Execute(label_sitk)
            #print(path_label)
            # logging.info()
            # resampled_img_itk, resampled_label_itk = itk_resample(img_sitk, label_sitk, out_spacing=spacing_c_a_s,
            #                                                       interpolation=sitk.sitkLinear, dtype=sitk.sitkFloat32)
            fixed_size_label = sitk.GetArrayFromImage(label_sitk)
            # fixed_size_label = pad_crop(resampled_label, (sagital_num_slices, self.other_num, self.other_num), mode='constant', value=0)
            resample_img = sitk.GetArrayFromImage(img_sitk)


        #fixed_size_img = pad_crop(resample_img, (sagital_num_slices, self.other_num, self.other_num), mode='constant', value=0)
        # resample_img=pad_crop(resample_img,(sagital_num_slices,self.other_num,self.other_num), mode='constant', value=0)
        # fixed_size_label = pad_crop(fixed_size_label, (sagital_num_slices, self.other_num, self.other_num), mode='constant', value=0)

        fixed_size_img = (resample_img - resample_img.mean()) / resample_img.std()
        # logging.info(str(fixed_size_img.shape)+","+str(fixed_size_label.shape))
        # print(fixed_size_img.shape,fixed_size_label.dtype)

        return fixed_size_img[np.newaxis, :], fixed_size_label.astype(np.uint8)

    def __len__(self):
        return self.num_unlabeled+self.num_labeled

    def __repr__(self):
        return 'train_data_len: labeled cases: ' + str(self.num_labeled) + '  unlabeled cases: ' + str(
            self.num_unlabeled)



class MyData_SEMI_did(data.Dataset):
    def __init__(self,list_label,list_unlabel,root_dir='../../../data/oai',sagittal_num=32,other_num=512):
        #TODO:
        #1. 数据集输入为有label（img，label，left）tuple和无label的path to image
        #2. 
        self.image = []
        self.label = []
        self.path = []
        self.right_flag = []
        self.idd = []
        self.sagittal_num=sagittal_num
        self.file_path = []
        self.file_path_labeled=[]
        self.file_path_labeled_label=[]
        self.other_num=other_num
        # self.file_path_bl=[]
        # self.file_path_12m=[]
        self.file_path_unlabeled=[]
        self.filpper=sitk.FlipImageFilter()
        self.filpper.SetFlipAxes([False,False,True])
        #self.temp_zero = np.zeros((144, 224, 272)).astype(np.uint8)
        self.temp_zero = np.zeros((16,288,352)).astype(np.uint8)
        ## load labeled dataset
        self.root_dir=root_dir

        for subject_name,img,lbl in list_label:
            self.file_path_labeled.append(img)
            self.file_path_labeled_label.append(lbl)
            self.idd.append(subject_name)
        if list_unlabel==None:
            self.num_labeled=len(self.file_path_labeled)
            self.num_unlabeled=0
            return
        for subject_name,img in list_unlabel:
            self.file_path_unlabeled.append(img)
            self.idd.append(subject_name)
        
        self.num_labeled=len(self.file_path_labeled)
        self.num_unlabeled=len(self.file_path_unlabeled)
        logging.info("Using MyData_Semi_DID")

        # with open(list_label, 'r') as f:
        #     for i in f.readlines():
        #         path_i = i.split("\n")[0]
        #         if "bl" in i:
        #             mode = 0
        #         else:
        #             mode = 1
        #         self.file_path_labeled_label.append(os.path.join(self.root_dir,path_i))
        #         name=os.path.basename(path_i)
        #         names=name.split("_")
        #         subject_name=names[0]+"_"+names[1]
        #         if mode == 0:
        #             path_i = os.path.join(self.root_dir, "img_with_mask", "bl_sag_3d_dess_mhd", subject_name + ".mha")
        #         else:
        #             path_i = os.path.join(self.root_dir, "img_with_mask", "12m_sag_3d_dess_mhd", subject_name + ".mhd")
        #         id = subject_name
        #         self.idd.append(id)
        #         self.file_path_labeled.append(path_i)
        #         self.path.append(path_i)
        # self.num_labeled = len(self.file_path_labeled_label)
        # with open(list_bl, 'r') as f:
        #     for i in f.readlines():
        #         stuff = i.split()
        #         self.file_path_bl.append(
        #             os.path.join(self.root_dir, "bl", stuff[0] + "_" + stuff[1] + "_" + stuff[2] + ".mha"))
        #         self.idd.append(stuff[0])
        # self.num_bl=len(self.file_path_bl)
        # with open(list_12m, 'r') as f:
        #     for i in f.readlines():
        #         stuff = i.split()
        #         self.file_path_bl.append(
        #             os.path.join(self.root_dir, "12m", stuff[0] + "_" + stuff[1] + "_" + stuff[2] + ".mha"))
        #         self.idd.append(stuff[0])
        # self.num_12m=len(self.file_path_12m)
        ## load unlabeled dataset
        # file_unlabeled = os.path.join(self.root_dir,'list_train_unlabel.txt')
        # with open(file_unlabeled, 'r') as f:
        #     for i in f.readlines():
        #         path_i = i.split("\n")[0]
        #         self.file_path.append(os.path.join(self.root_dir,path_i))

        # self.num_unlabeled = self.num_12m+self.num_bl
        #print('_data_len: ',len(self.file_path))

    def __getitem__(self, index):
        if index<self.num_labeled:
            mode=0
            path_i=self.file_path_labeled[index]
            path_label=self.file_path_labeled_label[index]
        else:
            mode=1
            path_i=self.file_path_unlabeled[index-self.num_labeled]
            path_label=None
        # path_i = self.file_path[index]
        # if 'org' in path_i:
        #     path_label = path_i.replace('org', 'seg')
        #     id = path_i.split('/')[-2].split('.')[0]
        # elif 'unlabel' in path_i:
        #     path_label = None
        ####
        #spacing_c_a_s = [0.3125, 0.3125, 3.3]
        #spacing_c_a_s=[0.36458333,0.36458333,3.3]
        # spacing_c_a_s=[0.417753906,0.417753906,1.5859375]
        sagital_num_slices = self.sagittal_num
        ####
        # crop_filter=sitk.CropImageFilter()
        # # crop_filter.SetUpperBoundaryCropSize([64,64,0])
        # # crop_filter.SetLowerBoundaryCropSize([16,16,0])
        # crop_filter.SetUpperBoundaryCropSize([65,82,0])
        # crop_filter.SetLowerBoundaryCropSize([31,14,0])
        img_sitk = sitk.ReadImage(path_i)
        # img_sitk=crop_filter.Execute(img_sitk)
        #print(path_i)
        if path_label == None:
            
            # resampled_img_itk = itk_resample_only(img_sitk, out_spacing=spacing_c_a_s,
            #                                       interpolation=sitk.sitkLinear, dtype=sitk.sitkFloat32)
            resample_img = sitk.GetArrayFromImage(img_sitk)
            # if "RIGHT" in path_i:
            #     resample_img=np.flip(resample_img,0)
            fixed_size_label = self.temp_zero
            # if "RIGHT" in path_i:
                #do something for right stuff
        else:
            label_sitk = sitk.ReadImage(path_label)
            # label_sitk=crop_filter.Execute(label_sitk)
            #print(path_label)
            # logging.info()
            # resampled_img_itk, resampled_label_itk = itk_resample(img_sitk, label_sitk, out_spacing=spacing_c_a_s,
            #                                                       interpolation=sitk.sitkLinear, dtype=sitk.sitkFloat32)
            fixed_size_label = sitk.GetArrayFromImage(label_sitk)
            # fixed_size_label = pad_crop(resampled_label, (sagital_num_slices, self.other_num, self.other_num), mode='constant', value=0)
            resample_img = sitk.GetArrayFromImage(img_sitk)


        #fixed_size_img = pad_crop(resample_img, (sagital_num_slices, self.other_num, self.other_num), mode='constant', value=0)
        # resample_img=pad_crop(resample_img,(sagital_num_slices,self.other_num,self.other_num), mode='constant', value=0)
        # fixed_size_label = pad_crop(fixed_size_label, (sagital_num_slices, self.other_num, self.other_num), mode='constant', value=0)

        fixed_size_img = (resample_img - resample_img.mean()) / resample_img.std()
        # logging.info(str(fixed_size_img.shape)+","+str(fixed_size_label.shape))
        # print(fixed_size_img.shape,fixed_size_label.dtype)
        # logging.info(fixed_size_img.shape)
        # logging.info(fixed_size_label.shape)
        return fixed_size_img[np.newaxis, :], fixed_size_label.astype(np.uint8)

    def __len__(self):
        return self.num_unlabeled+self.num_labeled

    def __repr__(self):
        return 'train_data_len: labeled cases: ' + str(self.num_labeled) + '  unlabeled cases: ' + str(
            self.num_unlabeled)

# ### data load for OAI
# class MyData_OAI(MyData):  ## children of MyData
#     def __init__(self, file, transforms=None):
#         self.istrain = True if 'train' in file else False
#         self.image = []
#         self.label = []
#         self.path = []
#         self.right_flag = []
#         self.idd = []
#         self.transforms = transforms
#
#         print('---OAI-dataloader---')
#
#         # spacing_c_a_s =[0.3125,0.3125,3.3] #[0.30303,0.30303,3.3]
#         right_flag = dict()  # 暂时没有找到right标签
#         with open('oai/knee_right_desc.txt', 'r') as f:
#             for i in f.readlines():
#                 ff = i.split("\n")[0]
#                 # print(ff,' mm  ',ff[0:-2])
#                 right_flag[ff[0:-2]] = int(ff[-1])
#         with open(file, 'r') as f:
#             for i in f.readlines():
#
#                 path_i = i.split("\n")[0]
#                 id_whole = basename(path_i).split('.')[0]
#                 bl_12m = path_i.split('/')[-2].split('_')[0]
#                 path_label = '/media/likang/xinbinyu/dataset/to_sensetime_onlyLabels/' + id_whole + '_' + bl_12m + '_lb.mhd'
#                 path_label = path_label
#                 id = id_whole.split('_')[0]
#
#                 img_sitk = sitk.ReadImage(path_i)
#                 label_sitk = sitk.ReadImage(path_label)
#                 if right_flag[id] == 1:
#                     img_sitk = itk_right2left(img_sitk)
#                     label_sitk = itk_right2left(label_sitk)
#                 # resampled_img_itk,resampled_label_itk = itk_resample(img_sitk, label_sitk, out_spacing =spacing_c_a_s, interpolation=sitk.sitkLinear, dtype=sitk.sitkFloat32)
#
#                 if not debug:
#                     debug_arr0 = sitk.GetArrayFromImage(resampled_img_itk)
#                     print('debug: resampled size:', debug_arr0.shape)
#
#                 resample_img = sitk.GetArrayFromImage(img_sitk)
#                 resampled_label = sitk.GetArrayFromImage(label_sitk)
#                 # sagital_num_slices = 30#resample_img.shape[0]  ###################### notice
#                 # print('sagital_num_slices',sagital_num_slices)
#                 fixed_size_img = resample_img  # pad_crop(resample_img, (128, 448, 352), mode='constant', value=0)
#                 fixed_size_label = resampled_label  # pad_crop(resampled_label, (128, 448, 352), mode='constant', value=0)
#
#                 # resampled_itk = resamplingImage_Sp2(img_sitk,spacing_c_a_s,sampling_way = sitk.sitkLinear,sampling_pixel = sitk.sitkFloat32)
#                 if not debug:
#                     print('debug: fixed img size:', fixed_size_img.shape)
#                 # image = nib.load(path_i).get_data()
#                 # image = image/image.max()
#                 fixed_size_img = (fixed_size_img - fixed_size_img.mean()) / fixed_size_img.std()
#                 #  print(image.max())
#                 #   label = nib.load(path_i.replace('image','label')).get_data().astype('uint8')
#                 self.image.append(fixed_size_img)
#                 self.label.append(fixed_size_label.astype(np.uint8))
#                 self.path.append(path_i)
#                 self.idd.append(id_whole)
#                 # self.right_flag.append(right_flag[id])
#                 # print(fixed_size_label.astype(np.uint8).max())
#                 if 0:
#                     img_itk = sitk.GetImageFromArray(fixed_size_img)
#                     img_itk.SetSpacing(resampled_img_itk.GetSpacing())
#                     img_itk.SetOrigin(resampled_img_itk.GetOrigin())
#                     img_itk.SetDirection(resampled_img_itk.GetDirection())
#
#                     label_itk = sitk.GetImageFromArray(fixed_size_label)
#                     label_itk.SetSpacing(resampled_img_itk.GetSpacing())
#                     label_itk.SetOrigin(resampled_img_itk.GetOrigin())
#                     label_itk.SetDirection(resampled_img_itk.GetDirection())
#
#                     sitk.WriteImage(img_itk,
#                                     '/media/likang/xinbinyu/dataset/seg_data_knee_MRI/T2/knee_seg_arg/output/resampled/' + id + '_img.mha')
#                     sitk.WriteImage(label_itk,
#                                     '/media/likang/xinbinyu/dataset/seg_data_knee_MRI/T2/knee_seg_arg/output/resampled/' + id + '_label.mha')
#         # print('_data_len: ',len(self.image))
#
#     def __getitem__(self, index):
#         image = self.image[index]
#         label = self.label[index]
#         path = self.path[index]
#         id = self.idd[index]
#
#         return image[np.newaxis, :], label  # ,path,id,right

if __name__=='__main__':
    # ist_label, list_bl, list_12m, root_dir = '../../../data/oai', sagittal_num = 32
    # root_dir = '../../../data/OAI'
    # list_label="train.txt"
    # list_bl="bl_sag_3d_dess_leftRight.txt"
    # list_12m="12m_sag_3d_dess_leftRight.txt"
    train_data = MyData_SEMI("../../data/OAI/train.txt","../../data/OAI/bl_sag_3d_dess_leftRight.txt","../../data/OAI/12m_sag_3d_dess_leftRight.txt",root_dir="../../data/OAI",other_num=256)
    for x,y in train_data:
        print(x.shape)
        print(y.shape)
    n_train = len(train_data.file_path)  # Number of training samples
    # for i in train_data:
    #     print(i.)
    #
    # # # if partition == "homo":
    # total_num = n_train
    # idxs = np.random.permutation(total_num)
    # batch_idxs = np.array_split(idxs, 10)  # As many splits as n_nets = number of clients
    # net_data_idx_map = {i: batch_idxs[i] for i in range(10)}
    # data_subset=[Subset(train_data, batch) for batch in batch_idxs]
