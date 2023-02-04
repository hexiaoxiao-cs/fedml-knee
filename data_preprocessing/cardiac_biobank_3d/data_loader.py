from .dataset import *
import numpy as np
from torch.utils.data import *
import random
import itertools
import logging
# # for local devices
# def get_dataloader_test(data_dir, train_bs, test_bs, image_size, data_idxs_train=None, data_idxs_test=None):
#     return get_dataloader_cityscapes_test(data_dir, train_bs, test_bs, image_size, data_idxs_train, data_idxs_test)

def load_OAI_Data(list_file,root_dir,sagittal_slice=32,other_num=512):
    # transform_train, transform_test = _data_transforms_cityscapes(image_size)
    # root_dir = '../../../data/sensetime'
    list_file=os.path.join(root_dir,list_file)
    train_ds = OAI_data(list_file,root_dir=root_dir,sagittal_num=sagittal_slice,other_num=other_num)


    # train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    # test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_ds

def load_sensetime_data_SEMI(list_label,list_bl,list_12m,root_dir,sagittal_slice=32,other_num=512):
    # transform_train, transform_test = _data_transforms_cityscapes(image_size)
    list_label=os.path.join(root_dir,list_label)
    list_bl=os.path.join(root_dir,list_bl)
    list_12m=os.path.join(root_dir,list_12m)
    train_ds = MyData_SEMI(list_label,list_bl,list_12m,root_dir=root_dir,sagittal_num=sagittal_slice,other_num=other_num)
    # test_ds = MyData_test(data_dir,sagittal_num=sagittal_slice)

    # train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    # test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_ds

def load_sensetime_data_SEMI_with_did(list_label,list_bl,list_12m,train_did,val_did,root_dir,sagittal_slice=32,other_num=512):
    # transform_train, transform_test = _data_transforms_cityscapes(image_size)
    list_label=os.path.join(root_dir,list_label)
    list_bl=os.path.join(root_dir,list_bl)
    list_12m=os.path.join(root_dir,list_12m)
    train_ds = MyData_SEMI_did(list_label,list_bl,list_12m,root_dir=root_dir,sagittal_num=sagittal_slice,other_num=other_num)
    # test_ds = MyData_test(data_dir,sagittal_num=sagittal_slice)

    # train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    # test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_ds
# def record_data_stats(y_train, net_dataidx_map, task='segmentation'):
#     net_cls_counts = {}
#
#     for net_i, dataidx in net_dataidx_map.items():
#         unq, unq_cnt = np.unique(np.concatenate(y_train[dataidx]), return_counts=True) if task == 'segmentation' \
#             else np.unique(y_train[dataidx], return_counts=True)
#         tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
#         net_cls_counts[net_i] = tmp
#     logging.debug('Data statistics: %s' % str(net_cls_counts))
#     return net_cls_counts
def partition_data(n_nets, sagittal_slice=32,other_num=512):
    logging.info("********************* Partitioning data **********************")
    train_ds= load_OAI_Data("train.txt","../../../data/OAI",sagittal_slice,other_num=other_num)
    n_train = len(train_ds)  # Number of training samples

    # if partition == "homo":
    total_num = n_train
    #Since we have one label and one unlabel data, we will split such data evenly
    labeled_num=n_train
    # unlabeled_num=1527 #Hard Code
    assert labeled_num==n_train,print("Labeled_num mismatch")
    logging.info("We will use a evenly distributed (both labeled and unlabeled samples are the same across the training) training data set.")
    idxs_labeled_num = np.random.permutation(labeled_num)
    # idxs_unlabeled_num = np.random.permutation(unlabeled_num)
    batch_idxs_labeled = np.array_split(idxs_labeled_num, n_nets)  # As many splits as n_nets = number of clients
    # batch_idxs_unlabeled = np.array_split(idxs_unlabeled_num, n_nets)  # As many splits as n_nets = number of clients
    size_labeled=[len(i) for i in batch_idxs_labeled]
    # size_unlabeled = [len(i) for i in batch_idxs_unlabeled]
    net_data_idx_map = {i: batch_idxs_labeled[i] for i in range(n_nets)}

    # # non-iid data distribution
    # # TODO: Add custom non-iid distribution option - hetero-fix
    # elif partition == "hetero":
    #     # This is useful if we allow custom category lists, currently done for consistency
    #     categories = [train_categories.index(c) for c in train_categories]
    #     net_data_idx_map = non_iid_partition_with_dirichlet_distribution(train_targets, n_nets, categories, alpha,
    #                                                                      task='segmentation')

    # train_data_cls_counts = record_data_stats(train_targets, net_data_idx_map, task='segmentation')

    return train_ds,net_data_idx_map,size_labeled

def partition_data_semi(n_nets,list_label,list_bl,list_12m,root_dir, sagittal_slice=32,other_num=512):
    logging.info("********************* Partitioning data **********************")
    net_data_idx_map = None
    # list_label=os.path.join(root_dir,list_label)
    # list_bl=os.path.join(root_dir,list_bl)
    # list_12m=os.path.join(root_dir,list_12m)
    train_ds= load_sensetime_data_SEMI(list_label,list_bl,list_12m,root_dir,sagittal_slice,other_num=other_num)
    n_train = len(train_ds)  # Number of training samples

    # if partition == "homo":
    total_num = n_train
    #Since we have one label and one unlabel data, we will split such data evenly
    labeled_num=train_ds.num_labeled #Hard Code
    unlabeled_num=train_ds.num_unlabeled #Hard Code
    assert labeled_num+unlabeled_num==n_train,print("Labeled_num+Unlabeled_num not equal to total training samples,{}+{}!={}".format(labeled_num,unlabeled_num,n_train))
    logging.info("We will use a evenly distributed (both labeled and unlabeled samples are the same across the training) training data set.")
    idxs_labeled_num = np.random.permutation(labeled_num)
    idxs_unlabeled_num = np.random.permutation(unlabeled_num)
    batch_idxs_labeled = np.array_split(idxs_labeled_num, n_nets)  # As many splits as n_nets = number of clients
    batch_idxs_unlabeled = np.array_split(idxs_unlabeled_num, n_nets)  # As many splits as n_nets = number of clients
    size_labeled=[len(i) for i in batch_idxs_labeled]
    size_unlabeled = [len(i) for i in batch_idxs_unlabeled]
    net_data_idx_map = {i: np.concatenate([batch_idxs_labeled[i],batch_idxs_unlabeled[i]]) for i in range(n_nets)}

    # # non-iid data distribution
    # # TODO: Add custom non-iid distribution option - hetero-fix
    # elif partition == "hetero":
    #     # This is useful if we allow custom category lists, currently done for consistency
    #     categories = [train_categories.index(c) for c in train_categories]
    #     net_data_idx_map = non_iid_partition_with_dirichlet_distribution(train_targets, n_nets, categories, alpha,
    #                                                                      task='segmentation')

    # train_data_cls_counts = record_data_stats(train_targets, net_data_idx_map, task='segmentation')

    return train_ds,net_data_idx_map,size_labeled,size_unlabeled

def partition_data_semi_with_did(file_labeled,file_unlabeled):
    labeled_train=collections.defaultdict(list)
    unlabeled_train=collections.defaultdict(list)
    labeled_folder="/filer/tmp1/xh172/ukbb/train_extracted/lbl"
    unlabeled_folder="/filer/tmp1/xh172/ukbb/train_extracted/img"
    f = open(file_labeled,"r")
    list_labeled=f.read().splitlines()
    f.close()
    for i in list_labeled:
        s=i.split("\t")
        labeled_train[int(s[1])].append(os.path.join(labeled_folder,s[0].replace("\t","").replace(" ","")))
    f = open(file_unlabeled,"r")
    list_unlabeled=f.read().splitlines()
    f.close()
    for i in list_unlabeled:
        s=i.split("\t")
        unlabeled_train[int(s[1])].append(os.path.join(unlabeled_folder,s[0].replace("\t","").replace(" ","")))
    train_ds=[]
    for i in range(len(labeled_train)):
        train_ds.append(BB_data(labeled_train[i],unlabeled_train[i]))
    train_label=len(list_labeled)
    train_unlabel=len(list_unlabeled)
    f=open("/research/cbim/vast/xh172/UKBiobank-Processing/val_3d.txt","r")
    list_validation=f.read().splitlines()
    f.close()
    f=open("/research/cbim/vast/xh172/UKBiobank-Processing/test_3d.txt","r")
    list_testing=f.read().splitlines()
    f.close()
    for i in range(len(list_validation)):
        list_validation[i]=os.path.join("/filer/tmp1/xh172/ukbb/test_extracted/lbl",list_validation[i])
    val_ds=BB_data(list_validation,None)
    # test_ds=BB_data(list_testing,None)
    val_size=len(list_validation)
    return train_ds,val_ds,train_label,train_unlabel,val_size



# def partition_data_semi_with_did(n_nets,list_label,list_bl,list_12m,root_dir,training_did, sagittal_slice=32,other_num=512):
#     logging.info("********************* Partitioning data **********************")
#     net_data_idx_map = None
#     # list_label=os.path.join(root_dir,list_label)
#     # list_bl=os.path.join(root_dir,list_bl)
#     # list_12m=os.path.join(root_dir,list_12m)

#     list_label=os.path.join(root_dir,list_label)
#     list_bl=os.path.join(root_dir,list_bl)
#     list_12m=os.path.join(root_dir,list_12m)
#     label={0:[],1:[],2:[],3:[],4:[],5:[]}
#     label_train={0:[],1:[],2:[],3:[],4:[],5:[]}
#     label_val=[]
#     label_test=[]
#     unlabel={0:[],1:[],2:[],3:[],4:[],5:[]}
#     to_exclude_pid=[]
#     to_exclude_serial=[]
#     train_ds=[]
#     with open(list_label,"r") as f:
#         for line in f.readlines():
#             path_i=line.split()
#             # print(path_i)
#             name=os.path.basename(path_i[0])
#             names=name.split("_")
#             to_exclude_pid.append(names[0])
#             to_exclude_serial.append(names[1])
#             subject_name=names[0]+"_"+names[1]
#             p_l=os.path.join(root_dir,path_i[0])
#             # if "bl" in p_l:
#             #     p_i = os.path.join(root_dir, "img_with_mask", "bl_sag_3d_dess_mhd", subject_name + ".mha")
#             # else:
#             #     p_i = os.path.join(root_dir, "img_with_mask", "12m_sag_3d_dess_mhd", subject_name + ".mhd")
#             #print(path_i[1])
#             '''This is for OAI params Data'''
#             # p_l=os.path.join(root_dir,"cropped_new_new_labeled",subject_name+"_lbl.mha")
#             # p_i=os.path.join(root_dir,"cropped_new_new_labeled",subject_name+"_img.mha")
#             '''This is for ST style DATA'''
#             p_l=os.path.join(root_dir,"to_st_labeled",subject_name+"_lbl.mha")
#             p_i=os.path.join(root_dir,"to_st_labeled",subject_name+"_img.mha")
#             label[int(path_i[1])].append((subject_name,p_i,p_l))
#     #Spliting label into train/val/test in 8/1/1 ratio(proritize training)
#     tra=[43,28,41,26] #precalculated training ratio
#     for i in range(4):
#         label_train[i]=label[i][:tra[i]]
#         label_val+=label[i][tra[i]:tra[i]+int((len(label[i])-tra[i])/2)]
#         label_test+=label[i][int((len(label[i])-tra[i])/2):]
#     for i in [4,5]:
#         label_test+=label[i]
#     #labeled finished
#     #start unlabeled
#     #need to exclude all training images with label by personid & serial

#     '''This is for OAI data'''
#     # with open(list_bl,"r") as f:
#     #     for line in f.readlines():
#     #         terms=line.split()
#     #         if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
#     #             continue
#     #         else:
#     #             unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "cropped_new_new", terms[0] + "_" + terms[1] + ".mha")))
#     # with open(list_12m,"r") as f:
#     #     for line in f.readlines():
#     #         terms=line.split()
#     #         if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
#     #             continue
#     #         else:
#     #             unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "cropped_new_new", terms[0] + "_" + terms[1] + ".mha")))
#     '''This is for ST data'''
#     with open(list_bl,"r") as f:
#         for line in f.readlines():
#             terms=line.split()
#             if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
#                 continue
#             else:
#                 unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "to_st", terms[0] + "_" + terms[1] + ".mha")))
#     with open(list_12m,"r") as f:
#         for line in f.readlines():
#             terms=line.split()
#             if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
#                 continue
#             else:
#                 unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "to_st", terms[0] + "_" + terms[1] + ".mha")))
#     train_label=0
#     train_unlabel=0
#     for i in range(4):
#         train_ds.append(MyData_SEMI_did(label_train[i],unlabel[i],root_dir=root_dir,sagittal_num=sagittal_slice,other_num=other_num))
#         train_label+=len(label_train[i])
#         train_unlabel+=len(unlabel[i])

#     test_ds=MyData_SEMI_did(label_val,None,root_dir=root_dir,sagittal_num=sagittal_slice,other_num=other_num)
#     test_size=len(label_val)#modified to use validation for validation

#     # train_ds= load_sensetime_data_SEMI_with_did(list_label,list_bl,list_12m,root_dir,sagittal_slice)
#     # n_train = len(train_ds)  # Number of training samples

#     # # if partition == "homo":
#     # total_num = n_train
#     # #Since we have one label and one unlabel data, we will split such data evenly
#     # labeled_num=train_ds.num_labeled #Hard Code
#     # unlabeled_num=train_ds.num_unlabeled #Hard Code
#     # assert labeled_num+unlabeled_num==n_train,print("Labeled_num+Unlabeled_num not equal to total training samples,{}+{}!={}".format(labeled_num,unlabeled_num,n_train))
#     # logging.info("We will use a evenly distributed (both labeled and unlabeled samples are the same across the training) training data set.")
#     # idxs_labeled_num = np.random.permutation(labeled_num)
#     # idxs_unlabeled_num = np.random.permutation(unlabeled_num)
#     # batch_idxs_labeled = np.array_split(idxs_labeled_num, n_nets)  # As many splits as n_nets = number of clients
#     # batch_idxs_unlabeled = np.array_split(idxs_unlabeled_num, n_nets)  # As many splits as n_nets = number of clients
#     # size_labeled=[len(i) for i in batch_idxs_labeled]
#     # size_unlabeled = [len(i) for i in batch_idxs_unlabeled]
#     # net_data_idx_map = {i: np.concatenate([batch_idxs_labeled[i],batch_idxs_unlabeled[i]]) for i in range(n_nets)}

#     # # # non-iid data distribution
#     # # # TODO: Add custom non-iid distribution option - hetero-fix
#     # # elif partition == "hetero":
#     # #     # This is useful if we allow custom category lists, currently done for consistency
#     # #     categories = [train_categories.index(c) for c in train_categories]
#     # #     net_data_idx_map = non_iid_partition_with_dirichlet_distribution(train_targets, n_nets, categories, alpha,
#     # #                                                                      task='segmentation')

#     # # train_data_cls_counts = record_data_stats(train_targets, net_data_idx_map, task='segmentation')

#     return train_ds,test_ds,train_label,train_unlabel,test_size

def partition_data_semi_with_did_test(n_nets,list_label,list_bl,list_12m,root_dir,training_did, sagittal_slice=32,other_num=512):
    logging.info("********************* Partitioning data **********************")
    net_data_idx_map = None
    # list_label=os.path.join(root_dir,list_label)
    # list_bl=os.path.join(root_dir,list_bl)
    # list_12m=os.path.join(root_dir,list_12m)

    list_label=os.path.join(root_dir,list_label)
    list_bl=os.path.join(root_dir,list_bl)
    list_12m=os.path.join(root_dir,list_12m)
    label={0:[],1:[],2:[],3:[],4:[],5:[]}
    label_train={0:[],1:[],2:[],3:[],4:[],5:[]}
    label_val=[]
    label_test=[]
    unlabel={0:[],1:[],2:[],3:[],4:[],5:[]}
    to_exclude_pid=[]
    to_exclude_serial=[]
    train_ds=[]
    with open(list_label,"r") as f:
        for line in f.readlines():
            path_i=line.split()
            # print(path_i)
            name=os.path.basename(path_i[0])
            names=name.split("_")
            to_exclude_pid.append(names[0])
            to_exclude_serial.append(names[1])
            subject_name=names[0]+"_"+names[1]
            p_l=os.path.join(root_dir,path_i[0])
            if "bl" in p_l:
                p_i = os.path.join(root_dir, "img_with_mask", "bl_sag_3d_dess_mhd", subject_name + ".mha")
            else:
                p_i = os.path.join(root_dir, "img_with_mask", "12m_sag_3d_dess_mhd", subject_name + ".mhd")
            #print(path_i[1])
            label[int(path_i[1])].append((subject_name,p_i,p_l))
    #Spliting label into train/val/test in 8/1/1 ratio(proritize training)
    tra=[43,28,41,26] #precalculated training ratio
    for i in range(4):
        label_train[i]=label[i][:tra[i]]
        label_val+=label[i][tra[i]:int((len(label[i])-tra[i])/2)]
        label_test+=label[i][int((len(label[i])-tra[i])/2):]
    for i in [4,5]:
        label_test+=label[i]
    #labeled finished
    #start unlabeled
    #need to exclude all training images with label by personid & serial
    with open(list_bl,"r") as f:
        for line in f.readlines():
            terms=line.split()
            if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
                continue
            else:
                unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "bl", terms[0] + "_" + terms[1] + "_" + terms[2] + ".mha")))
    with open(list_12m,"r") as f:
        for line in f.readlines():
            terms=line.split()
            if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
                continue
            else:
                unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "12m", terms[0] + "_" + terms[1] + "_" + terms[2] + ".mha")))
    train_label=0
    train_unlabel=0
    for i in range(4):
        train_ds.append(MyData_SEMI_did(label_train[i],unlabel[i],root_dir=root_dir,sagittal_num=sagittal_slice))
        train_label+=len(label_train[i])
        train_unlabel+=len(unlabel[i])

    test_ds=MyData_SEMI_did(label_test,None,root_dir=root_dir,sagittal_num=sagittal_slice)
    test_size=len(label_test)

    # train_ds= load_sensetime_data_SEMI_with_did(list_label,list_bl,list_12m,root_dir,sagittal_slice)
    # n_train = len(train_ds)  # Number of training samples

    # # if partition == "homo":
    # total_num = n_train
    # #Since we have one label and one unlabel data, we will split such data evenly
    # labeled_num=train_ds.num_labeled #Hard Code
    # unlabeled_num=train_ds.num_unlabeled #Hard Code
    # assert labeled_num+unlabeled_num==n_train,print("Labeled_num+Unlabeled_num not equal to total training samples,{}+{}!={}".format(labeled_num,unlabeled_num,n_train))
    # logging.info("We will use a evenly distributed (both labeled and unlabeled samples are the same across the training) training data set.")
    # idxs_labeled_num = np.random.permutation(labeled_num)
    # idxs_unlabeled_num = np.random.permutation(unlabeled_num)
    # batch_idxs_labeled = np.array_split(idxs_labeled_num, n_nets)  # As many splits as n_nets = number of clients
    # batch_idxs_unlabeled = np.array_split(idxs_unlabeled_num, n_nets)  # As many splits as n_nets = number of clients
    # size_labeled=[len(i) for i in batch_idxs_labeled]
    # size_unlabeled = [len(i) for i in batch_idxs_unlabeled]
    # net_data_idx_map = {i: np.concatenate([batch_idxs_labeled[i],batch_idxs_unlabeled[i]]) for i in range(n_nets)}

    # # # non-iid data distribution
    # # # TODO: Add custom non-iid distribution option - hetero-fix
    # # elif partition == "hetero":
    # #     # This is useful if we allow custom category lists, currently done for consistency
    # #     categories = [train_categories.index(c) for c in train_categories]
    # #     net_data_idx_map = non_iid_partition_with_dirichlet_distribution(train_targets, n_nets, categories, alpha,
    # #                                                                      task='segmentation')

    # # train_data_cls_counts = record_data_stats(train_targets, net_data_idx_map, task='segmentation')

    return train_ds,test_ds,train_label,train_unlabel,test_size
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

def load_partition_data_oai(client_number,batch_size,sagittal_num,other_num=512):
    root_dir = '../../../data/OAI'
    train_ds, net_data_idx_map, size_labeled = partition_data(
        client_number,
        sagittal_slice=sagittal_num, other_num=other_num)
    train_data_num=sum([len(net_data_idx_map[r]) for r in range(client_number)])

    test = os.path.join(root_dir, 'val.txt')
    test_ds = MyData_test(test,root_dir=root_dir,sagittal_num=sagittal_num,other_num=other_num)
    train_data_global=DataLoader(dataset=train_ds, batch_size=batch_size, pin_memory=True)
    test_data_global = DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=False, drop_last=False,num_workers=8,pin_memory=True)
    test_data_num = len(test_ds)
    train_data_local_dict=dict()
    test_data_local_dict=dict()
    data_local_num_dict=dict()
    for client_idx in range(client_number):
        data_idxs = net_data_idx_map[client_idx]  # get dataId list for client generated using Dirichlet sampling
        local_data_num = len(data_idxs)  # How many samples does client have?
        local_data_labeled_num=size_labeled[client_idx]
        # local_data_unlabeled_num=size_unlabeled[client_idx]
        assert local_data_labeled_num==local_data_num, logging.info("Labeled_num+unlabeled_num not match the total number of data")
        logging.info("Total number of local images: {} with {} labeled in client ID {}".format(local_data_num, local_data_labeled_num,client_idx))

        data_local_num_dict[client_idx] = local_data_num
        train_dataset_local=Subset(train_ds,data_idxs)

        # train_loader = data.DataLoader(dataset=train_data, batch_sampler = batch_sampler,
        #                                num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
        # val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE_VAL, shuffle=False, drop_last=False,
        #                              num_workers=8, pin_memory=True)
        train_data_local = DataLoader(dataset=train_dataset_local,batch_size=batch_size, num_workers=10,pin_memory=True)
        test_data_local=DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=False, drop_last=False,num_workers=8,pin_memory=True)
        # train_data_local, test_data_local, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size, image_size,
        #                                                               data_idxs)
        logging.info(
            "Number of local train batches: {} and test batches: {} in client ID {}".format(len(train_data_local),
                                                                                            len(test_data_local),
                                                                                            client_idx))

        # Store data loaders for each client as they contain specific data
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
           train_data_local_dict, test_data_local_dict, 4 ##Hard Code Classes num
    # self.root_dir = '../../../data/sensetime'
    # file_labeled = os.path.join(self.root_dir, 'list_train.txt')


def load_partition_data_oai_semi(client_number, batch_size, batch_size_lb,sagittal_num,other_num=512):
    root_dir = '../../../data/OAI'
    list_label="train_changed.txt"
    list_bl="bl_sag_3d_dess_leftRight_processed.txt"
    list_12m="12m_sag_3d_dess_leftRight_processed.txt"
    train_ds,net_data_idx_map,size_labeled,size_unlabeled = partition_data_semi(
                                                             client_number,list_label,list_bl,list_12m,root_dir,
                                                             sagittal_slice=sagittal_num,other_num=other_num)

    train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])
    test = os.path.join(root_dir, 'val_changed.txt')
    test_ds = MyData_test(test, root_dir=root_dir, sagittal_num=sagittal_num,other_num=other_num)
    # Global train and test data
    # train_data_global, test_data_global, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size, image_size)
    # logging.info(
    #     "Number of global train batches: {} and test batches: {}".format(len(train_data_global), len(test_data_global)))

    # test_data_num = len(test_data_global)
    # train_loader = data.DataLoader(dataset=train_data, batch_sampler = batch_sampler,
    #                                num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
    # val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE_VAL, shuffle=False, drop_last=False,
    #                              num_workers=8, pin_memory=True)
    labeled_num=train_ds.num_labeled #Hard Code
    unlabeled_num=train_ds.num_unlabeled #Hard Code
    sampler = TwoStreamBatchSampler(range(labeled_num), range(labeled_num,labeled_num+unlabeled_num), batch_size,
                                    batch_size - batch_size_lb)
    def worker_init_fn(worker_id):
        random.seed(10 + worker_id)
    train_data_global=DataLoader(dataset=train_ds, batch_sampler=sampler, num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
    test_data_global = DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=False, drop_last=False,num_workers=8,pin_memory=True)
    test_data_num = len(test_ds)
    # get local dataset
    data_local_num_dict = dict()  # Number of samples for each client
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        data_idxs = net_data_idx_map[client_idx]  # get dataId list for client generated using Dirichlet sampling
        local_data_num = len(data_idxs)  # How many samples does client have?
        local_data_labeled_num=size_labeled[client_idx]
        local_data_unlabeled_num=size_unlabeled[client_idx]
        assert local_data_labeled_num+local_data_unlabeled_num==local_data_num, logging.info("Labeled_num+unlabeled_num not match the total number of data")
        logging.info("Total number of local images: {} with {} labeled and {} unlabeled in client ID {}".format(local_data_num, local_data_labeled_num,local_data_unlabeled_num,client_idx))

        data_local_num_dict[client_idx] = local_data_num
        train_dataset_local=Subset(train_ds,data_idxs)
        sampler=TwoStreamBatchSampler(range(local_data_labeled_num),range(local_data_labeled_num,local_data_unlabeled_num+local_data_labeled_num),batch_size,batch_size-batch_size_lb)
        def worker_init_fn(worker_id):
            random.seed(10+ worker_id)
        # train_loader = data.DataLoader(dataset=train_data, batch_sampler = batch_sampler,
        #                                num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
        # val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE_VAL, shuffle=False, drop_last=False,
        #                              num_workers=8, pin_memory=True)
        train_data_local = DataLoader(dataset=train_dataset_local,batch_sampler=sampler, num_workers=10,pin_memory=True,worker_init_fn=worker_init_fn)
        test_data_local=DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=False, drop_last=False,num_workers=8,pin_memory=True)
        # train_data_local, test_data_local, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size, image_size,
        #                                                               data_idxs)
        logging.info(
            "Number of local train batches: {} and test batches: {} in client ID {}".format(len(train_data_local),
                                                                                            len(test_data_local),
                                                                                            client_idx))

        # Store data loaders for each client as they contain specific data
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
           train_data_local_dict, test_data_local_dict, 4 ##Hard Code Classes num



def load_partition_data_oai_semi_with_did(client_number, batch_size, batch_size_lb,sagittal_num,other_num=512):

    root_dir = '../../../data/OAI'
    '''This is for OAI format data'''
    list_label="label_file_did.txt"
    list_bl="bl_new.txt"
    list_12m="12m_new.txt"
    '''This is for ST format data'''
    # list_label="label_file_did_ST.txt"
    # list_bl="bl_ST.txt"
    # list_12m="12m_ST.txt"

    training_did=[0,1,2,3]
    
    train_ds,test_ds,train_label,train_unlabel,test_size = partition_data_semi_with_did(client_number,list_label,list_bl,list_12m,root_dir,training_did,
                                                    sagittal_slice=sagittal_num,other_num=other_num)
    train_data_num=train_label+train_unlabel
    # train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])
    # test = os.path.join(root_dir, 'val.txt')
    # test_ds = MyData_test(test, root_dir=root_dir, sagittal_num=sagittal_num)
    # Global train and test data
    # train_data_global, test_data_global, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size, image_size)
    # logging.info(
    #     "Number of global train batches: {} and test batches: {}".format(len(train_data_global), len(test_data_global)))

    # test_data_num = len(test_data_global)
    # train_loader = data.DataLoader(dataset=train_data, batch_sampler = batch_sampler,
    #                                num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
    # val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE_VAL, shuffle=False, drop_last=False,
    #                              num_workers=8, pin_memory=True)
    # labeled_num=train_ds.num_labeled #Hard Code
    # unlabeled_num=train_ds.num_unlabeled #Hard Code
    labeled_num=train_label
    unlabeled_num=train_unlabel
    sampler = TwoStreamBatchSampler(range(labeled_num), range(labeled_num,labeled_num+unlabeled_num), batch_size,
                                    batch_size - batch_size_lb)
    def worker_init_fn(worker_id):
        random.seed(10 + worker_id)
    train_ds_global=ConcatDataset(train_ds)
    train_data_global=DataLoader(dataset=train_ds_global, batch_sampler=sampler, num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
    test_data_global = DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=False, drop_last=False,num_workers=8,pin_memory=True)
    test_data_num = len(test_ds)
    # get local dataset
    data_local_num_dict = dict()  # Number of samples for each client
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        train_dataset_local=train_ds[client_idx]
        # data_idxs = net_data_idx_map[client_idx]  # get dataId list for client generated using Dirichlet sampling
        local_data_num = train_dataset_local.num_labeled+train_dataset_local.num_unlabeled  # How many samples does client have?
        local_data_labeled_num=train_dataset_local.num_labeled
        local_data_unlabeled_num=train_dataset_local.num_unlabeled
        # assert local_data_labeled_num+local_data_unlabeled_num==local_data_num, logging.info("Labeled_num+unlabeled_num not match the total number of data")
        logging.info("Total number of local images: {} with {} labeled and {} unlabeled in client ID {}".format(local_data_num, local_data_labeled_num,local_data_unlabeled_num,client_idx))

        data_local_num_dict[client_idx] = local_data_num
        # train_dataset_local=Subset(train_ds,data_idxs)
        sampler=TwoStreamBatchSampler(range(local_data_labeled_num),range(local_data_labeled_num,local_data_unlabeled_num+local_data_labeled_num),batch_size,batch_size-batch_size_lb)
        def worker_init_fn(worker_id):
            random.seed(10+ worker_id)
        # train_loader = data.DataLoader(dataset=train_data, batch_sampler = batch_sampler,
        #                                num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
        # val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE_VAL, shuffle=False, drop_last=False,
        #                              num_workers=8, pin_memory=True)
        train_data_local = DataLoader(dataset=train_dataset_local,batch_sampler=sampler, num_workers=10,pin_memory=True,worker_init_fn=worker_init_fn)
        test_data_local=DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=False, drop_last=False,num_workers=8,pin_memory=True)
        # train_data_local, test_data_local, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size, image_size,
        #                                                               data_idxs)
        logging.info(
            "Number of local train batches: {} and test batches: {} in client ID {}".format(len(train_data_local),
                                                                                            len(test_data_local),
                                                                                            client_idx))

        # Store data loaders for each client as they contain specific data
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
           train_data_local_dict, test_data_local_dict, 4 ##Hard Code Classes num


def load_partition_data_bb_3d_with_did(client_number, batch_size, batch_size_lb,other_num=256):

    # root_dir = '../../../data/OAI'
    # '''This is for OAI format data'''
    # list_label="label_file_did.txt"
    # list_bl="bl_new.txt"
    # list_12m="12m_new.txt"
    '''This is for ST format data'''
    # list_label="label_file_did_ST.txt"
    # list_bl="bl_ST.txt"
    # list_12m="12m_ST.txt"
    list_unlabel="/research/cbim/vast/xh172/UKBiobank-Processing/unlabeled_3d_file.txt"
    list_label="/research/cbim/vast/xh172/UKBiobank-Processing/labeled_3d_file.txt"
    training_did=[0,1,2]
    
    train_ds,test_ds,train_label,train_unlabel,test_size = partition_data_semi_with_did(list_label,list_unlabel)
    train_data_num=train_label+train_unlabel
    # train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])
    # test = os.path.join(root_dir, 'val.txt')
    # test_ds = MyData_test(test, root_dir=root_dir, sagittal_num=sagittal_num)
    # Global train and test data
    # train_data_global, test_data_global, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size, image_size)
    # logging.info(
    #     "Number of global train batches: {} and test batches: {}".format(len(train_data_global), len(test_data_global)))

    # test_data_num = len(test_data_global)
    # train_loader = data.DataLoader(dataset=train_data, batch_sampler = batch_sampler,
    #                                num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
    # val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE_VAL, shuffle=False, drop_last=False,
    #                              num_workers=8, pin_memory=True)
    # labeled_num=train_ds.num_labeled #Hard Code
    # unlabeled_num=train_ds.num_unlabeled #Hard Code
    labeled_num=train_label
    unlabeled_num=train_unlabel
    sampler = TwoStreamBatchSampler(range(labeled_num), range(labeled_num,labeled_num+unlabeled_num), batch_size,
                                    batch_size - batch_size_lb)
    def worker_init_fn(worker_id):
        random.seed(10 + worker_id)
    train_ds_global=ConcatDataset(train_ds)
    train_data_global=DataLoader(dataset=train_ds_global, batch_sampler=sampler, num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
    test_data_global = DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=False, drop_last=False,num_workers=8,pin_memory=True)
    test_data_num = len(test_ds)
    # get local dataset
    data_local_num_dict = dict()  # Number of samples for each client
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        train_dataset_local=train_ds[client_idx]
        # data_idxs = net_data_idx_map[client_idx]  # get dataId list for client generated using Dirichlet sampling
        local_data_num = train_dataset_local.num_labeled+train_dataset_local.num_unlabeled  # How many samples does client have?
        local_data_labeled_num=train_dataset_local.num_labeled
        local_data_unlabeled_num=train_dataset_local.num_unlabeled
        # assert local_data_labeled_num+local_data_unlabeled_num==local_data_num, logging.info("Labeled_num+unlabeled_num not match the total number of data")
        logging.info("Total number of local images: {} with {} labeled and {} unlabeled in client ID {}".format(local_data_num, local_data_labeled_num,local_data_unlabeled_num,client_idx))

        data_local_num_dict[client_idx] = local_data_num
        # train_dataset_local=Subset(train_ds,data_idxs)
        sampler=TwoStreamBatchSampler(range(local_data_labeled_num),range(local_data_labeled_num,local_data_unlabeled_num+local_data_labeled_num),batch_size,batch_size-batch_size_lb)
        def worker_init_fn(worker_id):
            random.seed(10+ worker_id)
        # train_loader = data.DataLoader(dataset=train_data, batch_sampler = batch_sampler,
        #                                num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
        # val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE_VAL, shuffle=False, drop_last=False,
        #                              num_workers=8, pin_memory=True)
        train_data_local = DataLoader(dataset=train_dataset_local,batch_sampler=sampler, num_workers=32,pin_memory=True,worker_init_fn=worker_init_fn)
        test_data_local=DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=False, drop_last=False,num_workers=32,pin_memory=True)
        # train_data_local, test_data_local, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size, image_size,
        #                                                               data_idxs)
        logging.info(
            "Number of local train batches: {} and test batches: {} in client ID {}".format(len(train_data_local),
                                                                                            len(test_data_local),
                                                                                            client_idx))

        # Store data loaders for each client as they contain specific data
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
           train_data_local_dict, test_data_local_dict, 4 ##Hard Code Classes num


# def load_partition_data_sensetime(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, image_size):


# def get_dataloader_sensetime(data_dir, train_bs, test_bs, image_size, data_idxs):
#     train_data = MyData("../../data/sensetime")

if __name__=="__main__":
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
    train_data_local_dict, test_data_local_dict, num=load_partition_data_bb_with_did(3,4,2,other_num=256) #client_number, batch_size, batch_size_lb,sagittal_num)
    print(train_data_num)
    print(test_data_num)

    for x,y in train_data_local_dict[1]:
        print(x.shape)
    # for x,y in test_data_global:
    #     print(x.size())