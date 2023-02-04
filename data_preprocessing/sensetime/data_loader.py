from .dataset import *
import numpy as np
from torch.utils.data import *
import random
import itertools
import logging
# # for local devices
# def get_dataloader_test(data_dir, train_bs, test_bs, image_size, data_idxs_train=None, data_idxs_test=None):
#     return get_dataloader_cityscapes_test(data_dir, train_bs, test_bs, image_size, data_idxs_train, data_idxs_test)
def load_sensetime_data(sagittal_slice=32):
    # transform_train, transform_test = _data_transforms_cityscapes(image_size)
    root_dir = '../../../data/sensetime'
    train = os.path.join(root_dir, 'list_train.txt')
    train_ds = MyData(train,sagittal_num=sagittal_slice)


    # train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    # test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_ds

def load_sensetime_data_SEMI(sagittal_slice=32,isOAI=False):
    # transform_train, transform_test = _data_transforms_cityscapes(image_size)

    train_ds = MyData_SEMI(sagittal_num=sagittal_slice,isOAI=isOAI)
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
def partition_data(n_nets, sagittal_slice=32):
    logging.info("********************* Partitioning data **********************")
    train_ds= load_sensetime_data(sagittal_slice)
    n_train = len(train_ds)  # Number of training samples

    # if partition == "homo":
    total_num = n_train
    #Since we have one label and one unlabel data, we will split such data evenly
    labeled_num=68 #Hard Code
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

def partition_data_semi(n_nets, sagittal_slice=32,isOAI=False):
    logging.info("********************* Partitioning data **********************")
    net_data_idx_map = None
    train_ds= load_sensetime_data_SEMI(sagittal_slice,isOAI=isOAI)
    n_train = len(train_ds)  # Number of training samples

    # if partition == "homo":
    total_num = n_train
    #Since we have one label and one unlabel data, we will split such data evenly
    labeled_num=68 #Hard Code
    unlabeled_num=1527 #Hard Code
    # assert labeled_num+unlabeled_num==n_train,print("Labeled_num+Unlabeled_num not equal to total training samples,{}+{}!={}".format(labeled_num,unlabeled_num,n_train))
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

def partition_data_semi_lbl(n_nets, sagittal_slice=32,isOAI=False):
    logging.info("********************* Partitioning data **********************")
    net_data_idx_map = None
    train_ds= MyData_SEMI_new(sagittal_num=sagittal_slice,isOAI=isOAI)
    n_train = len(train_ds)  # Number of training samples

    # if partition == "homo":
    total_num = n_train
    #Since we have one label and one unlabel data, we will split such data evenly
    labeled_num=68 #Hard Code
    unlabeled_num=4000 #Hard Code
    # assert labeled_num+unlabeled_num==n_train,print("Labeled_num+Unlabeled_num not equal to total training samples,{}+{}!={}".format(labeled_num,unlabeled_num,n_train))
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

def load_partition_data_sensetime(client_number,batch_size,sagittal_num):
    train_ds, net_data_idx_map, size_labeled = partition_data(
        client_number,
        sagittal_slice=sagittal_num)
    train_data_num=sum([len(net_data_idx_map[r]) for r in range(client_number)])
    root_dir = '../../../data/sensetime'
    test = os.path.join(root_dir, 'list_val.txt')
    test_ds = MyData_test(test,sagittal_num=sagittal_num)
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
           train_data_local_dict, test_data_local_dict, 9 ##Hard Code Classes num
    # self.root_dir = '../../../data/sensetime'
    # file_labeled = os.path.join(self.root_dir, 'list_train.txt')


def load_partition_data_sensetime_semi(client_number, batch_size, batch_size_lb,sagittal_num,isOAI=False):
    train_ds,net_data_idx_map,size_labeled,size_unlabeled = partition_data_semi(
                                                             client_number,
                                                             sagittal_slice=sagittal_num,isOAI=isOAI)

    train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])
    test_ds=MyData_val_semi("../../../data/sensetime/list_val_new.txt",sagittal_num,isOAI=isOAI)
    # Global train and test data
    # train_data_global, test_data_global, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size, image_size)
    # logging.info(
    #     "Number of global train batches: {} and test batches: {}".format(len(train_data_global), len(test_data_global)))

    # test_data_num = len(test_data_global)
    # train_loader = data.DataLoader(dataset=train_data, batch_sampler = batch_sampler,
    #                                num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
    # val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE_VAL, shuffle=False, drop_last=False,
    #                              num_workers=8, pin_memory=True)
    labeled_num=68 #Hard Code
    unlabeled_num=1527 #Hard Code
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
    if isOAI==True:
        return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
               train_data_local_dict, test_data_local_dict, 3 ##Hard Code Classes num
    else:
        return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
               train_data_local_dict, test_data_local_dict, 7  ##Hard Code Classes num


def load_partition_data_sensetime_semi_new(client_number, batch_size, batch_size_lb,sagittal_num,isOAI=False):
    train_ds,net_data_idx_map,size_labeled,size_unlabeled = partition_data_semi_lbl(
                                                             client_number,
                                                             sagittal_slice=sagittal_num,isOAI=isOAI)

    train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])
    test_ds=MyData_val_semi("../../../data/sensetime/list_val_ST_lbl.txt",sagittal_num,isOAI=isOAI)
    # Global train and test data
    # train_data_global, test_data_global, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size, image_size)
    # logging.info(
    #     "Number of global train batches: {} and test batches: {}".format(len(train_data_global), len(test_data_global)))

    # test_data_num = len(test_data_global)
    # train_loader = data.DataLoader(dataset=train_data, batch_sampler = batch_sampler,
    #                                num_workers=10, pin_memory=True,worker_init_fn=worker_init_fn)
    # val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE_VAL, shuffle=False, drop_last=False,
    #                              num_workers=8, pin_memory=True)
    labeled_num=68 #Hard Code
    unlabeled_num=1527 #Hard Code
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
    if isOAI==True:
        return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
               train_data_local_dict, test_data_local_dict, 3 ##Hard Code Classes num
    else:
        return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
               train_data_local_dict, test_data_local_dict, 7  ##Hard Code Classes num

# def load_partition_data_sensetime(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, image_size):


# def get_dataloader_sensetime(data_dir, train_bs, test_bs, image_size, data_idxs):
#     train_data = MyData("../../data/sensetime")

if __name__=="__main__":
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
    train_data_local_dict, test_data_local_dict, num=load_partition_data_sensetime(5,4,32) #client_number, batch_size, batch_size_lb,sagittal_num)
    print(train_data_num)
    print(test_data_num)

    for x,y in enumerate(train_data_local_dict[1]):
        print(x.size())
    # for x,y in test_data_global:
    #     print(x.size())