import argparse
import logging
import os
import random
import socket
import sys
import datetime

import numpy as np
import psutil
import setproctitle
import torch
import wandb
from torchinfo import summary
# add the FedML root directory to the python path
# from model.segmentation.Unet_3D import Unet_3D

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML")))

from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

from FedML.fedml_api.distributed.fedseg.utils import count_parameters

#from data_preprocessing.coco.segmentation.data_loader.py import load_partition_data_distributed_coco_segmentation, load_partition_data_coco_segmentation
from data_preprocessing.pascal_voc_augmented.data_loader import load_partition_data_distributed_pascal_voc, \
    load_partition_data_pascal_voc
from data_preprocessing.cityscapes.data_loader import load_partition_data_distributed_cityscapes, \
    load_partition_data_cityscapes
from data_preprocessing.sensetime.data_loader import load_partition_data_sensetime, load_partition_data_sensetime_semi,load_partition_data_sensetime_semi_new
from data_preprocessing.cardiac_biobank.data_loader import load_partition_data_bb_with_did
from data_preprocessing.cardiac_biobank_3d.data_loader import load_partition_data_bb_3d_with_did
from data_preprocessing.cardiac_MnM.data_loader import load_partition_data_MnM_with_did
from data_preprocessing.cardiac_MnM_3D.data_loader import load_partition_data_MnM_3D_with_did
from model.segmentation.deeplabV3_plus import DeepLabV3_plus
# from model.segmentation.unet import UNet
from model.segmentation.Vnet_3D import VNet
from model.segmentation.Unet_2d import UNet
from data_preprocessing.OAI.data_loader import  *


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

    parser.add_argument('--model', type=str, default='semi', metavar='N',required=True,choices=['Vnet_3D',"Vnet_OAI","semi","semi_OAI","semi_OAI_distil","SSFL","YD","cardiac_semi","cardiac_semi_distil","cardiac_semi_3D"],
                        help='neural network used in training')
    parser.add_argument('--slow',type=bool,default=False)
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
                        choices=['sensetime', 'sensetime_semi','OAI',"OAI_semi","sensetime_oai_semi","OAI_semi_did","cardiac_biobank","cardiac_MnM","cardiac_biobank_3d","cardiac_MnM_3D"],
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='../../../data/sensetime',
                        help='data directory (default = ../../../data/sensetime)')
 
    parser.add_argument('--checkname', type=str, default='Vnet3D-sensetime-hetero', help='set the checkpoint name')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=4, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--batch_size_lb', type=int,default=1,metavar='N',help='input batch size of labeled data for training (default 1)')

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

    parser.add_argument('--comm_round', type=int, default=30,
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

    parser.add_argument('--image_size_sagittal', type=int, default=32,
                        help='Specify the size of the image (along sagittal direction)')
    parser.add_argument('--image_size_other',type=int,default=512,help='Specify the size of the image (other than sagittal direction)')
    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    parser.add_argument('--net_channel', type=int, default=2)
    parser.add_argument('--alpha',type=float, default=0.1)
    parser.add_argument('--aggregate_net',type=str,default="student",choices=["both","teacher","student","none"])
    parser.add_argument('--both_pretrained',type=bool,default=False)
    parser.add_argument('--is_new_lbl',type=bool,default=False)
    args = parser.parse_args()

    return args


def load_data(process_id, args, dataset_name):
    data_loader = None
    if dataset_name == "sensetime_semi":
       data_loader=load_partition_data_sensetime_semi
       train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
       train_data_local_dict, test_data_local_dict, class_num = data_loader(
           args.client_num_in_total, args.batch_size, args.batch_size_lb, args.image_size_sagittal)

    elif dataset_name == "sensetime":
        data_loader=load_partition_data_sensetime
        train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
        train_data_local_dict, test_data_local_dict, class_num = data_loader(
            args.client_num_in_total, args.batch_size,args.image_size)
        # client_number, batch_size, batch_size_lb, sagittal_num

    elif dataset_name=="OAI":
        data_loader=load_partition_data_oai
        train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
        train_data_local_dict, test_data_local_dict, class_num = data_loader(
            args.client_num_in_total, args.batch_size,args.image_size)
    elif dataset_name=="OAI_semi":
        data_loader=load_partition_data_oai_semi
        train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
        train_data_local_dict, test_data_local_dict, class_num = data_loader(
            args.client_num_in_total, args.batch_size,args.batch_size_lb,args.image_size_sagittal)
    elif dataset_name=="sensetime_oai_semi" and args.is_new_lbl==False:
        logging.info("True")
        data_loader=load_partition_data_sensetime_semi
        train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
        train_data_local_dict, test_data_local_dict, class_num = data_loader(
            args.client_num_in_total, args.batch_size,args.batch_size_lb, args.image_size_sagittal,isOAI=True)
    elif dataset_name=="sensetime_oai_semi" and args.is_new_lbl==True:
        logging.info("NEW LBL NEW LBL!!!!!!!!!!!!!!!!!!!!!!!")
        data_loader=load_partition_data_sensetime_semi_new
        train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
        train_data_local_dict, test_data_local_dict, class_num = data_loader(
            args.client_num_in_total, args.batch_size,args.batch_size_lb, args.image_size_sagittal,isOAI=True)
    elif dataset_name=="OAI_semi_did":
        logging.info("OAI_WITH_DID")
        data_loader=load_partition_data_oai_semi_with_did
        train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
        train_data_local_dict, test_data_local_dict, class_num = data_loader(
            args.client_num_in_total, args.batch_size,args.batch_size_lb, args.image_size_sagittal,other_num=args.image_size_other)
    elif dataset_name=="cardiac_biobank":
        logging.info("Biobank")
        data_loader=load_partition_data_bb_with_did
        train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
        train_data_local_dict, test_data_local_dict, class_num = data_loader(
            args.client_num_in_total, args.batch_size,args.batch_size_lb, other_num=args.image_size_other)
    elif dataset_name=="cardiac_biobank_3d":
        logging.info("Biobank_3d")
        data_loader=load_partition_data_bb_3d_with_did
        train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
        train_data_local_dict, test_data_local_dict, class_num = data_loader(
            args.client_num_in_total, args.batch_size,args.batch_size_lb, other_num=args.image_size_other)
    elif dataset_name=="cardiac_MnM":
        logging.info("MnM")
        data_loader=load_partition_data_MnM_with_did
        train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
        train_data_local_dict, test_data_local_dict, class_num = data_loader(
            args.client_num_in_total, args.batch_size,args.batch_size_lb, other_num=args.image_size_other)
    elif dataset_name=="cardiac_MnM_3D":
        logging.info("MnM_3D")
        data_loader=load_partition_data_MnM_3D_with_did
        train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
        train_data_local_dict, test_data_local_dict, class_num = data_loader(
            args.client_num_in_total, args.batch_size,args.batch_size_lb, other_num=args.image_size_other)
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict,
    train_data_local_dict, test_data_local_dict, class_num]

    return dataset


def create_model(args, model_name):
    # if model_name == "deeplabV3_plus":
    #     model = DeepLabV3_plus(backbone=args.backbone,
    #                       image_size=img_size,
    #                       n_classes=output_dim,
    #                       output_stride=args.outstride,
    #                       pretrained=args.backbone_pretrained,
    #                       freeze_bn=args.freeze_bn,
    #                       sync_bn=args.sync_bn)
    #
    #     if args.backbone_freezed:
    #         logging.info('Freezing Backbone')
    #         for param in model.feature_extractor.parameters():
    #             param.requires_grad = False
    #     else:
    #         logging.info('Finetuning Backbone')
    #
    #     num_params = count_parameters(model)
    #     logging.info("DeepLabV3_plus Model Size : {}".format(num_params))
    #


    #Adapt to OAI need to reduce out_class
    #TODO:Change Sensetime Data to 3 Classes
    if model_name == "Vnet_3D":
        model = VNet(args.net_channel,
                     out_class=4)
        #batch_size = 1
        #logging.info(summary(model, input_size=(batch_size, 1, 32, 512, 512)))
        return model
        # num_params = count_parameters(model)
        # logging.info("Unet Model Size : {}".format(num_params))
    elif model_name == "semi":
        model= VNet(args.net_channel, out_class = 9, norm='bn', elu='elu', dropout=1)
        ema_net = VNet(args.net_channel, out_class = 9, norm='bn', elu='elu', dropout=1)
        return model, ema_net
    elif model_name == "Vnet_OAI":
        model = VNet(model_channel=args.net_channel, out_class=4)
        #logging.info(summary(model, input_size=(2, 1, 32, 512, 512)))
        #print(summary(model,input_size=(2,1,32,512,512)))
        return model
    elif model_name=="SSFL":
        model = VNet(model_channel=args.net_channel,out_class=4)
        return model
    elif model_name=="YD":
        model =  VNet(model_channel=args.net_channel,out_class=4)
        return model
    elif model_name=="semi_OAI":
        model=VNet(args.net_channel, out_class=4)
        ema_net=VNet(args.net_channel,out_class=4)
        for param in ema_net.parameters():
            param.detach_()
        logging.info("detached")
        #logging.info(summary(model, input_size=(2, 1, args.image_size_sagittal, args.image_size_other, args.image_size_other)))
        return model,ema_net
    elif model_name=="semi_OAI_distil":
        model=VNet(args.net_channel, out_class=4)
        ema_net=VNet(args.net_channel,out_class=4)
        for param in ema_net.parameters():
            param.detach_()
        #logging.info(summary(model, input_size=(2, 1, args.image_size_sagittal, args.image_size_other, args.image_size_other)))
        return model,ema_net
    elif model_name=="cardiac_semi" or model_name=="cardiac_semi_distil":
        model=UNet(args.net_channel,4)
        ema_net=UNet(args.net_channel,4)
        for param in ema_net.parameters():
            param.detach_()
        #logging.info(summary(model, input_size=(2, 1, args.image_size_sagittal, args.image_size_other, args.image_size_other)))
        # logging.info(summary(model, input_size=(4, 1, 256, 256)))
        return model,ema_net
    else:
        raise ('Not Implemented Error')




def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine, gpu_server_num):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:" + str(gpu_server_num) if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = (client_index % gpu_num_per_machine)
        process_gpu_dict[client_index] = gpu_index + gpu_server_num
    
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info('GPU process allocation {0}'.format(process_gpu_dict))
    logging.info('GPU device available {0}'.format(device))
    return device


if __name__ == "__main__":

    # initialize distributed computing (MPI)
    from FedML.fedml_api.distributed.fedseg.FedSegAPI import FedML_init
    comm, process_id, worker_number = FedML_init()

    # customize the log format
    logging.basicConfig(filename='info.log',
                        level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    now = datetime.datetime.now()
    time_start = now.strftime("%Y-%m-%d %H:%M:%S")    
    logging.info("Executing Image Segmentation at time: {0}".format(time_start))

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    if "semi" in args.model:
        from FedML.fedml_api.distributed.fedsegsemi.FedSegAPI import FedML_FedSeg_distributed
        logging.info("Utilizing fedsegsemi")
    else:
        from FedML.fedml_api.distributed.fedseg.FedSegAPI import FedML_FedSeg_distributed
    logging.info('Given arguments {0}'.format(args))

    # customize the process name
    str_process_name = args.process_name + str(process_id)
    setproctitle.setproctitle(str_process_name)

    hostname = socket.gethostname()
    logging.info("Host and process details")
    logging.info("process ID: {0}, host name: {1}, process ID: {2}, process name: {3}, worker number: {4}".format(process_id,hostname,os.getpid(), psutil.Process(os.getpid()), worker_number))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            project = "fedcv-segmentation",
            name = args.process_name + str(args.partition_method) + "r" + str(args.comm_round) + "-e" + str(
                args.epochs) + "-lr" + str(
                args.lr),
            config = args,
            settings=wandb.Settings(start_method="fork")
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # GPU arrangement: Please customize this function according your own topology.
    # The GPU server list is configured at "mpi_host_file".
    # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
    # The 4 machines will be assigned as follows:
    # machine 1: worker0, worker4, worker8;
    # machine 2: worker1, worker5;
    # machine 3: worker2, worker6;
    # machine 4: worker3, worker7;
    # Therefore, we can see that workers are assigned according to the order of machine list.
   
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key)
    #device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server, args.gpu_server_num)

    # load data
    dataset = load_data(process_id, args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict,
     train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model=None
    ema_net=None
    if args.model=="semi":
        model,ema_net = create_model(args, model_name=args.model)
        for param in ema_net.parameters():
            param.detach_()
    elif args.model=="Vnet_3D" or args.model=="SSFL" or args.model=="YD":
        model = create_model(args, model_name=args.model)
    elif args.model=="Vnet_OAI":
        model=create_model(args,model_name=args.model)
    elif args.model=="semi_OAI" or args.model=="semi_OAI_distil":
        model,ema_net = create_model(args,model_name=args.model)
    elif args.model=="cardiac_semi" or args.model=="cardiac_semi_distil":
        model,ema_net=create_model(args,model_name=args.model)
    # logging.info(summary(model,input_size=(2,1,160,384,384)))
    # logging.info(summary(ema_net,input_size=(2,1,160,384,384)))
    if args.load_model:
        try:
            checkpoint = torch.load(args.model_path)
            if "semi" in args.model:
                ema_net.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
        except:
            raise("Failed to load pre-trained model")
    # if "distil" not in args.model and args.load_model:
    #     model.load_state_dict(checkpoint['state_dict']) #both teacher and student use the same pretrained network
    if args.both_pretrained==True:
        print("TEACHER STUDENT IDENTICAL")
        model.load_state_dict(checkpoint['state_dict'])#both teacher and student use the same pretrained network
    # define my own trainer
    if "distil" in args.model and "cardiac" not in args.model:
        from training.semi_distil_segmentation_trainer import SegmentationTrainer
        model_trainer = SegmentationTrainer(model,ema_net, args)
    elif "cardiac" in args.model and "distil" in args.model:
        from training.semi_distil_segmentation_2d_trainer import SegmentationTrainer
        model_trainer = SegmentationTrainer(model,ema_net, args)
    elif "SSFL" in args.model:
        from training.SSFL_segmentation_trainer import SegmentationTrainer
        model_trainer=SegmentationTrainer(model,args)
    elif "YD" in args.model:
        from training.YD_segmentation_trainer import SegmentationTrainer
        model_trainer=SegmentationTrainer(model,args)
    elif "cardiac" in args.model:
        from training.semi_segmentation_2d_trainer import SegmentationTrainer
        logging.info("Using 2D Semi Trainer")
        model_trainer=SegmentationTrainer(model,ema_net,args)
    elif "semi" in args.model:
        from training.semi_segmentation_trainer import SegmentationTrainer
        model_trainer = SegmentationTrainer(model,ema_net, args)
    else:
        from training.Vnet_segmentation_trainer import SegmentationTrainer
        model_trainer = SegmentationTrainer(model, args)
    if args.slow==True:
        from training.semi_segmentation_trainer_slow_update import SegmentationTrainer
        model_trainer=SegmentationTrainer(model,ema_net,args)
    logging.info("Calling FedML_FedSeg_distributed")
    if "semi" in args.model:
        FedML_FedSeg_distributed(process_id, worker_number, device, comm, model,ema_net, train_data_num, data_local_num_dict,
                                train_data_local_dict, test_data_local_dict, args, model_trainer)
    else:
        FedML_FedSeg_distributed(process_id, worker_number, device, comm, model, train_data_num, data_local_num_dict,
                                train_data_local_dict, test_data_local_dict, args, model_trainer)
