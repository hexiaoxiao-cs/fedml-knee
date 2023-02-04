import logging, time
from re import A
import sys, os

import numpy as np
import torch
import wandb
from .semi_segmentation_utils import *
# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from FedML.fedml_api.distributed.fedseg.utils import SegmentationLosses, Evaluator, LR_Scheduler, EvaluationMetricsKeeper
import torch.nn.functional as F

class SegmentationTrainer(ModelTrainer):
    def __init__(self, model,ema_net, args=None):
        super(SegmentationTrainer, self).__init__(model, args)
        self.ema_net=ema_net
        self.current_comm_round=0

    def get_model_params(self):
        logging.info('Initializing end-to-end model')
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info('Updating Global model')
        self.model.load_state_dict(model_parameters)
    def get_ema_params(self):
        logging.info('getting ema params')
        return self.ema_net.cpu().state_dict()
    def set_ema_params(self,model_parameters):
        logging.info('Updating EMA Net Parameters')
        self.ema_net.load_state_dict(model_parameters)
    def train(self, train_data, device):   
        model = self.model
        ema_net=self.ema_net
        args = self.args
        model.to(device)
        model.train()
        ema_net.to(device)
        ema_net.train()
        iter_num=0
        criterion = nn.CrossEntropyLoss()
        self.weight_dice = torch.FloatTensor([1, 1, 1, 1]).to(device)
        scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_data))

        # if args.client_optimizer == "sgd":
        #
        #     if args.backbone_freezed:
        #         optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr * 10,
        #                                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        #     else:
        #         train_params = [{'params': self.model.get_1x_lr_params(), 'lr': args.lr},
        #                         {'params': self.model.get_10x_lr_params(), 'lr': args.lr * 10}]
        #
        #         optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        # else:
        #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                                       lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=10 ** -7)
        consistency_criterion = softmax_mse_loss
        epoch_loss = []
        max_iterations = 54 * 225  # 54*500  #
        a=256
        b=256
        for epoch in range(args.epochs):
            t = time.time()
            batch_loss = []
            logging.info('Trainer_ID: {0}, Epoch: {1}'.format(self.id, epoch))
            model.train()
            ema_net.train()
            for (batch_idx, batch) in enumerate(train_data):
                x, labels = batch
                x, labels = x.to(device), labels.to(device)
                unlabeled_volume_batch = x[args.batch_size_lb:]
                noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
                ema_inputs = unlabeled_volume_batch + noise
                outputs = model(x.float())
                with torch.no_grad():
                    ema_output = ema_net(ema_inputs.float().to(device))
                ema_model=ema_net.enc_out

                #logging.info("ema_output"+str(ema_output.shape))
                # scheduler(optimizer, batch_idx, epoch)
                T = 8
                volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)  # 保证输入的 batchsize 固定
                #logging.info("volume_batch_r"+str(volume_batch_r.shape))
                stride = volume_batch_r.shape[0] // 2
                preds = torch.zeros([stride * T,4,a,b]).to(device)
                #logging.info("preds size"+str(preds.shape))
                for j in range(T // 2):
                    ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                        preds[2 * stride * j:2 * stride * (j + 1)] = ema_net(ema_inputs.float().to(device))
                preds = F.softmax(preds, dim=1)
                #logging.info("preds size"+str(preds.shape))

                preds = preds.reshape(T, stride, 4,a,b)
                #logging.info("preds size"+str(preds.shape))

                preds = torch.mean(preds, dim=0)  # (batch, 2, 112,112,80)
                #logging.info("preds size"+str(preds.shape))

                uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1,
                                               keepdim=True)  # (batch, 1, 112,112,80)

                pred_ = torch.nn.functional.softmax(outputs, dim=1)
                #logging.info(pred_.shape)
                loss_dice = WeightedDiceLoss(pred_[:args.batch_size_lb], labels[:args.batch_size_lb].long().to(device),
                                             self.weight_dice)
                loss_ce = criterion(outputs[:args.batch_size_lb], labels[:args.batch_size_lb].long().to(device))
                supervised_loss=(loss_dice+loss_ce)*0.5

                consistency_weight = args.alpha * sigmoid_rampup(epoch, args.epochs)#get_current_consistency_weight(iter_num // 150)
                consistency_dist = consistency_criterion(outputs[args.batch_size_lb:], ema_output)  # (batch, 2, 112,112,80)
                #logging.info("consistency_weight"+str(consistency_weight))
                threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
                mask = (uncertainty < threshold).float()
                consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
                #logging.info("consistency_dist"+str(consistency_dist))
                consistency_loss = consistency_weight * consistency_dist
                enc_model=model.enc_out[args.batch_size_lb:]
                enc_softmax=F.log_softmax(enc_model,dim=1)
                ema_softmax=F.log_softmax(ema_model,dim=1)
                kl_div = F.kl_div(enc_softmax,ema_softmax,reduction='mean',log_target=True)
                #print(kl_div.shape)
                
                loss = supervised_loss + consistency_loss+kl_div

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler(optimizer, batch_idx, epoch)

                iter_num=iter_num+1
                # print('loss_iter', loss_dice, loss_ce)
                # supervised_loss = (loss_dice + loss_ce) * 0.5
                # optimizer.zero_grad()
                # log_probs = model(x)
                # loss = criterion(log_probs, labels).to(device)                ema_model=ema_net.out256[args.batch_size_lb:]

                # if (batch_idx % 100 == 0):
                logging.info('Trainer_ID: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.id, batch_idx, loss, (time.time()-t)/60))
                # if epoch > 0 and epoch % 75 == 0:
                #     lr_ = self.args.lr * 0.1 ** (epoch // 75)
                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] = lr_
            self.current_comm_round=self.current_comm_round+1
            if self.current_comm_round>6:
                update_ema_variables_less_freq(model, ema_net, 0.99, epoch,args.epochs)
            # iter_num += 1
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Trainer_ID: {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.id,
                                                                epoch, sum(epoch_loss) / len(epoch_loss)))


    def test(self, test_data, device):
        logging.info("Evaluating on Trainer ID: {}".format(self.id))
        model = self.model

        args = self.args
        # evaluator = Evaluator(model.n_classes)

        model.eval()
        model.to(device)

        t = time.time()
        # evaluator.reset()
        test_acc = test_acc_class = test_mIoU = test_FWIoU = test_loss = test_total = 0.
        # criterion = SegmentationLosses().build_loss(mode=args.loss_type)
        criterion = nn.CrossEntropyLoss()
        self.weight_dice = torch.FloatTensor([1, 1, 1,1]).to(device)
        with torch.no_grad():
            score_ = 4 * [0]
            loss_val_=0
            for (batch_idx, batch) in enumerate(test_data):
                x, target = batch
                x, target = x.to(device), target.to(device)
                output = model(x.float())
                pred_ = torch.nn.functional.softmax(output, dim=1)
                # pred_=pred_.long().cpu()
                # target=target.cpu()
                loss_dice = WeightedDiceLoss(pred_.to(device), target.long().to(device), self.weight_dice)
                # output=output.long().cpu()
                loss_ce = criterion(output.to(device), target.long().to(device))
                loss_val = (loss_dice + loss_ce) * 0.5
                test_loss += loss_val.item()
                test_total += target.size(0)
                # pred = output.cpu().numpy()
                # target = target.cpu().numpy()
                # pred = np.argmax(pred, axis = 1)
                # evaluator.add_batch(target, pred)
                pred = torch.argmax(torch.nn.functional.softmax(output.to(device), dim=1),
                                    dim=1)  # torch.nn.functional.softmax(
                score=dice_score(pred,target.to(device))
                score_[0]+=(score[0]+score[1]+score[2]+score[3])/4.
                for ii in range(1, 4):
                    score_[ii] += score[ii - 1]
                if (batch_idx % 10 == 0):
                    logging.info('Trainer_ID: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.id, batch_idx,loss_val, (time.time()-t)/60))
            loss_val_ /= len(test_data)
            # loss_val_saver.append(loss_val_)
                # time_end_test_per_batch = time.time()
                # logging.info("time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))
                # logging.info("Client = {0} Batch = {1}".format(self.client_index, batch_idx)
                                                                            
        # Evaluation Metrics (Averaged over number of samples)
        # test_acc = evaluator.Pixel_Accuracy()
        # test_acc_class = evaluator.Pixel_Accuracy_Class()
        # test_mIoU = evaluator.Mean_Intersection_over_Union()
        # test_FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        # test_loss = test_loss / test_total

        logging.info("Trainer_ID={0}, test_acc={1}, test_acc_class={2}, test_mIoU={3}, test_FWIoU={4}, test_loss={5}".format(
            self.id, test_loss, test_loss, test_loss, test_loss, test_loss))
        
        eval_metrics = EvaluationMetricsKeeper(test_loss, test_loss, test_loss, test_loss, test_loss)
        return eval_metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None):
        pass