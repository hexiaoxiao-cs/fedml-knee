mpirun -np 5 -hostfile ./mpi_host_hermes python ./main_fedseg.py --model semi_OAI --load_model True --model_path "/research/cbim/vast/xh172/FedCV/experiments/distributed/segmentation_knee/run/OAI_semi_did/semi_OAI/OAI_ST_reso_cropped_July_netchannel4_bs8/model_best.pth.tar" --dataset sensetime_oai_semi --client_num_in_total 4 --client_num_per_round 4 --batch_size 8 --batch_size_lb 4 --gpu_server_num 1 --gpu_num_per_server 4 --gpu_mapping_key hermes_0_1_2_3 --checkname ST_st_format_both_pretrained_senario_1 --epoch 30  --net_channel 4 --comm_round 30 --is_new_lbl True --both_pretrained True