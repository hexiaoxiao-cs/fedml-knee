mpirun -np 5 -hostfile ./mpi_host_hermes python ./main_fedseg.py --model semi_OAI --load_model False --model_path "/research/cbim/vast/xh172/FedCV/experiments/distributed/segmentation_knee/run/OAI_semi_did/semi_OAI/OAI_ST_reso_cropped_July_netchannel4_bs8/model_best.pth.tar" --dataset sensetime_oai_semi --client_num_in_total 4 --client_num_per_round 4 --batch_size 8 --batch_size_lb 4 --gpu_server_num 1 --gpu_num_per_server 4 --gpu_mapping_key hermes_4_5_6_7 --checkname ST_st_format_no_fed --epoch 30  --net_channel 4 --comm_round 30 --is_new_lbl True --lr 0.01  