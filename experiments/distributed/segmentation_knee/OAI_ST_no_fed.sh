mpirun -np 5 -hostfile ./mpi_host_hermes python ./main_fedseg.py --model semi_OAI --dataset OAI_semi_did --client_num_in_total 4 --client_num_per_round 4 --batch_size 8 --batch_size_lb 4 --gpu_server_num 1 --gpu_num_per_server 4 --gpu_mapping_key hermes_0_1_2_3 --checkname OAI_ST_reso_cropped_Aug_netchannel4_bs8_local_only --epoch 30 --image_size_other 352 --image_size_sagittal 32 --net_channel 4 --comm_round 30 --aggregate_net none