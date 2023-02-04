mpirun -np 5 -hostfile ./mpi_host_hermes python ./main_fedseg.py --model semi_OAI --load_model False --dataset sensetime_oai_semi --client_num_in_total 4 --client_num_per_round 4 --batch_size 4 --batch_size_lb 2 --gpu_server_num 1 --gpu_num_per_server 4 --gpu_mapping_key hermes_0_1_2_3 --checkname ST_cropped_both_net_initialized_e30_c30 --epoch 30 --net_channel 2 --comm_round 30