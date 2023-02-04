mpirun -np 4 -hostfile ./mpi_host_mykonos python ./main_fedseg.py --model cardiac_semi --dataset cardiac_biobank --client_num_in_total 3 --client_num_per_round 3 --batch_size 32 --batch_size_lb 16 --gpu_server_num 1 --gpu_num_per_server 3 --gpu_mapping_key mykonos_1_2_3 --checkname biobank_3machine --epoch 20 --net_channel 1