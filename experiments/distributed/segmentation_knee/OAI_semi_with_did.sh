mpirun -np 5 -hostfile ./mpi_host_file python ./main_fedseg.py --model semi_OAI --dataset OAI_semi_did --client_num_in_total 4 --client_num_per_round 4 --batch_size 2 --batch_size_lb 1 --gpu_server_num 1 --gpu_num_per_server 4 --gpu_mapping_key mapping_mykonos --checkname OAI_SEMI_with_did_epoch_20_4_machine_fixed --epoch 20