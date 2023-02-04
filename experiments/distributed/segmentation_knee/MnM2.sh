mpirun -np 4 -hostfile ./mpi_host_mykonos python ./main_fedseg.py --model cardiac_semi_distil --load_model True --model_path "/research/cbim/vast/xh172/FedCV/experiments/distributed/segmentation_knee/run/cardiac_biobank/cardiac_semi/biobank_3machine/experiment_10/checkpoint.pth.tar" --dataset cardiac_MnM --client_num_in_total 3 --client_num_per_round 3 --batch_size 32 --batch_size_lb 16 --gpu_server_num 1 --gpu_num_per_server 3 --gpu_mapping_key mykonos_5_6_7 --checkname MnM2_our_method_e30_c30_3cli_b32_u16_new --epoch 30  --net_channel 1 --comm_round 30