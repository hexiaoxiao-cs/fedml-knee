mpirun -np 5 -hostfile ./mpi_host_santorini python ./main_fedseg.py --model semi_OAI --load_model True --model_path "/research/cbim/vast/xh172/FedCV/experiments/distributed/segmentation_knee/run/OAI_semi_did/semi_OAI/OAI_full_reso_cropped_bs_8/model_best.pth.tar" --dataset sensetime_oai_semi --client_num_in_total 4 --client_num_per_round 4 --batch_size 4 --batch_size_lb 2 --gpu_server_num 1 --gpu_num_per_server 4 --gpu_mapping_key hermes_4_5_6_7 --checkname ST_full_reso_cropped_e30_c10_aggregate_after_comm --epoch 30  --net_channel 2 --comm_round 10