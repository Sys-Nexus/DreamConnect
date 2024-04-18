log_dirs=(
    logs/test_conv_adaptor_idback_test_conv_adaptor_idback_2024-04-17/visualize/images/val
    # logs/test_conv_adaptor_test_conv_adaptor_2024-04-13/visualize/images/val 
    # logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val
    # logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_pix2pix
    # logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_dif
    # logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_magic_brush
    # logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit
)


eval_modes=(
    'id_sim'
)
eval_methods=(
    'dino'
    'clip'
)
# for eval_mode in ${eval_modes[@]}
# do
#     echo 'eval_mode: '$eval_mode
#     for eval_method in ${eval_methods[@]}
#     do
#         echo 'eval_method: '$eval_method
#         for root_dir in ${log_dirs[@]}
#         do
#             python  metrics/edit_eval.py --eval_method ${eval_method} --eval_mode ${eval_mode} --root_dir ${root_dir}
#         done
#     done
# done


eval_modes=(
    'inst_sim'
)
eval_methods=(
    'clip'
)

for eval_mode in ${eval_modes[@]}
do
    echo 'eval_mode: '$eval_mode
    for eval_method in ${eval_methods[@]}
    do
        echo 'eval_method: '$eval_method
        for root_dir in ${log_dirs[@]}
        do
            python  metrics/edit_eval.py --eval_method ${eval_method} --eval_mode ${eval_mode} --root_dir ${root_dir}
        done
    done
done
