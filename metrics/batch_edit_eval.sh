log_dirs=(
    logs/test_conv_adaptor_test_conv_adaptor_2024-04-13/visualize/images/val 
    # logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val
    # logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_pix2pix
    # logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_dif
    # logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_magic_brush
    # logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit
)
eval_modes=(
    'id_sim'
    'dino'
    'inst_sim'
)

# eval_mode='inst_sim'
# eval_mode='id_sim'
# eval_method='dino'

eval_method='clip'
for eval_mode in ${eval_modes[@]}
do
    echo 'eval_mode: '$eval_mode
    for root_dir in ${log_dirs[@]}
    do
        python  metrics/edit_eval.py --eval_method ${eval_method} --eval_mode ${eval_mode} --root_dir ${root_dir}
    done
done
