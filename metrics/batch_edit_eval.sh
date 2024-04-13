log_dirs=(
    logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val
    logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_pix2pix
    logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_dif
    logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_magic_brush
    logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit
)

# python  metrics/edit_eval.py --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_pix2pix
# python  metrics/edit_eval.py --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_dif
# python  metrics/edit_eval.py --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_magic_brush
# python  metrics/edit_eval.py --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit


# python  metrics/edit_eval.py --eval_method dino --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_pix2pix
# python  metrics/edit_eval.py --eval_method dino --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_dif
# python  metrics/edit_eval.py --eval_method dino --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_magic_brush
# python  metrics/edit_eval.py --eval_method dino --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit


eval_mode='inst_sim'
eval_mode='id_sim'
# eval_method='dino'
eval_method='clip'
root_dir=${log_dirs[0]}
python  metrics/edit_eval.py --eval_method ${eval_method} --eval_mode ${eval_mode} --root_dir ${root_dir}
