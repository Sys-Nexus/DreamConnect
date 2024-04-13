# python  metrics/edit_eval.py --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_pix2pix
# python  metrics/edit_eval.py --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_dif
# python  metrics/edit_eval.py --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_magic_brush
# python  metrics/edit_eval.py --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit


# python  metrics/edit_eval.py --eval_method dino --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_pix2pix
# python  metrics/edit_eval.py --eval_method dino --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_dif
# python  metrics/edit_eval.py --eval_method dino --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_magic_brush
# python  metrics/edit_eval.py --eval_method dino --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit


eval_mode='inst_sim'
python  metrics/edit_eval.py --eval_method dino --eval_mode ${eval_mode} --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_pix2pix
# python  metrics/edit_eval.py --eval_method dino --eval_mode ${eval_mode} --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_dif
# python  metrics/edit_eval.py --eval_method dino --eval_mode ${eval_mode} --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_magic_brush
# python  metrics/edit_eval.py --eval_method dino --eval_mode ${eval_mode} --root_dir logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit
