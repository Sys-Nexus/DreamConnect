mode=${1-'image'}
which_gpu=${2-'1,'}


if [[ $mode == 'image' ]]; then
    ckpt_path='train_logs/latent_diffusion_image_fp32_resume/last.pth'
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'image' --is_test True \
                --batch_size 1 \
                --ckpt_path ${ckpt_path}
fi


if [[ $mode == 'text' ]]; then
    ckpt_path='train_logs/latent_diffusion_text_fp32_resume2/last.pth' # eval
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'text' --is_test True \
                --batch_size 1 \
                --ckpt_path ${ckpt_path}
fi
