mode=${1-'image'}
which_gpu=${2-'2,'}

if [[ $mode == 'image' ]]; then
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'image' \
                --max_epoch 240 \
                --ckpt_path 'dummy' \
                --jobname 'latent_diffusion_image_pct'
fi

if [[ $mode == 'text' ]]; then
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'text' \
                --batch_size 128 \
                --ckpt_path 'dummy' \
                --max_epoch 480 \
                --jobname 'latent_diffusion_text_pct'
fi
