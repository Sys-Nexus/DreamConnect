mode=${1-'image'}
which_gpu=${2-'2,'}


if [[ $mode == 'image' ]]; then
    # ckpt_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_image_backup0326/last.pth'
    ckpt_path='dummy'
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'image' \
                --max_epoch 240 \
                --ckpt_path ${ckpt_path} \
                --jobname 'latent_diffusion_image_from_scratch'
fi


if [[ $mode == 'text' ]]; then
    # ckpt_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_text_pct/ep_175.pth'
    # ckpt_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_text_pct_all/last.pth'
    # ckpt_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_text_backup0326/last.pth'
    ckpt_path='dummy'
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'text' \
                --batch_size 128 \
                --ckpt_path ${ckpt_path} \
                --max_epoch 480 \
                --jobname 'latent_diffusion_text_from_scratch'
fi
