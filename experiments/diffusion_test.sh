mode=${1-'image'}
which_gpu=${2-'1,'}

if [[ $mode == 'image' ]]; then
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'image' --is_test True \
                --batch_size 1 \
                --ckpt_path '/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_image_pct033_backup_img_emb_scale27/ep_75.pth'
fi

if [[ $mode == 'text' ]]; then
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'text' --is_test True \
                --batch_size 1 \
                --ckpt_path '/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_text_backup0326/last.pth'
fi
