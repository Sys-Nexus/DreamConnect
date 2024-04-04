mode=${1-'image'}
which_gpu=${2-'1,'}

if [[ $mode == 'image' ]]; then
    # ckpt_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_image_from_scratch/ep_75.pth'
    ckpt_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_image_fp32/last.pth'
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'image' --is_test True \
                --batch_size 1 \
                --ckpt_path ${ckpt_path}
                # --ckpt_path '/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_image_from_scratch/ep_75.pth'
                # --ckpt_path '/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_image_backup0326/last.pth'
fi

if [[ $mode == 'text' ]]; then
    ckpt_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_text_fp32/last.pth'
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'text' --is_test True \
                --batch_size 1 \
                --ckpt_path ${ckpt_path}
                # --ckpt_path '/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_text_backup0326/last.pth'
fi
