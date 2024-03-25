mode=${1-'image'}
which_gpu=${2-'1,'}

if [[ $mode == 'image' ]]; then
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'image'
fi

if [[ $mode == 'text' ]]; then
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'text' \
                --batch_size 64 \
                --ckpt_path '/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_text/last.pth'
fi
