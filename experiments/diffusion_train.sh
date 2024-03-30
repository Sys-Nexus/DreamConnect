mode=${1-'image_w_nce'}
which_gpu=${2-'2,'}

if [[ $mode == 'image' ]]; then
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'image' \
                --max_epoch 480 \
                --ckpt_path 'dummy' \
                --jobname 'latent_diffusion_image'
fi

if [[ $mode == 'image_w_nce' ]]; then
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'image' \
                --max_epoch 480 \
                --ckpt_path 'dummy' \
                --jobname 'latent_diffusion_image_w_nce'
fi

if [[ $mode == 'text' ]]; then
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'text' \
                --batch_size 128 \
                --ckpt_path '/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_text/last.pth' \
                --max_epoch 480 \
                --jobname 'latent_diffusion_text'
fi
