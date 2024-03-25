mode=${1-'image'}
which_gpu=${1-1}

if [[ 'mode' == 'image' ]]; then
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'image'
fi

if [[ 'mode' == 'text' ]]; then
    CUDA_VISIBLE_DEVICES=${which_gpu} python train_diffusion_prior.py --mode 'text' --ckpt_path 'dummy'
fi
