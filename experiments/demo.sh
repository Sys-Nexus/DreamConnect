input=$1
edit=$2
gpu_id=${3:-2}

CUDA_VISIBLE_DEVICES=${gpu_id}, python fmri_edit_cli.py --resolution 512 --steps 100 \
        --config configs/fmri_instruct_diffusion.yaml \
        --ckpt checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt \
        --cfg-text 5.0 --cfg-image 1.25 --seed 93151 \
        --input ${input} \
        --edit ${edit}
