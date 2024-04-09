jobname=${1-'magic_brush'}
gpu_id=${2-'2,'}


if [[ ${jobname} == 'magic_brush' ]]; then
    CUDA_VISIBLE_DEVICES=${gpu_id} python quanti_cmp/instruct_cmp.py
fi

