jobname=${1-'magic_brush'}
gpu_id=${2-'2,'}


# if [[ ${jobname} == 'magic_brush' ]]; then
#     CUDA_VISIBLE_DEVICES=${gpu_id} python quanti_cmp/instruct_cmp.py
# fi


# if [[ ${jobname} == 'inst_dif' ]]; then
#     CUDA_VISIBLE_DEVICES=${gpu_id} python quanti_cmp/instruct_cmp.py
# fi


# if [[ ${jobname} == 'inst_pix2pix' ]]; then
#     CUDA_VISIBLE_DEVICES=${gpu_id} python quanti_cmp/instruct_cmp.py
# fi

CUDA_VISIBLE_DEVICES=${gpu_id} python quanti_cmp/instruct_cmp.py --method ${jobname}
