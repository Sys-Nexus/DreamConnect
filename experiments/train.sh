
# jobname=${1:-'versatile_dualstream_instruct_pre_post_w_x0'}
jobname=${1:-'versatile_dualstream_instruct_pre_post_w_x0_prospect'}
filter_mode=${2:-'tune_sideconv'} #no_filter

gpus=${3:-'1,'}
master_port=${4:-'27699'}
isTrain=${5:-1}
cfg_text=${6:-7.5}
cfg_text_edit=${6:-20.0}
mode=${7:-'single'}
vis=${8:-1}

num_nodes=1
if [[ ${mode} == 'single' ]]; then
  unset WORLD_SIZE
  unset NODE_RANK
  export MASTER_PORT=${master_port}
  export RANK=0
  export WORLD_SIZE=1
  num_nodes=1
  run_cmd='torchrun --master_port '${master_port}
fi

export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

current_date=$(date +"%Y-%m-%d")
exp_name=${jobname}"_"${current_date}

# prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs/apple2024-02-28T22-30-10_apple/checkpoints/embeddings.pt'
# prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs/panda2024-02-29T05-29-09_panda/checkpoints/embeddings.pt'
# prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs_backup0305/panda2024-02-29T05-29-09_panda/checkpoints/embeddings.pt'
prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs_backup0305/kanagawa2024-02-29T06-01-27_kanagawa/checkpoints/embeddings.pt'
layout_path='/data/yashengsun/Proj/MMEdit/ProSpect/images4prospect/layout_collections/baozi/baozi.jpg'
# resume_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/checkpoints/bohan_recon/'
resume_path=''

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${gpus} ${run_cmd} main.py --name ${exp_name} \
          --base configs/${jobname}.yaml \
          --train \
          --gpus ${gpus} \
          --resume '' \
          --num_nodes ${num_nodes} \
          --filter_mode ${filter_mode} \
          --no-test True \
          --vis ${vis} \
          --isTrain ${isTrain} \
          --cfg_text ${cfg_text} \
          --prospect_ckpt_path ${prospect_ckpt_path} \
          --cfg_text_edit ${cfg_text_edit} #\
          # --layout_path ${layout_path} #\
          # --is_inst_gen True
          # --is_inst_edit True \
