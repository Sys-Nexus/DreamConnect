
# jobname=${1:-'versatile_dualstream_instruct_pre_post_w_x0'}
jobname=${1:-'versatile_dualstream_instruct_pre_post_w_x0_prospect'}
filter_mode=${2:-'tune_sideconv'} #no_filter

gpus=${3:-'1,'}
master_port=${4:-'27699'}
isTrain=${5:-1}
mode=${6:-'single'}
vis=${7:-1}

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

prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs/apple2024-02-28T22-30-10_apple/checkpoints/embeddings.pt'
CUDA_VISIBLE_DEVICES=${gpus} ${run_cmd} main.py --name ${exp_name} \
          --base configs/${jobname}.yaml \
          --train \
          --gpus ${gpus} \
          --resume '' \
          --num_nodes ${num_nodes} \
          --filter_mode ${filter_mode} \
          --no-test True \
          --vis ${vis} \
          --isTrain ${isTrain} \
          --cfg_text 7.5 \
          --prospect_ckpt_path ${prospect_ckpt_path}