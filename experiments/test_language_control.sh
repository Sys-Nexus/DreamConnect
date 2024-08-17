jobname=${1:-'test_res_value_inject_idback_css15'}
filter_mode=${2:-'tune_sideconv'} #do not use no_filter, which will fail the memory


gpus=${3:-'0,'}
master_port=${4:-'27699'}
isTrain=${5:-0}
cfg_text=${6:-7.5}
cfg_text_edit=${6:-7.5}
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


CUDA_VISIBLE_DEVICES=${gpus} ${run_cmd} main.py --name ${exp_name} \
    --base configs/test/${jobname}.yaml \
    --train \
    --gpus ${gpus} \
    --resume '' \
    --num_nodes ${num_nodes} \
    --filter_mode ${filter_mode} \
    --no-test True \
    --vis ${vis} \
    --isTrain ${isTrain} \
    --cfg_text ${cfg_text} \
    --cfg_text_edit ${cfg_text_edit}
