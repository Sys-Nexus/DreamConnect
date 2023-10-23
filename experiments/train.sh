# jobname=${1:-'fmri_reconstruct_codi'}
# jobname=${1:-'instru_tune_sideconv'}

jobname=${1:-'fmri_instruct_encoder_context'} # lower learning rate compared to instru_tune_sideconv
filter_mode=${2:-'no_filter'}

# jobname=${1:-'instru_tune_sideconv_sdunlock'} # lower learning rate compared to instru_tune_sideconv
# filter_mode=${2:-'tune_sideconv_sdunlock'} #no_filter

# jobname=${1:-'recon_tune_sideconv'}
# filter_mode=${2:-'tune_sideconv'} #no_filter
gpus=${3:-'3,'}
master_port=${4:-'27198'}
mode=${5:-'single'}
vis=${6:-0}


num_nodes=1
if [[ ${mode} == 'single' ]]; then
  unset WORLD_SIZE
  unset NODE_RANK
  export MASTER_PORT=${master_port}
  export RANK=0
  export WORLD_SIZE=1
  export MASTER_ADDR=${master_port}
  num_nodes=1
  run_cmd='torchrun --master_port '${master_port}
fi

export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

cmd_suffix=''

current_date=$(date +"%Y-%m-%d")
exp_name=${jobname}"_"${current_date}

CUDA_VISIBLE_DEVICES=${gpus} ${run_cmd} main.py --name ${exp_name} \
          --base configs/${jobname}.yaml \
          --train \
          --gpus ${gpus} \
          --resume '' \
          --num_nodes $num_nodes \
          --filter_mode ${filter_mode} \
          --no-test True ${cmd_suffix} \
          --vis ${vis}