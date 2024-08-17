
# jobname=${1:-'test_conv_adaptor'}
# jobname=${1:-'test_zero_adaptor_idback'} # test_res_value_inject_idback_css15
# jobname=${1:-'test_conv_adaptor_mixing1_text'}

jobname=${1:-'test_res_value_inject_idback_css15'}
filter_mode=${2:-'tune_sideconv'} #do not use no_filter, which will fail the memory


# cfg_text_edit=${6:-20.0}
# cfg_text_edit=${6:-1.5}

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

# prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs/apple2024-02-28T22-30-10_apple/checkpoints/embeddings.pt'
# prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs/panda2024-02-29T05-29-09_panda/checkpoints/embeddings.pt'
# prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs_backup0305/panda2024-02-29T05-29-09_panda/checkpoints/embeddings.pt'
# prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs_backup0305/kanagawa2024-02-29T06-01-27_kanagawa/checkpoints/embeddings.pt'
# layout_path='/data/yashengsun/Proj/MMEdit/ProSpect/images4prospect/layout_collections/baozi/baozi.jpg'
# resume_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/checkpoints/bohan_recon/'
# resume_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/logs/wo_prospect_ctx_adaptor_bs4_wo_prospect_ctx_adaptor_bs4_2024-03-13/checkpoints/ckpt_epoch_5'

# ${HOME}/.local/bin/wandb login 56d149bd571b8312fbca5e3802d7859909ea00c1

# whitelist_path for mask enhance
# whitelist_path='configs/test/mask_enhance_list.txt'  # essentially a whitelist for us to screen the key 

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
    --cfg_text_edit ${cfg_text_edit} #\
    # --prospect_ckpt_path ${prospect_ckpt_path} \
    # --whitelist_path ${whitelist_path}
    # --layout_path ${layout_path} #\
    # --is_inst_gen True
    # --is_inst_edit True \
