
# jobname=${1:-'prospect_test_conv_adaptor_mixing0_image'}
# jobname=${1:-'prospect_test_conv_adaptor_mixing1_text'}
jobname=${1:-'prospect_test_conv_adaptor'}
filter_mode=${2:-'tune_sideconv'} #do not use no_filter, which will fail the memory

gpus=${3:-'0,'}
master_port=${4:-'27699'}
isTrain=${5:-0}
cfg_text=${6:-7.5}
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

prospect_ckpt_paths=(
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/starryNew2024-03-04T23-34-17_starryNew/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/window_grille2024-03-04T23-28-37_window_grille/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/circuit2024-03-04T23-23-03_circuit/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/blackstrokes2024-03-04T23-17-24_blackstrokes/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/udnie2024-03-04T23-11-50_udnie/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/cloud2024-03-04T23-06-12_cloud/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/stars2024-03-04T23-00-36_stars/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/leaves2024-03-04T22-54-59_leaves/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/yellow_square2024-03-04T22-49-23_yellow_square/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/occean2024-03-04T22-43-46_occean/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/rain_princess2024-03-04T22-38-07_rain_princess/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/wave2024-03-04T22-32-31_wave/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/202004282208292024-03-04T22-26-58_20200428220829/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/colorful_cubes2024-03-04T22-21-24_colorful_cubes/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/bricks2024-03-04T22-15-50_bricks/checkpoints/embeddings.pt
  /data/yashengsun/Proj/MMEdit/ProSpect/logs/feathers2024-03-04T22-10-18_feathers/checkpoints/embeddings.pt
  '/data/yashengsun/Proj/MMEdit/ProSpect/logs_backup0305/kanagawa2024-02-29T06-01-27_kanagawa/checkpoints/embeddings.pt'
)

style_names=(
  'bricks'
  'leaves'
  'cloud'
  'colorful_cubes'
  'window_grille'
  'yellow_square'
  'wave'
  'udnie'
  'rain_princess'
  'feathers'
  'blackstrokes'
  '20200428220829'
  'stars'
  'starryNew'
  'occean'
  'circuit'
  'kanagawa'
)

cfg_text_edits=(
  7.0
  7.0
  7.0
  7.0
  7.0
  7.0
  7.0
  7.0
  7.0
  7.0
  7.0
  7.0
  7.0
  7.0
  7.0
  7.0
)


layout_path='/data/yashengsun/Proj/MMEdit/ProSpect/images4prospect/layout_collections/baozi/baozi.jpg'
resume_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/logs/wo_prospect_ctx_adaptor_bs4_wo_prospect_ctx_adaptor_bs4_2024-03-13/checkpoints/ckpt_epoch_5'

# prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs/apple2024-02-28T22-30-10_apple/checkpoints/embeddings.pt'
# prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs/panda2024-02-29T05-29-09_panda/checkpoints/embeddings.pt'
# prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs_backup0305/panda2024-02-29T05-29-09_panda/checkpoints/embeddings.pt'
# prospect_ckpt_path='/data/yashengsun/Proj/MMEdit/ProSpect/logs_backup0305/kanagawa2024-02-29T06-01-27_kanagawa/checkpoints/embeddings.pt'
# resume_path='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/checkpoints/bohan_recon/'
# ${HOME}/.local/bin/wandb login 56d149bd571b8312fbca5e3802d7859909ea00c1

length=${#style_names[@]}

for ((i = 0; i < length; i++)); do
  style_name=${style_names[i]}
  prospect_ckpt_path=${prospect_ckpt_paths[i]}
  cfg_text_edit=${cfg_text_edits[i]}

  exp_name=${jobname}"_"${current_date}"_"${style_name}

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
      --prospect_ckpt_path ${prospect_ckpt_path} \
      --cfg_text_edit ${cfg_text_edit} \
      --is_inst_gen True \
      --is_inst_edit True
done


# --layout_path ${layout_path} #\
## set both is_inst_gen and is_inst_edit to True for correct inference
## is_inst_gen: control the generation process of versatile diffusion
## is_inst_edit: control the editing process of instruction diffusion
