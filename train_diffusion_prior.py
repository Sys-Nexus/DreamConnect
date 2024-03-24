from types import SimpleNamespace
import sys
import os
import yaml
from easydict import EasyDict
import argparse

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(proj_root,"stable_diffusion"))
from ldm.util import instantiate_from_config

sys.path.append('third_party/versatile_diffusion')
from lib.model_zoo.vd import VDCLIP


def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def trainer(args, train_dl, val_dl, diffusion_prior, vd_clip, optimizer, distributed=False):
    lr_scheduler_type = 'cycle'
    num_train = len(train_dl)
    max_lr, num_epochs, local_rank, clip_size = args.max_lr, args.max_epoch, args.local_rank, args.clip_size
    total_steps=int(num_epochs*(num_train))*5

    if lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=total_steps,
            last_epoch=-1
        )
    elif lr_scheduler_type == 'cycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/num_epochs
        )
    
    outdir = os.path.abspath(f'train_logs/{args.jobname}')
    
    epoch = args.epoch
    if args.resume_from_ckpt:
        if os.path.exists(args.ckpt_path):
            epoch = resume_ckpt(args.ckpt_path, optimizer, lr_scheduler, diffusion_prior)
        else:
            print('{} does not exist.'.format(args.ckpt_path))
            import pdb; pdb.set_trace();

    if args.is_tensorboard_log:
        tensorboard_dir = os.path.join(outdir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        set_summary_writer(tensorboard_dir)
        losses_dict, meters, loss_dict = {}, {}, {}

    print(f"starting with epoch {epoch} / {num_epochs}")
    progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))

    hidden, prior, v2c = True, True, True

    mixup_pct = 0.
    soft_loss_temps = cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
    if hidden:
        prior_mult = 30
    else:
        prior_mult = .03
    losses, val_losses, lrs = [], [], []
    nce_losses, val_nce_losses = [], []
    sim_losses, val_sim_losses = [], []
    best_val_loss = 1e9

    if not args.is_test:
        save_at_end, ckpt_saving = False, True
        for epoch in progress_bar:
            diffusion_prior.train()

            sims_base = 0.
            val_sims_base = 0.
            fwd_percent_correct = 0.
            bwd_percent_correct = 0.
            val_fwd_percent_correct = 0.
            val_bwd_percent_correct = 0.
            loss_nce_sum = 0.
            loss_prior_sum = 0.
            val_loss_nce_sum = 0.
            val_loss_prior_sum = 0.



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--clip_size', type=int, default=768)
    parser.add_argument('--jobname', type=str, default='latent_diffusion_image')
    parser.add_argument('--resume_from_ckpt', type=bool, default=True)
    parser.add_argument('--is_tensorboard_log', type=bool, default=True)
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--ckpt_path', type=str, default='dummy')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--clip_size', type=int, default=768)
    parser.add_argument('--use_projector', type=bool, default=True)
    
    args = parser.parse_args()

    clip_cfg = {'symbol': 'clip',
                'args': {},
                'name': 'clip_frozen',
                'type': 'clip_frozen'}

    clip_cfg = EasyDict(clip_cfg)
    vd_clip = VDCLIP(clip_cfg)

    dataset_cfg_str = """
    train:
      target: third_party.StableDiffusionReconstruction.codes.utils.nsd_creater.NIPS23NSDDataset
      params:
        nsd_root: '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd'
        resolution: 320
        split: 'train'
        is_reconstruct_mode: False
        url: 'nsd_data_dir/train_subj01_{0..17}.tar'
        reconstruct_prob: 0.05

    validation:
      target: third_party.StableDiffusionReconstruction.codes.utils.nsd_creater.NIPS23NSDDataset
      params:
        nsd_root: '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd'
        resolution: 320
        split: 'test'
        is_reconstruct_mode: False
        url: 'nsd_data_dir/test_subj01_{0..1}.tar'
        reconstruct_prob: 0.05
    """
    dataset_cfg = EasyDict(yaml.safe_load(dataset_cfg_str))
    # import pdb; pdb.set_trace();
    train_dataset = instantiate_from_config(dataset_cfg['train'])
    val_dataset = instantiate_from_config(dataset_cfg['validation'])

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    diffusion_prior = None
    vd_clip = None
    optimizer = None
    prior_network = None

    # clip text to emotion latent model
    clip_size = args.clip_size
    out_dim = clip_size * 257
    voxel2clip_kwargs = dict(in_dim=768,out_dim=clip_size,clip_size=clip_size,use_projector=args.use_projector)
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)

    # use dalle interface to include prior model and clip text-to-emotion models
    timesteps = 100
    diffusion_prior = InstructDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip,)
    assert torch.cuda.is_available()
    diffusion_prior = diffusion_prior.to(torch.device("cuda"))

    ## optimizer
    max_lr = args.max_lr
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {'params': [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

    trainer(args, train_dl, val_dl, diffusion_prior, vd_clip, optimizer, distributed=False)



if __name__ == '__main__':
    main()
