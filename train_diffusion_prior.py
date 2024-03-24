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
from third_party.fMRI_reconstruction_NSD.src.models import BrainNetwork 
from third_party.fMRI_reconstruction_NSD.src.diffusion_prior import InstructDiffusionPrior, VersatileDiffusionPriorNetwork

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

        for train_i, (_, _, _, _, _, _, _, _, file_name, text_descr) in tqdm(enumerate(train_dl)):
            train_iter = train_i + len(train_dl)*epoch
            # torch.cuda.synchronize()
            t = time.time()
            
            samples, clip_target = prepare_train_data(fp_parser, file_name, talking_head, base_sample, silent_frames_start, silent_frames_end)

            # torch.cuda.synchronize()
            data_t = time.time() -t
            t = time.time()
            
            # import pdb; pdb.set_trace();
            with torch.cuda.amp.autocast():
                optimizer.zero_grad()

                with torch.no_grad():
                    voxel = clip_text_embdder(text_descr)
                    voxel = torch.mean(voxel, dim=1) #### TODO, we use mean 77 tokens to obtain semantic info
                voxel = voxel.requires_grad_(True)
                
                clip_voxels, clip_voxels_proj = diffusion_prior.module.voxel2clip(voxel) if distributed else diffusion_prior.voxel2clip(voxel)
                # import pdb; pdb.set_trace()
                
                if hidden:
                    clip_voxels = clip_voxels.view(len(voxel),-1,clip_size)
                
                if prior:
                    loss_prior, aligned_clip_voxels = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)
                    aligned_clip_voxels /= diffusion_prior.module.image_embed_scale if distributed else diffusion_prior.image_embed_scale
                else:
                    aligned_clip_voxels = clip_voxels

                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                # import pdb; pdb.set_trace()

                if epoch < int(mixup_pct * num_epochs):
                    loss_nce = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006, 
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    loss_nce = soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)

                if prior and v2c:
                    loss_nce_sum += loss_nce.item()
                    loss_prior_sum += loss_prior.item()
                    loss = loss_nce + (prior_mult * loss_prior)
                elif v2c:
                    loss_nce_sum += loss_nce.item()
                    loss = loss_nce
                elif prior:
                    loss_prior_sum += loss_prior.item()
                    loss = prior_mult * loss_prior
                check_loss(loss)
                # utils.check_loss(loss)
                
                # accelerator.backward(loss)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                sims_base += nn.functional.cosine_similarity(clip_target_norm,clip_voxels_norm).mean().item()

                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_target_norm)).to(torch.device("cuda")) 
                fwd_percent_correct += topk(batchwise_cosine_similarity(clip_voxels_norm,clip_target_norm), labels, k=1)
                bwd_percent_correct += topk(batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1)

                if lr_scheduler_type is not None:
                    lr_scheduler.step()
            forward_t = time.time() - t

            if args.is_tensorboard_log:
                loss_dict['train_sims_base'] = sims_base / (train_i + 1)
                loss_dict['train_fwd_percent_correct'] = fwd_percent_correct / (train_i + 1)
                loss_dict['train_bwd_percent_correct'] = bwd_percent_correct / (train_i + 1)
                loss_dict['train_loss_nce'] = loss_nce_sum / (train_i + 1)
                loss_dict['train_loss_prior'] = loss_prior_sum / (train_i + 1)
                loss_dict['train_loss'] = np.mean(losses[-(train_i+1):])
                losses_dict.update(loss_dict)
                write_loss_meters(meters, losses_dict)

                if train_iter % args.log_loss_steps == 0:
                    flush_meters(meters, train_iter)
            # print('data: ', data_t, ' forward: ', forward_t, ' loss:', loss.item())
            # import pdb; pdb.set_trace()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--clip_size', type=int, default=128)
    parser.add_argument('--jobname', type=str, default='latent_diffusion_image')
    parser.add_argument('--resume_from_ckpt', type=bool, default=True)
    parser.add_argument('--is_tensorboard_log', type=bool, default=True)
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--ckpt_path', type=str, default='dummy')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_projector', type=bool, default=True)
    parser.add_argument("--epoch", type=int, default=0, help='number of epochs')

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

    optimizer = None
    # prior model
    guidance_scale = 3.5
    timesteps = 100
    depth = 6
    dim_head = 64
    clip_size = args.clip_size
    out_dim = clip_size
    heads = clip_size//16
    # import pdb; pdb.set_trace();
    prior_network = VersatileDiffusionPriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = 1,
            learned_query_mode="pos_emb"
        )
    # import pdb; pdb.set_trace();
    prior_network = prior_network.to(torch.device("cuda"))

    # clip text to emotion latent model
    clip_size = args.clip_size
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
