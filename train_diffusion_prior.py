from types import SimpleNamespace
import sys
import os
import yaml
from easydict import EasyDict
import argparse
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import nlpaug.augmenter.word as naw

proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(proj_root,"stable_diffusion"))
from ldm.util import instantiate_from_config

sys.path.append('third_party/versatile_diffusion')
from lib.model_zoo.vd import VDCLIP
from third_party.fMRI_reconstruction_NSD.src.models import BrainNetwork 
from third_party.fMRI_reconstruction_NSD.src.diffusion_prior import InstructDiffusionPrior, VersatileDiffusionPriorNetwork

import kornia
from kornia.augmentation.container import AugmentationSequential
img_augment = AugmentationSequential(
    kornia.augmentation.RandomResizedCrop((224,224), (0.9,1), p=0.3),
    kornia.augmentation.Resize((224, 224)),
    # kornia.augmentation.RandomHorizontalFlip(p=0.5),
    # kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
    # kornia.augmentation.RandomGrayscale(p=0.3),
    # data_keys=["input"],
)

def write_summary(name, summary, step, hist=False):
    """Utility function for write summary to log_writer.
    """
    global LOG_WRITER
    lw = LOG_WRITER
    if lw is None:
        raise Exception("Log writer not set.")
    if hist:
        lw.add_histogram(name, summary, step)
    else:
        lw.add_scalar(name, summary, step)


class Meter(object):
    """Meter is to keep track of statistics along steps.
    Meters write values for purpose like printing average values.
    Meters can be flushed to log files (i.e. TensorBoard for now)
    regularly.

    Args:
        name (str): the name of meter
    """

    def __init__(self, name):
        self.name = name
        self.values = []

    def reset(self):
        r"""Reset the meter values"""
        self.values = []

    def write(self, value):
        r"""Record the value"""
        self.values.append(value)

    def flush(self, step):
        r"""Write the value in the tensorboard.

        Args:
            step (int): Epoch or iteration number.
        """
        if not all(math.isfinite(x) for x in self.values):
            print("meter {} contained a nan or inf.".format(self.name))
        filtered_values = list(filter(lambda x: math.isfinite(x), self.values))
        if float(len(filtered_values)) != 0:
            value = float(sum(filtered_values)) / float(len(filtered_values))
            write_summary(self.name, value, step)
        self.reset()

    def write_image(self, img_grid, step):
        r"""Write the value in the tensorboard.

        Args:
            img_grid:
            step (int): Epoch or iteration number.
        """
        global LOG_WRITER
        lw = LOG_WRITER
        if lw is None:
            raise Exception("Log writer not set.")
        lw.add_image("Visualizations", img_grid, step)


def write_loss_meters(meters, losses_dict):
    r"""Write all loss values to tensorboard."""
    for loss_name, loss in losses_dict.items():
        full_loss_name = 'diffusion' + '/' + loss_name
        if full_loss_name not in meters.keys():
            # Create a new meter if it doesn't exist.
            meters[full_loss_name] = Meter(full_loss_name)
        # meters[full_loss_name].write(loss.item())
        meters[full_loss_name].write(loss)

def flush_meters(meters, current_iteration):
    r"""Flush all meters using the current iteration."""
    for meter in meters.values():
        meter.flush(current_iteration)

# if resume_from_ckpt:
def resume_ckpt(ckpt_path, optimizer, lr_scheduler, diffusion_prior):
    print("\n---resuming from last.pth ckpt---\n")
    # try:
    # checkpoint = torch.load(outdir+'/last.pth', map_location='cpu')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # except:
    #     print('last.pth failed... trying last_backup.pth')
    #     checkpoint = torch.load(outdir+'/last_backup.pth', map_location='cpu')
    epoch = checkpoint['epoch']
    print("Epoch",epoch)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    diffusion_prior.load_state_dict(checkpoint['model_state_dict'])
    return epoch

def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def save_ckpt(tag, outdir, epoch, diffusion_prior, optimizer, lr_scheduler, losses, val_losses, lrs):
    ckpt_path = outdir+f'/{tag}.pth'
    os.makedirs(outdir, exist_ok=True)
    print(f'saving {ckpt_path}',flush=True)
    # try:
    torch.save({
        'epoch': epoch,
        'model_state_dict': diffusion_prior.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'train_losses': losses,
        'val_losses': val_losses,
        'lrs': lrs,
        }, ckpt_path)

@torch.no_grad()
def prepare_train_data(batch_dict, vd_clip, use_image_aug=False, use_text_aug=True, mode='image'):
    voxel = batch_dict['fmri'].cuda()
    vd_clip.clip.fp16 = False
    vd_clip.clip.cuda()
        
    if mode == 'image':
        image = batch_dict['image'].cuda()
        image = F.interpolate(image, size=(224,224))
        # import pdb; pdb.set_trace();
        if use_image_aug:
            image = img_augment(image)
        clip_target = vd_clip.clip_encode_vision(image)
    elif mode == 'text':
        cap = batch_dict['cap']#.cuda()
        if use_text_aug:
            aug_p = 0.1
            aug = naw.SynonymAug(aug_p=aug_p)
            # augmented_text = aug.augment(cap[0])
            cap = [aug.augment(text_i)[0] for text_i in cap]
        clip_target = vd_clip.clip_encode_text(cap)
    else:
        raise ValueError
    # import pdb; pdb.set_trace();
    # clip_target = clip_target.cuda()
    return voxel, clip_target


def mixco(voxels, beta=0.15, s_thresh=0.5):
    perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select


def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss


def set_summary_writer(log_dir):
    r"""Set summary writer

    Args:
        log_dir (str): Log directory.
    """
    global LOG_DIR, LOG_WRITER
    LOG_DIR = log_dir
    LOG_WRITER = SummaryWriter(log_dir=log_dir)

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def has_nan(tensor):
    return torch.isnan(tensor).any()

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
            # import pdb; pdb.set_trace();

    if args.is_tensorboard_log:
        tensorboard_dir = os.path.join(outdir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        set_summary_writer(tensorboard_dir)
        losses_dict, meters, loss_dict = {}, {}, {}

    print(f"starting with epoch {epoch} / {num_epochs}")
    progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))

    hidden, prior, v2c = True, True, True

    # mixup_pct = 0.33 if args.mode == 'image' else 0.66
    mixup_pct = 1.0
    soft_loss_temps = cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
    # import pdb; pdb.set_trace();
    nce_mult = 1.0
    if hidden:
        prior_mult = 30
        # nce_mult = 0.1
        # if args.mode == 'text': nce_mult = 0.01
        # if args.mode == 'image': 
        #     nce_mult = 0.00001
        #     prior_mult = 3
    else:
        prior_mult = .03
        # nce_mult = 1.0
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

            # import pdb; pdb.set_trace();
            for train_i, batch_dict in tqdm(enumerate(train_dl)):
                train_iter = train_i + len(train_dl)*epoch
                t = time.time()
                # import pdb; pdb.set_trace();
                voxel, clip_target = prepare_train_data(batch_dict, vd_clip, use_image_aug=False, mode=args.mode)

                with torch.no_grad():
                    # voxel = clip_text_embdder(text_descr)
                    # voxel = torch.mean(voxel, dim=1).float() #### TODO, we use mean 77 tokens to obtain semantic info
                    voxel = voxel[:, train_i%3].float()
                    
                    if epoch < int(mixup_pct * num_epochs):
                        voxel, perm, betas, select = mixco(voxel)
                
                voxel = voxel.requires_grad_(True)
                # torch.cuda.synchronize()
                data_t = time.time() -t
                t = time.time()
                
                # import pdb; pdb.set_trace();
                with torch.cuda.amp.autocast():
                    optimizer.zero_grad()

                    clip_voxels, clip_voxels_proj = diffusion_prior.module.voxel2clip(voxel) if distributed else diffusion_prior.voxel2clip(voxel)
                    import pdb; pdb.set_trace()
                    
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
                        loss_nce = mixco_nce(
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
                        loss = nce_mult * loss_nce + (prior_mult * loss_prior)
                    elif v2c:
                        loss_nce_sum += loss_nce.item()
                        loss = nce_mult * loss_nce
                    elif prior:
                        loss_prior_sum += loss_prior.item()
                        loss = prior_mult * loss_prior
                    
                    if has_nan(clip_voxels_norm) or has_nan(clip_target_norm) or has_nan(loss) or any(has_nan(param) for param in diffusion_prior.parameters()):
                        print("NaN detected during training!")
                        # break
                        import pdb; pdb.set_trace()

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

            if local_rank==0:
                # ckpt_saving = (train_iter % 100 == 0)
                # if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
                #     # save best model
                #     val_loss = np.mean(val_losses[-(val_i+1):])
                #     if val_loss < best_val_loss:
                #         best_val_loss = val_loss
                #         save_ckpt('best', outdir, epoch, diffusion_prior, optimizer, lr_scheduler, losses, val_losses, lrs)
                #     else:
                #         print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')
                        
                # # if utils.is_interactive():
                # #     clear_output(wait=True)
                    
                # logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                #     "val/loss": np.mean(val_losses[-(val_i+1):]),
                #     "train/lr": lrs[-1],
                #     "train/num_steps": len(losses),
                #     "val/num_steps": len(val_losses),
                #     "train/cosine_sim_base": sims_base / (train_i + 1),
                #     "val/cosine_sim_base": val_sims_base / (val_i + 1),
                #     "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                #     "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                #     "val/val_fwd_pct_correct": val_fwd_percent_correct / (val_i + 1),
                #     "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1),
                #     "train/loss_nce": loss_nce_sum / (train_i + 1),
                #     "train/loss_prior": loss_prior_sum / (train_i + 1),
                #     "val/loss_nce": val_loss_nce_sum / (val_i + 1),
                #     "val/loss_prior": val_loss_prior_sum / (val_i + 1)}
                # progress_bar.set_postfix(**logs)

                # Save model checkpoint and reconstruct
                save_ckpt(f'last', outdir, epoch, diffusion_prior, optimizer, lr_scheduler, losses, val_losses, lrs)
                if epoch % 25 == 0:
                    save_ckpt('ep_{}'.format(epoch), outdir, epoch, diffusion_prior, optimizer, lr_scheduler, losses, val_losses, lrs)
                # import pdb; pdb.set_trace()
    else:
        print('Usage of talking face instruction...')
        t = time.time()
        prefix = 'img' if args.mode == 'image' else 'text'
        for val_i, batch_dict in tqdm(enumerate(val_dl)):
            with torch.no_grad():
                voxel, clip_target = prepare_train_data(batch_dict, vd_clip, use_image_aug=False, mode=args.mode)
                image_embed = None
                pred_img_embed = voxel2img_emb(voxel, diffusion_priors=diffusion_prior, image_embed=image_embed)
                # import pdb; pdb.set_trace()
                s = batch_dict['s'][0].item()

                os.makedirs(prefix+'_clip', exist_ok=True)
                np.save(prefix+'_clip/{:05d}.npy'.format(s), pred_img_embed.cpu().numpy())


def voxel2img_emb(
    voxel, 
    diffusion_priors=None,
    recons_per_sample = 1,
    plotting=True,
    verbose=False,
    img_variations=False,
    seed = 0,
    retrieve = False,
    timesteps_prior = 100,
    n_samples_save=1,
    image_embed=None,
    no_diffusion = False):
    
    device = voxel.device
    brain_recons = None
    
    # voxel=voxel[:n_samples_save]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if diffusion_priors is not None:
        if not isinstance(diffusion_priors, list):
            diffusion_priors = [diffusion_priors]
        brain_clip_embeddings_sum = None
        for diffusion_prior in diffusion_priors:
            brain_clip_embeddings0, proj_embeddings = diffusion_prior.voxel2clip(voxel.to(device).float())
            if retrieve:
                continue
            brain_clip_embeddings0 = brain_clip_embeddings0.view(len(voxel),-1,768)
            if recons_per_sample>0:
                # import pdb; pdb.set_trace()
                if no_diffusion:
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1, 1)
                    # brain_clip_embeddings = copy.deepcopy(proj_embeddings)
                    brain_clip_embeddings = F.normalize(proj_embeddings, p=2, dim=-1) * 2.0
                    # import pdb; pdb.set_trace()
                elif not img_variations:
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1, 1)
                    try:
                        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps_prior,
                                                generator=generator, image_embed=image_embed)
                    except:
                        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps_prior, image_embed=image_embed)
                    # import pdb; pdb.set_trace()
                else:
                    brain_clip_embeddings0 = brain_clip_embeddings0.view(-1,768)
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1)
                    brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = 1000, #1000 timesteps used from nousr pretraining
                                                generator=generator, image_embed=image_embed)
                if brain_clip_embeddings_sum is None:
                    brain_clip_embeddings_sum = brain_clip_embeddings
                else:
                    brain_clip_embeddings_sum += brain_clip_embeddings

        # average embeddings for all diffusion priors
        if recons_per_sample>0:
            brain_clip_embeddings = brain_clip_embeddings_sum / len(diffusion_priors)
    # import pdb; pdb.set_trace()
    return brain_clip_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_lr', type=float, default=0.0002)
    parser.add_argument('--max_epoch', type=int, default=240)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--clip_size', type=int, default=768)
    parser.add_argument('--jobname', type=str, default='latent_diffusion_image')
    parser.add_argument('--resume_from_ckpt', type=bool, default=True)
    parser.add_argument('--is_tensorboard_log', type=bool, default=True)
    parser.add_argument('--is_test', type=bool, default=False)
    parser.add_argument('--ckpt_path', type=str, default='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/train_logs/latent_diffusion_image/last.pth')
    parser.add_argument('--mode', type=str, default='image')

    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--use_projector', type=bool, default=True)
    parser.add_argument("--epoch", type=int, default=0, help='number of epochs')
    parser.add_argument("--log_loss_steps", type=int, default=5)

    args = parser.parse_args()

    # if args.mode == 'image': args.jobname = 'latent_diffusion_image'
    # elif args.mode == 'text': args.jobname = 'latent_diffusion_text'
    # else: raise ValueError

    clip_cfg = {'symbol': 'clip',
                'args': {},
                'name': 'clip_frozen',
                'type': 'clip_frozen'}

    clip_cfg = EasyDict(clip_cfg)
    vd_clip = VDCLIP(clip_cfg)
    # vd_clip.text_model = None # delete text branch for memory saving
    for param in vd_clip.parameters():
        param.requires_grad = False
    vd_clip = vd_clip.eval()

    dataset_cfg_str = """
    train:
      target: third_party.StableDiffusionReconstruction.codes.utils.nsd_creater.NIPS23NSDDataset
      params:
        nsd_root: '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd'
        resolution: 320
        split: 'train'
        is_reconstruct_mode: True
        url: 'nsd_data_dir/train_subj01_{0..17}.tar'
        reconstruct_prob: 1.05

    validation:
      target: third_party.StableDiffusionReconstruction.codes.utils.nsd_creater.NIPS23NSDDataset
      params:
        nsd_root: '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd'
        resolution: 320
        split: 'test'
        is_reconstruct_mode: True
        url: 'nsd_data_dir/test_subj01_{0..1}.tar'
        reconstruct_prob: 1.05
    """
    dataset_cfg = EasyDict(yaml.safe_load(dataset_cfg_str))
    # import pdb; pdb.set_trace();
    train_dataset = instantiate_from_config(dataset_cfg['train'])
    val_dataset = instantiate_from_config(dataset_cfg['validation'])

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = None
    # prior model
    guidance_scale = 3.5
    timesteps = 100
    depth = 6
    dim_head = 64
    clip_size = args.clip_size
    # out_dim = clip_size
    heads = clip_size//16
    # import pdb; pdb.set_trace();
    num_tokens = 257 if args.mode == 'image' else 77
    prior_network = VersatileDiffusionPriorNetwork(
            dim=clip_size,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = num_tokens,
            learned_query_mode="pos_emb"
        )
    # import pdb; pdb.set_trace();
    prior_network = prior_network.to(torch.device("cuda"))

    # clip text to emotion latent model
    clip_size = args.clip_size
    num_voxels = 15724
    voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=clip_size*num_tokens,clip_size=clip_size,use_projector=args.use_projector)
    voxel2clip = BrainNetwork(**voxel2clip_kwargs).cuda()

    # use dalle interface to include prior model and clip text-to-emotion models
    timesteps = 100
    diffusion_prior = InstructDiffusionPrior(
        net=prior_network,
        image_embed_dim=clip_size,
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
