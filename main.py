# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Removed Pytorch-lightning and supported deepspeed by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import argparse, os, sys, datetime, glob
import numpy as np
import time
import json
import pickle
import wandb
import deepspeed

from packaging import version
from omegaconf import OmegaConf
from functools import partial
from PIL import Image

from timm.utils import AverageMeter

import torch
import torch.nn as nn
import torchvision
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(proj_root,"stable_diffusion"))

# from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma
from utils.logger import create_logger
from utils.utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper
from utils.deepspeed import create_ds_config

# for inference
import k_diffusion as K
import einops
import random
from einops import rearrange

third_party_proj_root = os.path.join(proj_root, 'third_party')
sys.path.append(third_party_proj_root)
sys.path.append('third_party/CoDI')
sys.path.append('third_party/versatile_diffusion')


def concat_dict(c_dict):
    res_dict = {}
    for k, v in c_dict.items():
        if isinstance(v, list):
            res_dict[k] = torch.cat(v,dim=1)
        elif isinstance(v, dict):
            sub_res_dict = v
            for sub_k, sub_v in sub_res_dict.items():
                sub_res_dict[sub_k] = sub_v
            res_dict[k] = sub_res_dict
        else: 
            raise ValueError
    return res_dict

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, fmri_cfg_scale):
        cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
        
        cond = {k: torch.cat(v,dim=1) for k,v in cond.items()}
        uncond = {k: torch.cat(v,dim=1) for k,v in uncond.items()}

        # import pdb; pdb.set_trace();

        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"], uncond["c_crossattn"], uncond["c_crossattn"]])],
            "c_concat": [torch.cat([cond["c_concat"], cond["c_concat"], uncond["c_concat"]])],
        }

        # cond = concat_dict(cond)
        # uncond = concat_dict(uncond)
        # cfg_cond = {
        #     "c_crossattn": [torch.cat([cond["c_crossattn"], uncond["c_crossattn"], cond["c_crossattn"]])],
        #     "c_crossattn_1": [torch.cat([cond["c_crossattn_1"], cond["c_crossattn_1"], uncond["c_crossattn_1"]])],
        #     "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        # }


        # cfg_cond['c_crossattn_1'] = {}
        # cfg_cond["c_crossattn_1"]["image_emb"] = [torch.cat([cond["c_crossattn_1"]["image_emb"],
        #                                                     cond["c_crossattn_1"]["image_emb"],
        #                                                     uncond["c_crossattn_1"]["image_emb"]])]
        # cfg_cond["c_crossattn_1"]["text_emb"] = [torch.cat([cond["c_crossattn_1"]["text_emb"],
        #                                                     cond["c_crossattn_1"]["text_emb"],
        #                                                     uncond["c_crossattn_1"]["text_emb"]])]
        # cfg_cond["c_crossattn_1"]["fmri_vae"] = [torch.cat([cond["c_crossattn_1"]["fmri_vae"],
        #                                                     cond["c_crossattn_1"]["fmri_vae"],
        #                                                     uncond["c_crossattn_1"]["fmri_vae"]])]
        # import pdb; pdb.set_trace()
        out_cond, out_img_cond, out_txt_cond \
            = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return 0.5 * (out_img_cond + out_txt_cond) + \
            text_cfg_scale * (out_cond - out_img_cond) + \
                fmri_cfg_scale * (out_cond - out_txt_cond)


def wandb_log(*args, **kwargs):
    if dist.get_rank() == 0:
        wandb.log(*args, **kwargs)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--amd",
        action="store_true",
        default=False,
        help="amd",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        # required=False,
        default=int(os.environ.get('LOCAL_RANK', 0)),
        help="local rank for DistributedDataParallel",
    )
    parser.add_argument(
        "--filter_mode",
        type=str,
        default='no_filter',
        help="",
    )
    parser.add_argument(
        "--cfg_text",
        type=float,
        default=7.5,
        help="",
    )
    parser.add_argument(
        "--vis",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--isTrain",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--prospect_ckpt_path",
        type=str,
        default='',
        help="",
    )
    parser.add_argument(
        "--is_inst_edit",
        type=bool,
        default=False,
        help="",
    )
    parser.add_argument(
        "--is_inst_gen",
        type=bool,
        default=False,
        help="",
    )
    parser.add_argument(
        "--cfg_text_edit",
        type=float,
        default=7.5,
        help="",
    )
    parser.add_argument(
        "--layout_path",
        type=str,
        default='',
        help="",
    )
    parser.add_argument(
        "--whitelist_path",
        type=str,
        default='',
        help="",
    )
    return parser


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig():
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            if "target" in train:
                self.dataset_configs["train"] = train
                self.train_dataloader = self._train_dataloader
            else:
                for ds in train:
                    ds_name = str([key for key in ds.keys()][0])
                    self.dataset_configs[ds_name] = ds
                self.train_dataloader = self._train_concat_dataloader

        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_concat_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['ds1'], Txt2ImgIterableBaseDataset)

        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        concat_dataset = []
        for ds in self.datasets.keys():
            concat_dataset.append(self.datasets[ds])

        concat_dataset = ConcatDataset(concat_dataset)
        sampler_train = torch.utils.data.DistributedSampler(
            concat_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )
        return DataLoader(concat_dataset, batch_size=self.batch_size, sampler=sampler_train,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=True)

    def _train_dataloader(self):
        # is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        is_iterable_dataset = False
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        sampler_train = torch.utils.data.DistributedSampler(
            self.datasets["train"], num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, sampler=sampler_train,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=True)

    def _val_dataloader(self, shuffle=False):
        is_iterable_dataset = False
        # is_iterable_dataset = isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, persistent_workers=True)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, persistent_workers=True)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=True)

def test_one_epoch(config, model, model_ema, data_loader, val_data_loader, optimizer, epoch, 
        lr_scheduler, scaler, model_wrap, model_wrap_cfg, save_dir, cfg_text, cfg_text_edit,
        prospect_words=None, is_inst_edit=False, is_inst_gen=False, layout_in=None, whitelist_path=''):
    # import pdb; pdb.set_trace()
    model.eval()
    epoch, idx = 999999, 999999
    whitelist = None
    if os.path.exists(whitelist_path): 
        with open(whitelist_path, 'r') as f:
            whitelist = f.read().splitlines()
            import pdb; pdb.set_trace()
        
    with torch.no_grad():
        for val_idx, batch in enumerate(val_data_loader):
            batch_size = batch['image'].shape[0]
            if model_wrap is not None:
                if is_inst_edit is True or is_inst_gen is True:
                    if whitelist is not None and '{:06d}'.format(batch['s'][0]) not in whitelist: continue

                    model.log_images(batch, epoch, idx, val_idx, model_wrap, model_wrap_cfg, 
                                 save_dir, 'val', cfg_text=cfg_text, cfg_text_edit=cfg_text_edit, cfg_fmri=2.5,
                                 prospect_words=prospect_words, is_inst_edit=is_inst_edit, is_inst_gen=is_inst_gen)
                    # if val_idx > 15: break
                    ## only run 100 iterations for quick selection of images
                    
                # elif layout_in is not None:
                #     model.log_images(batch, epoch, idx, val_idx, model_wrap, model_wrap_cfg, 
                #                  save_dir, 'val', cfg_text=cfg_text, cfg_text_edit=cfg_text_edit, cfg_fmri=2.5,
                #                  prospect_words=None, is_inst_edit=False, layout_in=layout_in)
                else:
                    model.save_text(batch, epoch, idx, val_idx, model_wrap, model_wrap_cfg, 
                                 save_dir, 'val', cfg_text=cfg_text, cfg_text_edit=cfg_text_edit, cfg_fmri=2.5,
                                 prospect_words=None, is_inst_edit=False, is_inst_gen=False)
                    # model.save_fmri_vae(batch, epoch, idx, val_idx, model_wrap, model_wrap_cfg, 
                    #              save_dir, 'val', cfg_text=cfg_text, cfg_text_edit=cfg_text_edit, cfg_fmri=2.5,
                    #              prospect_words=None, is_inst_edit=False, is_inst_gen=False)
                    # import pdb; pdb.set_trace()
                    model.log_images(batch, epoch, idx, val_idx, model_wrap, model_wrap_cfg, 
                                 save_dir, 'val', cfg_text=cfg_text, cfg_text_edit=cfg_text_edit, cfg_fmri=2.5,
                                 prospect_words=None, is_inst_edit=False, is_inst_gen=False)
    model.train()

def train_one_epoch(config, model, model_ema, data_loader, val_data_loader, optimizer, epoch, 
        lr_scheduler, scaler, model_wrap, model_wrap_cfg, save_dir, cfg_text):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    accumul_steps = config.trainer.accumulate_grad_batches
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_scale_meter = AverageMeter()
    loss_scale_meter_min = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, batch in enumerate(data_loader):
        batch_size = batch['image'].shape[0]
        if config.model.params.deepspeed != '':
            loss, _ = model(batch, idx, accumul_steps)
            model.backward(loss)
            # import pdb; pdb.set_trace()

            model.step()
            loss_scale = optimizer.cur_scale
            grad_norm = model.get_global_grad_norm()

            with torch.no_grad():
                if idx % config.trainer.accumulate_grad_batches == 0:
                    model_ema(model)

            loss_number = loss.item()
        else:
            with amp.autocast(enabled=config.model.params.fp16):
                loss, _ = model(batch, idx, accumul_steps)

            if config.trainer.accumulate_grad_batches > 1:
                loss = loss / config.trainer.accumulate_grad_batches
                scaler.scale(loss).backward()
                # loss.backward()
                if config.trainer.clip_grad > 0.0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.trainer.accumulate_grad_batches == 0:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                    # scaler.unscale_grads()
                    # optimizer.step()
                    # optimizer.zero_grad()
                    # lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if config.trainer.clip_grad > 0.0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
                # lr_scheduler.step_update(epoch * num_steps + idx)
            
            loss_scale = scaler.get_scale()
            loss_number = loss.item() * config.trainer.accumulate_grad_batches

        torch.cuda.synchronize()
        
        loss_meter.update(loss_number, batch_size)
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        else:
            norm_meter.update(0.0)

        loss_scale_meter.update(loss_scale)
        # loss_scale_meter.update(0.0)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {loss_scale_meter.val:.4f} ({loss_scale_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

        if (epoch * num_steps * data_loader.batch_size + idx) % 10000 == 0:
            log_message = dict(
                lr=optimizer.param_groups[0]['lr'], 
                time=batch_time.val, 
                epoch=epoch, 
                iter=idx, 
                loss=loss_meter.val, 
                grad_norm=norm_meter.val, 
                loss_scale=loss_scale_meter.val, 
                memory=torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
                global_iter=epoch * num_steps + idx)

            # log_message.update({'ref_img': wandb.Image(unnormalize(img[:8].cpu().float())), 'mask': wandb.Image(mask[:8].cpu().float().unsqueeze(1))})
            # if x_rec is not None:
                # log_message.update({'rec_img': wandb.Image(unnormalize(x_rec[:8].cpu().float()))})
            wandb_log(
                data=log_message,
                step=epoch * num_steps + idx,
            )

        # print(epoch * num_steps + idx)
        # import pdb; pdb.set_trace();
        save_dir = visdir
        if (epoch * num_steps + idx) % 4000 == 0:
            with torch.no_grad():
                # if random.uniform(0,1) > 0.5:
                if True:
                    for val_idx, batch in enumerate(val_data_loader):
                        batch_size = batch['image'].shape[0]
                        if model_wrap is not None:
                            model.log_images(batch, epoch, idx, val_idx, model_wrap, model_wrap_cfg, save_dir, 'val', cfg_text=cfg_text, cfg_fmri=2.5)

                        if val_idx == 5:
                            break
                else:
                    for val_idx, batch in enumerate(data_loader):
                        batch_size = batch['image'].shape[0]
                        if model_wrap is not None:
                            model.log_images(batch, epoch, idx, val_idx, model_wrap, model_wrap_cfg, save_dir, 'train', cfg_text=cfg_text, cfg_fmri=2.5)

                        if val_idx == 5:
                            break

        if idx == num_steps - 1:
            with torch.no_grad():
                model_ema.store(model.parameters())
                model_ema.copy_to(model)
                for val_idx, batch in enumerate(val_data_loader):
                    batch_size = batch['image'].shape[0]

                    loss, _ = model(batch, -1, 1)

                    loss_number = loss.item()
                    val_loss_meter.update(loss_number, batch_size)
                    if val_idx % 10 == 0:
                        logger.info(
                            f'Val: [{val_idx}/{len(val_data_loader)}]\t'
                            f'loss {val_loss_meter.val:.4f} ({val_loss_meter.avg:.4f})\t')
                    if val_idx == 50:
                        break
                model_ema.restore(model.parameters())

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

## this is to control which paramters are needed to be optimized
def filter_optimized_params(model, args):
    if args.filter_mode == 'no_filter':
        # param_groups = model.parameters()
        filtered_params = [param for name, param in model.named_parameters() if param.requires_grad is True]
        filtered_names = [name for name, param in model.named_parameters() if param.requires_grad is True]
        param_groups = [{'params': filtered_params, 'lr': model.learning_rate}]
        # import pdb; pdb.set_trace()
    elif args.filter_mode == 'tune_controlunet':
        filtered_params = [param for name, param in model.named_parameters() if param.requires_grad is True and 'model.diffusion_model.adaptor_blocks' in name ]
        filtered_names = [name for name, param in model.named_parameters() if param.requires_grad is True and 'model.diffusion_model.adaptor_blocks' in name]
        print(filtered_names)
        # import pdb; pdb.set_trace()
        param_groups = [{'params': filtered_params, 'lr': model.learning_rate}]
    elif args.filter_mode == 'tune_sideconv':
        # import pdb; pdb.set_trace();
        filtered_params = [param for name, param in model.named_parameters() if param.requires_grad is True and ('control_model.zero_convs' in name or 'control_model.middle_block_out' in name) or 'control_model.input_hint_block' in name or 'merge_blocks' in name or 'x0_block_out' in name]
        filtered_names = [name for name, param in model.named_parameters() if param.requires_grad is True and ('control_model.zero_convs' in name or 'control_model.middle_block_out' in name) or 'control_model.input_hint_block' in name or 'merge_blocks' in name or 'x0_block_out' in name]
        print(filtered_names)
        param_groups = [{'params': filtered_params, 'lr': model.learning_rate}]
    elif args.filter_mode == 'tune_instruct':
        filtered_params = [param for name, param in model.named_parameters() if param.requires_grad is True]
        filtered_names = [name for name, param in model.named_parameters() if param.requires_grad is True]
        param_groups = [{'params': filtered_params, 'lr': model.learning_rate}]
    elif args.filter_mode == 'tune_sideconv_sdunlock':
        filtered_params = [param for name, param in model.named_parameters() if param.requires_grad is True and ('control_model.zero_convs' in name or 'control_model.middle_block_out' in name or 'model.diffusion_model.out' in name)]
        filtered_names = [name for name, param in model.named_parameters() if param.requires_grad is True and ('control_model.zero_convs' in name or 'control_model.middle_block_out' in name or 'model.diffusion_model.out' in name)]
        print(filtered_names)
        # import pdb; pdb.set_trace();
        param_groups = [{'params': filtered_params, 'lr': model.learning_rate}]
    elif args.filter_mode == 'dual_condition':
        filtered_params = [param for name, param in model.named_parameters() if param.requires_grad is True and ('diffusion_model.' not in name or 'time_embed_condtion' in name)]
        param_groups = [{'params': filtered_params, 'lr': model.learning_rate}]
    elif args.filter_mode == 'dual_control':
        # filtered_params = [param for name, param in model.named_parameters() if param.requires_grad is True and ('control_model' in name or 'cond_stage_model_fmri' in name)]
        # filtered_params = [param for name, param in model.named_parameters() if param.requires_grad is True and ('cond_stage_model_fmri' in name)]
        filtered_params = [param for name, param in model.named_parameters() if param.requires_grad is True and ('control_model' in name)]
        param_groups = [{'params': filtered_params, 'lr': model.learning_rate}]
    else:
        raise ValueError
    print('we apply filter mode: ', args.filter_mode)
    # import pdb; pdb.set_trace()
    return param_groups


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    assert opt.name
    cfg_fname = os.path.split(opt.base[0])[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    nowname = f"{cfg_name}_{opt.name}"
    logdir = os.path.join(opt.logdir, nowname)

    random_seed = 8866
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # If you're using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    if opt.amd:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.local_rank)
        torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    else:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    
    seed = opt.seed + dist.get_rank()
    # seed = opt.seed
    # import pdb; pdb.set_trace()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    visdir = os.path.join(logdir, "visualize")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(visdir, exist_ok=True)

    # init and save configs
    # config: the configs in the config file
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    if config.model.params.deepspeed != '':
        create_ds_config(opt, config, cfgdir)

    if dist.get_rank() == 0:
        os.environ["WANDB_MODE"] = "offline"

        run = wandb.init(
            # offline=True,
            id=nowname,
            name=nowname,
            project='readoutpose',
            config=OmegaConf.to_container(config, resolve=True),
        )

    logger = create_logger(output_dir=logdir, dist_rank=dist.get_rank(), name=f"{nowname}")
    
    resume_file = auto_resume_helper(config, ckptdir)
    if resume_file:
        resume = True
        logger.info(f'resume checkpoint in {resume_file}')
    else:
        resume = False
        logger.info(f'no checkpoint found in {ckptdir}, ignoring auto resume')

    # model
    model = instantiate_from_config(config.model)
    if os.path.exists(opt.prospect_ckpt_path):
        if model.embedding_manager is not None:
            model.embedding_manager.load(opt.prospect_ckpt_path)
    else:
        print('{} not exist.'.format(opt.prospect_ckpt_path))
    # import pdb; pdb.set_trace()

    model_ema = LitEma(model, decay_resume=config.model.params.get('ema_resume', 0.9999))

    # data
    if not opt.isTrain:
        config.data['params']['batch_size'] = 1
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    data_loader_train = data.train_dataloader()
    data_loader_val = data.val_dataloader()

    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate    
    ngpu = dist.get_world_size()
    if 'accumulate_grad_batches' in config.trainer:
        accumulate_grad_batches = config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")

    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")
    # import pdb; pdb.set_trace();

    if not opt.amd:
        model.cuda()

    if config.model.params.fp16 and config.model.params.deepspeed == '':
        scaler = amp.GradScaler()
        param_groups = model.parameters()
    else:
        # scaler = None
        scaler = amp.GradScaler()
        param_groups = filter_optimized_params(model, opt)

    if config.model.params.deepspeed != '':
        model, optimizer, _, _ = deepspeed.initialize(
            args=config,
            model=model,
            model_parameters=param_groups,
            dist_init_required=False,
        )
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = model.learning_rate

        for name, param in model.named_parameters():
            param.global_name = name
        model_without_ddp = model
        lr_scheduler = None
        model_ema = model_ema.to(next(model.parameters()).device)
    else:
        optimizer, lr_scheduler = model.configure_optimizers()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], broadcast_buffers=False)
        model_without_ddp = model.module

    # print(optimizer.param_groups[1])
    if opt.resume != '':
        resume_file = opt.resume
    if resume_file:
        _, start_epoch = load_checkpoint(resume_file, config, model_without_ddp, model_ema, optimizer, lr_scheduler, scaler, logger)
    else:
        start_epoch = 0

    logger.info("Start training")
    start_time = time.time()

    # k-diffusion wrapper
    if opt.vis:
        model_wrap = K.external.CompVisDenoiser(model)
        model_wrap_cfg = CFGDenoiser(model_wrap)
    else:
        model_wrap = None
        model_wrap_cfg = None
    
    if opt.isTrain:
        for epoch in range(start_epoch, config.trainer.max_epochs):
            data_loader_train.sampler.set_epoch(epoch)
            # print(data_loader_train.batch_size)
            # import pdb; pdb.set_trace();
            train_one_epoch(config, model, model_ema, data_loader_train, data_loader_val, 
                    optimizer, epoch, lr_scheduler, scaler, model_wrap, model_wrap_cfg, visdir, cfg_text=opt.cfg_text)
            if epoch % config.trainer.save_freq == 0:
                save_checkpoint(ckptdir, config, epoch, model_without_ddp, model_ema, 0., optimizer, lr_scheduler, scaler, logger)
    else:
        epoch = 999999
        prospect_words = ['Change the picture to * style.'] #* 10
        from PIL import Image
        from torchvision import transforms
        if os.path.exists(opt.layout_path):
            layout_pil = Image.open(opt.layout_path)
            layout_in = transforms.ToTensor()(layout_pil).unsqueeze(0)*2. - 1.0
        else:
            layout_in = None
        test_one_epoch(config, model, model_ema, data_loader_train, data_loader_val, 
                        optimizer, epoch, lr_scheduler, scaler, model_wrap, model_wrap_cfg, visdir, 
                        cfg_text=opt.cfg_text, cfg_text_edit=opt.cfg_text_edit,
                        prospect_words=prospect_words, 
                        is_inst_edit=opt.is_inst_edit,
                        is_inst_gen=opt.is_inst_gen,
                        whitelist_path=opt.whitelist_path,
                        layout_in=layout_in)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
