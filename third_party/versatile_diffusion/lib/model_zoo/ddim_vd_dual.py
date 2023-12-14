import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from .diffusion_utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

from .ddim import DDIMSampler

class DDIMSampler_Dual(DDIMSampler):
    @torch.no_grad()
    def sample(self,
               steps,
               shape,
               xt=None,
               conditioning=None,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               xtype='image',
               ctype='prompt',
               eta=0.,
               temperature=1.,
               noise_dropout=0.,
               verbose=True,
               log_every_t=100,):

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
        print(f'Data shape for DDIM sampling is {shape}, eta {eta}')
        samples, intermediates = self.ddim_sampling(
            shape,
            xt=xt,
            conditioning=conditioning, 
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            xtype=xtype,
            ctype=ctype,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            log_every_t=log_every_t,)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, 
                      shape,
                      xt=None,
                      conditioning=None,
                      unconditional_guidance_scale=1., 
                      unconditional_conditioning=None,
                      xtype='image',
                      ctype='prompt',
                      ddim_use_original_steps=False,
                      timesteps=None, 
                      noise_dropout=0., 
                      temperature=1., 
                      log_every_t=100,):

        device = self.model.model.diffusion_model.device
        bs = shape[0]
        if xt is None:
            xt = torch.randn(shape, device=device, dtype=conditioning.dtype)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'pred_xt': [], 'pred_x0': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        pred_xt = xt
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(
                pred_xt, conditioning, ts, index, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning, 
                xtype=xtype,
                ctype=ctype,
                use_original_steps=ddim_use_original_steps,
                noise_dropout=noise_dropout,
                temperature=temperature,)
            pred_xt, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['pred_xt'].append(pred_xt)
                intermediates['pred_x0'].append(pred_x0)

        return pred_xt, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, conditioning, t, index, 
                      unconditional_guidance_scale=1., 
                      unconditional_conditioning=None, 
                      xtype='image',
                      ctype='prompt',
                      repeat_noise=False, 
                      use_original_steps=False, 
                      noise_dropout=0.,
                      temperature=1.,):

        b, *_, device = *x.shape, self.model.model.diffusion_model.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, conditioning, xtype=xtype, ctype=ctype)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, conditioning])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, xtype=xtype, ctype=ctype).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        if xtype == 'image':
            extended_shape = (b, 1, 1, 1)
        elif xtype == 'text':
            extended_shape = (b, 1)

        a_t = torch.full(extended_shape, alphas[index], device=device, dtype=x.dtype)
        a_prev = torch.full(extended_shape, alphas_prev[index], device=device, dtype=x.dtype)
        sigma_t = torch.full(extended_shape, sigmas[index], device=device, dtype=x.dtype)
        sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=x.dtype)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample_dc(self,
               steps,
               shape,
               xt=None,
               first_conditioning=None,
               second_conditioning=None,
               unconditional_guidance_scale=1.,
               xtype='image',
               first_ctype='prompt',
               second_ctype='prompt',
               eta=0.,
               temperature=1.,
               mixed_ratio=0.5,
               noise_dropout=0.,
               verbose=True,
               log_every_t=100,):

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
        print(f'Data shape for DDIM sampling is {shape}, eta {eta}')
        samples, intermediates = self.ddim_sampling_dc(
            shape,
            xt=xt,
            first_conditioning=first_conditioning,
            second_conditioning=second_conditioning,
            unconditional_guidance_scale=unconditional_guidance_scale,
            xtype=xtype,
            first_ctype=first_ctype,
            second_ctype=second_ctype,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            log_every_t=log_every_t,
            mixed_ratio=mixed_ratio, )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling_dc(self, 
                      shape,
                      xt=None,
                      first_conditioning=None,
                      second_conditioning=None,
                      unconditional_guidance_scale=1., 
                      xtype='image',
                      first_ctype='prompt',
                      second_ctype='prompt',
                      ddim_use_original_steps=False,
                      timesteps=None, 
                      noise_dropout=0., 
                      temperature=1.,
                      mixed_ratio=0.5,
                      log_every_t=100,):

        device = self.model.model.diffusion_model.device
        bs = shape[0]
        if xt is None:
            xt = torch.randn(shape, device=device, dtype=first_conditioning[1].dtype)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'pred_xt': [], 'pred_x0': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        pred_xt = xt
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim_dc(
                pred_xt, 
                first_conditioning, 
                second_conditioning, 
                ts, index, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                xtype=xtype,
                first_ctype=first_ctype,
                second_ctype=second_ctype,
                use_original_steps=ddim_use_original_steps,
                noise_dropout=noise_dropout,
                temperature=temperature,
                mixed_ratio=mixed_ratio,)
            pred_xt, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['pred_xt'].append(pred_xt)
                intermediates['pred_x0'].append(pred_x0)

        return pred_xt, intermediates

    @torch.no_grad()
    def p_sample_ddim_dc(self, x, 
                      first_conditioning,
                      second_conditioning,
                      t, index, 
                      unconditional_guidance_scale=1., 
                      xtype='image',
                      first_ctype='prompt',
                      second_ctype='prompt',
                      repeat_noise=False, 
                      use_original_steps=False, 
                      noise_dropout=0.,
                      temperature=1.,
                      mixed_ratio=0.5,):

        b, *_, device = *x.shape, self.model.model.diffusion_model.device

        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        first_c = torch.cat(first_conditioning)
        second_c = torch.cat(second_conditioning)

        e_t_uncond, e_t = self.model.apply_model_dc(
            x_in, t_in, first_c, second_c, xtype=xtype, first_ctype=first_ctype, second_ctype=second_ctype, mixed_ratio=mixed_ratio).chunk(2)

        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        if xtype == 'image':
            extended_shape = (b, 1, 1, 1)
        elif xtype == 'text':
            extended_shape = (b, 1)

        a_t = torch.full(extended_shape, alphas[index], device=device, dtype=x.dtype)
        a_prev = torch.full(extended_shape, alphas_prev[index], device=device, dtype=x.dtype)
        sigma_t = torch.full(extended_shape, sigmas[index], device=device, dtype=x.dtype)
        sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=x.dtype)

        # current prediction for x_0
        # import pdb; pdb.set_trace()
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
    def get_x0(self, x_gen, e_t_gen, index, use_original_steps=False):
        b, *_, device = *x_gen.shape, self.model.model.diffusion_model.device
        if xtype == 'image':
            extended_shape = (b, 1, 1, 1)
        elif xtype == 'text':
            extended_shape = (b, 1)
        
        import pdb; pdb.set_trace()
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        a_t = torch.full(extended_shape, alphas[index], device=device, dtype=x_gen.dtype)
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=x_gen.dtype)

        pred_x0_gen = (x_gen - sqrt_one_minus_at * e_t_gen) / a_t.sqrt()
        return pred_x0_gen

    @torch.no_grad()
    def p_sample_ddim_dual(self, 
                      x_gen, 
                      x_edit,
                      t, 
                      cond_dict,
                      index, 
                      unconditional_guidance_scale_gen=1., 
                      unconditional_guidance_scale_edit=1.,
                      xtype='image',
                      first_ctype='prompt',
                      second_ctype='prompt',
                      repeat_noise=False, 
                      use_original_steps=False, 
                      noise_dropout=0.,
                      temperature=1.,
                      mixed_ratio=0.5,):

        b, *_, device = *x_edit.shape, self.model.model.diffusion_model.device

        x_gen_in = torch.cat([x_gen] * 2)
        x_edit_in = torch.cat([x_edit] * 2)
        t_in = torch.cat([t] * 2)
        # first_c = torch.cat(first_conditioning)
        # second_c = torch.cat(second_conditioning)

        # e_t_gen_cat, e_t_edit_cat = self.model.apply_model_dc(
        #     x_gen_in, x_edit_in, t_in, first_c, second_c, xtype=xtype, first_ctype=first_ctype, second_ctype=second_ctype, mixed_ratio=mixed_ratio)#.chunk(4)
        # import pdb; pdb.set_trace()
        e_t_gen_cat, e_t_edit_cat = self.model.apply_model(
            x_gen_in, x_edit_in, t_in, cond_dict)#.chunk(4)

        e_t_uncond_gen, e_t_gen = e_t_gen_cat.chunk(2)
        e_t_uncond_edit, e_t_edit = e_t_edit_cat.chunk(2)

        e_t_gen = e_t_uncond_gen + unconditional_guidance_scale_gen * (e_t_gen - e_t_uncond_gen)
        e_t_edit = e_t_uncond_edit + unconditional_guidance_scale_edit * (e_t_edit - e_t_uncond_edit)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        if xtype == 'image':
            extended_shape = (b, 1, 1, 1)
        elif xtype == 'text':
            extended_shape = (b, 1)

        a_t = torch.full(extended_shape, alphas[index], device=device, dtype=x_edit.dtype)
        a_prev = torch.full(extended_shape, alphas_prev[index], device=device, dtype=x_edit.dtype)
        sigma_t = torch.full(extended_shape, sigmas[index], device=device, dtype=x_edit.dtype)
        sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=x_edit.dtype)

        # current prediction for x_0
        # import pdb; pdb.set_trace()
        pred_x0_gen = (x_gen - sqrt_one_minus_at * e_t_gen) / a_t.sqrt()
        pred_x0_edit = (x_edit - sqrt_one_minus_at * e_t_edit) / a_t.sqrt()
        dir_xt_gen = (1. - a_prev - sigma_t**2).sqrt() * e_t_gen
        dir_xt_edit = (1. - a_prev - sigma_t**2).sqrt() * e_t_edit
        noise_gen = sigma_t * noise_like(x_gen, repeat_noise) * temperature
        noise_edit = sigma_t * noise_like(x_edit, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise_gen = torch.nn.functional.dropout(noise_gen, p=noise_dropout)
            noise_edit = torch.nn.functional.dropout(noise_edit, p=noise_dropout)
        x_prev_gen = a_prev.sqrt() * pred_x0_gen + dir_xt_gen + noise_gen
        x_prev_edit = a_prev.sqrt() * pred_x0_edit + dir_xt_edit + noise_edit
        return x_prev_gen, pred_x0_gen, x_prev_edit, pred_x0_edit

    
    @torch.no_grad()
    def p_sample_ddim_dual_cfg(self, 
                      x_gen, 
                      x_edit,
                      t, 
                      cond_dict,
                      index, 
                      unconditional_guidance_scale_gen=1., 
                      unconditional_guidance_scale_text_edit=1.,
                      unconditional_guidance_scale_image_edit=1.,
                      delay_t=None,
                      xtype='image',
                      first_ctype='prompt',
                      second_ctype='prompt',
                      repeat_noise=False, 
                      use_original_steps=False, 
                      noise_dropout=0.,
                      temperature=1.,
                      mixed_ratio=0.5,
                      is_save_intermediate=True,
                      is_save_x0=False,
                      sqrt_one_minus_at=None,
                      a_t=None):

        b, *_, device = *x_edit.shape, self.model.model.diffusion_model.device

        x_gen_in = torch.cat([x_gen] * 3)
        x_edit_in = torch.cat([x_edit] * 3)
        t_in = torch.cat([t] * 3)
        
        e_t_gen_cat, e_t_edit_cat = self.model.apply_model(
                        x_gen_in, x_edit_in, t_in, cond_dict, 
                        is_save_intermediate=is_save_intermediate, 
                        is_save_x0=is_save_x0,
                        sqrt_one_minus_at=sqrt_one_minus_at,
                        a_t=a_t)#.chunk(4)

        e_t_uncond_gen, _, e_t_gen_full = e_t_gen_cat.chunk(3)
        e_t_uncond_text_edit, e_t_uncond_image_edit, e_t_edit_full = e_t_edit_cat.chunk(3)

        e_t_gen = e_t_uncond_gen + unconditional_guidance_scale_gen * (e_t_gen_full - e_t_uncond_gen)
        e_t_edit = 0.5 * (e_t_uncond_text_edit + e_t_uncond_image_edit) + \
                        unconditional_guidance_scale_text_edit * (e_t_edit_full - e_t_uncond_text_edit) + \
                        unconditional_guidance_scale_image_edit * (e_t_edit_full - e_t_uncond_image_edit)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        if xtype == 'image':
            extended_shape = (b, 1, 1, 1)
        elif xtype == 'text':
            extended_shape = (b, 1)

        a_t = torch.full(extended_shape, alphas[index], device=device, dtype=x_edit.dtype)
        a_prev = torch.full(extended_shape, alphas_prev[index], device=device, dtype=x_edit.dtype)
        sigma_t = torch.full(extended_shape, sigmas[index], device=device, dtype=x_edit.dtype)
        sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=x_edit.dtype)

        # current prediction for x_0
        # import pdb; pdb.set_trace()
        pred_x0_gen = (x_gen - sqrt_one_minus_at * e_t_gen) / a_t.sqrt()
        pred_x0_edit = (x_edit - sqrt_one_minus_at * e_t_edit) / a_t.sqrt()
        dir_xt_gen = (1. - a_prev - sigma_t**2).sqrt() * e_t_gen
        dir_xt_edit = (1. - a_prev - sigma_t**2).sqrt() * e_t_edit
        noise_gen = sigma_t * noise_like(x_gen, repeat_noise) * temperature
        noise_edit = sigma_t * noise_like(x_edit, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise_gen = torch.nn.functional.dropout(noise_gen, p=noise_dropout)
            noise_edit = torch.nn.functional.dropout(noise_edit, p=noise_dropout)
        x_prev_gen = a_prev.sqrt() * pred_x0_gen + dir_xt_gen + noise_gen
        x_prev_edit = a_prev.sqrt() * pred_x0_edit + dir_xt_edit + noise_edit
        return x_prev_gen, pred_x0_gen, x_prev_edit, pred_x0_edit
    
    @torch.no_grad()
    def asyn_p_sample_ddim_dual_cfg(self, 
                      x_gen, 
                      x_edit,
                      t_gen,
                      t_edit,
                      cond_dict,
                      index, 
                      offset,
                      unconditional_guidance_scale_gen=1., 
                      unconditional_guidance_scale_text_edit=1.,
                      unconditional_guidance_scale_image_edit=1.,
                      xtype='image',
                      first_ctype='prompt',
                      second_ctype='prompt',
                      repeat_noise=False, 
                      use_original_steps=False, 
                      noise_dropout=0.,
                      temperature=1.,
                      mixed_ratio=0.5,
                      is_save_intermediate=True,
                      is_save_x0=False,
                      sqrt_one_minus_at=None,
                      a_t=None):

        b, *_, device = *x_edit.shape, self.model.model.diffusion_model.device

        x_gen_in = torch.cat([x_gen] * 3)
        x_edit_in = torch.cat([x_edit] * 3)
        t_gen_in = torch.cat([t_gen] * 3)
        t_edit_in = torch.cat([t_edit] * 3)
        
        e_t_gen_cat, e_t_edit_cat = self.model.apply_model(
            x_gen_in, x_edit_in, t_gen_in, cond_dict, t_edit_in=t_edit_in,
                            is_save_intermediate=is_save_intermediate,
                            is_save_x0=is_save_x0,
                            sqrt_one_minus_at=sqrt_one_minus_at,
                            a_t=a_t)#.chunk(4)

        e_t_uncond_gen, _, e_t_gen_full = e_t_gen_cat.chunk(3)
        e_t_uncond_text_edit, e_t_uncond_image_edit, e_t_edit_full = e_t_edit_cat.chunk(3)

        e_t_gen = e_t_uncond_gen + unconditional_guidance_scale_gen * (e_t_gen_full - e_t_uncond_gen)
        e_t_edit = 0.5 * (e_t_uncond_text_edit + e_t_uncond_image_edit) + \
                        unconditional_guidance_scale_text_edit * (e_t_edit_full - e_t_uncond_text_edit) + \
                        unconditional_guidance_scale_image_edit * (e_t_edit_full - e_t_uncond_image_edit)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        if xtype == 'image':
            extended_shape = (b, 1, 1, 1)
        elif xtype == 'text':
            extended_shape = (b, 1)

        a_t = torch.full(extended_shape, alphas[index], device=device, dtype=x_edit.dtype)
        a_prev = torch.full(extended_shape, alphas_prev[index], device=device, dtype=x_edit.dtype)
        sigma_t = torch.full(extended_shape, sigmas[index], device=device, dtype=x_edit.dtype)
        sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=x_edit.dtype)

        # current prediction for x_0
        pred_x0_gen = (x_gen - sqrt_one_minus_at * e_t_gen) / a_t.sqrt()
        import pdb; pdb.set_trace()
        dir_xt_gen = (1. - a_prev - sigma_t**2).sqrt() * e_t_gen
        noise_gen = sigma_t * noise_like(x_gen, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise_gen = torch.nn.functional.dropout(noise_gen, p=noise_dropout)
        x_prev_gen = a_prev.sqrt() * pred_x0_gen + dir_xt_gen + noise_gen
        
        # offset = (t_edit - t_gen).mean().item() // 20
        # print('offset: ', offset)
        a_t_offset = torch.full(extended_shape, alphas[index+offset], device=device, dtype=x_edit.dtype)
        a_prev_offset = torch.full(extended_shape, alphas_prev[index+offset], device=device, dtype=x_edit.dtype)
        sigma_t_offset = torch.full(extended_shape, sigmas[index+offset], device=device, dtype=x_edit.dtype)
        sqrt_one_minus_at_offset = torch.full(extended_shape, sqrt_one_minus_alphas[index+offset], device=device, dtype=x_edit.dtype)

        pred_x0_edit = (x_edit - sqrt_one_minus_at_offset * e_t_edit) / a_t_offset.sqrt()
        dir_xt_edit = (1. - a_prev_offset - sigma_t_offset**2).sqrt() * e_t_edit
        noise_edit = sigma_t_offset * noise_like(x_edit, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise_edit = torch.nn.functional.dropout(noise_edit, p=noise_dropout)
        x_prev_edit = a_prev_offset.sqrt() * pred_x0_edit + dir_xt_edit + noise_edit

        return x_prev_gen, pred_x0_gen, x_prev_edit, pred_x0_edit

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
              unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
       num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

       assert t_enc <= num_reference_steps
       num_steps = t_enc

       if use_original_steps:
           alphas_next = self.alphas_cumprod[:num_steps]
           alphas = self.alphas_cumprod_prev[:num_steps]
       else:
           alphas_next = self.ddim_alphas[:num_steps]
           alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])
       
       alphas_next = alphas_next.to(x0.device)
       alphas = alphas.to(x0.device)
       x_next = x0
       intermediates = []
       inter_steps = []
       for i in tqdm(range(num_steps), desc='Encoding Image'):
           t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
           if unconditional_guidance_scale == 1.:
               noise_pred = self.model.apply_model(x_next, t, c)
           else:
               assert unconditional_conditioning is not None
               e_t_uncond, noise_pred = torch.chunk(
                   self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                          torch.cat((unconditional_conditioning, c))), 2)
               noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

           xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
           weighted_noise_pred = alphas_next[i].sqrt() * (
                   (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
           x_next = xt_weighted + weighted_noise_pred
           if return_intermediates and i % (
                   num_steps // return_intermediates) == 0 and i < num_steps - 1:
               intermediates.append(x_next)
               inter_steps.append(i)
           elif return_intermediates and i >= num_steps - 2:
               intermediates.append(x_next)
               inter_steps.append(i)
           if callback: callback(i)

       out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
       if return_intermediates:
           out.update({'intermediates': intermediates})
       return x_next, out
    
    
    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas
            
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(t.device)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(t.device)

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None, xtype='image', ctype='vision',
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, xtype=xtype, ctype=ctype, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    def get_al(self, x_dec_gen, index):
        b = x_dec_gen.shape[0]
        extended_shape = (b, 1, 1, 1)
        alphas = self.ddim_alphas
        a_t = torch.full(extended_shape, 1., device=x_dec_gen.device, dtype=x_dec_gen.dtype)
        for kk in range(b): a_t[kk] = alphas[index]
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sqrt_one_minus_at = torch.full(extended_shape, 1., device=x_dec_gen.device, dtype=x_dec_gen.dtype)
        for kk in range(b): sqrt_one_minus_at[kk] = sqrt_one_minus_alphas[index]
        return a_t, sqrt_one_minus_at

    @torch.no_grad()
    def decode_dual(self, x_latent_gen, x_latent_edit, t_start, cond_dict,
               unconditional_guidance_scale_gen=1.0, unconditional_guidance_scale_edit=1.0,
               unconditional_guidance_scale_text_edit=None, unconditional_guidance_scale_image_edit=None,
               delay_t=None, unconditional_conditioning=None, xtype='image', 
               first_ctype='vision', second_ctype='prompt',
               use_original_steps=False, mixed_ratio=0.5, is_save_intermediate=True, 
               is_save_x0=True, callback=None, coarse_spatial_steps=15):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec_gen, x_dec_edit = x_latent_gen.clone(), x_latent_edit.clone()
        x_dec_gen_info, x_dec_edit_info = [], []
        
        sqrt_one_minus_at, a_t = None, None
        ### first round to get a coarse spatial guidance
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent_edit.shape[0],), step, device=x_latent_edit.device, dtype=torch.long)
            a_t, sqrt_one_minus_at = self.get_al(x_dec_gen, index)

            import pdb; pdb.set_trace()
            x_dec_gen, x0_dec_gen, x_dec_edit, x0_dec_edit = self.p_sample_ddim_dual_cfg(
                x_dec_gen, 
                x_dec_edit,
                ts,
                cond_dict,
                index, 
                unconditional_guidance_scale_gen=unconditional_guidance_scale_gen,
                unconditional_guidance_scale_text_edit=unconditional_guidance_scale_text_edit,
                unconditional_guidance_scale_image_edit=unconditional_guidance_scale_image_edit,
                use_original_steps=use_original_steps,
                noise_dropout=0,
                temperature=1,
                mixed_ratio=mixed_ratio,
                is_save_intermediate=is_save_intermediate,
                is_save_x0=is_save_x0,
                sqrt_one_minus_at=sqrt_one_minus_at, 
                a_t=a_t)

            x_dec_gen_info.append(x0_dec_gen)
            x_dec_edit_info.append(x0_dec_edit)
            if i >= coarse_spatial_steps: break
            if callback: callback(i)
        
        # import pdb; pdb.set_trace()

        ### second round to get an edited image
        iterator_2nd = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec_edit = x_latent_edit.clone()
        start_index, start_step = None, None 
        for i, step in enumerate(iterator_2nd):
            if i <= coarse_spatial_steps: continue
            index = total_steps - i - 1
            gen_ts = torch.full((x_latent_edit.shape[0],), step, device=x_latent_edit.device, dtype=torch.long)
            edit_ts = torch.full((x_latent_edit.shape[0],), step+coarse_spatial_steps*20, device=x_latent_edit.device, dtype=torch.long)
            a_t, sqrt_one_minus_at = self.get_al(x_dec_gen, index)

            # cond_dict['noisy_c_concat'] = torch.cat([x0_dec_gen]*3, dim=0) / 0.18215 # be consistent with instructDiffusion
            # print('====', index, coarse_spatial_steps, i, step, '====')
            x_dec_gen, x0_dec_gen, x_dec_edit, x0_dec_edit = self.asyn_p_sample_ddim_dual_cfg(
                x_dec_gen, 
                x_dec_edit,
                gen_ts,
                edit_ts,
                cond_dict,
                index, 
                offset=coarse_spatial_steps,
                unconditional_guidance_scale_gen=unconditional_guidance_scale_gen,
                unconditional_guidance_scale_text_edit=unconditional_guidance_scale_text_edit,
                unconditional_guidance_scale_image_edit=unconditional_guidance_scale_image_edit,
                use_original_steps=use_original_steps,
                noise_dropout=0,
                temperature=1,
                mixed_ratio=mixed_ratio,
                is_save_intermediate=is_save_intermediate,
                is_save_x0=is_save_x0,
                sqrt_one_minus_at=sqrt_one_minus_at, a_t=a_t)
            x_dec_gen_info.append(x0_dec_gen)
            x_dec_edit_info.append(x0_dec_edit)
            
            start_index, start_step = index, step

            if callback: callback(i)

        ### third round to finalize the edited image
        x_dec_gen_ = x_dec_gen.clone()
        for i, step in enumerate(tqdm(range(start_step+coarse_spatial_steps*20-20, -1, -20), desc='Decoding image', total=coarse_spatial_steps)):
            index = start_index - i
            # print(i, step, index)
            # import pdb; pdb.set_trace()
            gen_ts = torch.full((x_latent_edit.shape[0],), start_step, device=x_latent_edit.device, dtype=torch.long)
            edit_ts = torch.full((x_latent_edit.shape[0],), step, device=x_latent_edit.device, dtype=torch.long)
            a_t, sqrt_one_minus_at = self.get_al(x_dec_gen, index)
            # cond_dict['noisy_c_concat'] = torch.cat([x0_dec_gen]*3, dim=0) / 0.18215 # be consistent with instructDiffusion
            # print('====', index, coarse_spatial_steps, i, step, '====')
            x_dec_gen_, x0_dec_gen_, x_dec_edit, x0_dec_edit = self.asyn_p_sample_ddim_dual_cfg(
                x_dec_gen_, 
                x_dec_edit,
                gen_ts,
                edit_ts,
                cond_dict,
                index, 
                offset=coarse_spatial_steps,
                unconditional_guidance_scale_gen=unconditional_guidance_scale_gen,
                unconditional_guidance_scale_text_edit=unconditional_guidance_scale_text_edit,
                unconditional_guidance_scale_image_edit=unconditional_guidance_scale_image_edit,
                use_original_steps=use_original_steps,
                noise_dropout=0,
                temperature=1,
                mixed_ratio=mixed_ratio,
                is_save_intermediate=is_save_intermediate,
                is_save_x0=is_save_x0,
                sqrt_one_minus_at=sqrt_one_minus_at, a_t=a_t)
            x_dec_gen_info.append(x0_dec_gen)
            x_dec_edit_info.append(x0_dec_edit)
            if callback: callback(i)

        if unconditional_guidance_scale_edit is not None:
            return x_dec_gen, x_dec_edit
        else:
            return x_dec_gen, x_dec_edit, x_dec_gen_info, x_dec_edit_info

    @torch.no_grad()
    def decode_dc(self, x_latent, first_conditioning, second_conditioning, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None, xtype='image', first_ctype='vision', second_ctype='prompt',
               use_original_steps=False, mixed_ratio=0.5, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim_dc(
                x_dec, 
                first_conditioning, 
                second_conditioning, 
                ts, index, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                xtype=xtype,
                first_ctype=first_ctype,
                second_ctype=second_ctype,
                use_original_steps=use_original_steps,
                noise_dropout=0,
                temperature=1,
                mixed_ratio=mixed_ratio,)
            if callback: callback(i)
        return x_dec
    
    
def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))