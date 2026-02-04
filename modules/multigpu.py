import os
import functools
import numpy as np

import torch
from torch import Tensor
import torch.distributed as dist
from xfuser.core.distributed import (
    init_distributed_environment,
    get_classifier_free_guidance_world_size, 
    get_classifier_free_guidance_rank,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    get_cfg_group
)

if os.getenv("TORCHELASTIC_RUN_ID") is not None:
    dist.init_process_group("nccl")
    init_distributed_environment(
        rank=dist.get_rank(), 
        world_size=dist.get_world_size()
    )

def parallel_transformer(pipe):
    transformer = pipe.dit
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        img: Tensor,
        landmark_img:Tensor,
        img_ids: Tensor,
        landmark_ids: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        llm_embedding: Tensor,
        t_vec: Tensor,
        mask: Tensor,
    ):  
        txt, y = self.connector(
            llm_embedding, t_vec, mask
        )   # 
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # ---------------------------------------------------------------------
        if (
            isinstance(timesteps, torch.Tensor)
            and timesteps.ndim != 0
            and timesteps.shape[0] == img.shape[0]
        ):
            timesteps = torch.chunk(
                timesteps, get_classifier_free_guidance_world_size(), dim=0
            )[get_classifier_free_guidance_rank()]

            y = torch.chunk(
                y, get_classifier_free_guidance_world_size(), dim=0
            )[get_classifier_free_guidance_rank()]
        # ---------------------------------------------------------------------
        
        img = self.img_in(img) 
        landmark_img = self.landmark_in(landmark_img.permute(0, 2, 1))
        img = torch.cat([img, landmark_img], dim=1)
        vec = self.time_in(self.timestep_embedding(timesteps, 256)) 

        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt) 


        # ---------------------------------------------------------------------
        # img cfg_usp
        img = torch.chunk(
            img, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        img = torch.chunk(
            img, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]
        #print(img.shape)
        # txt cfg_usp
        txt = torch.chunk(
            txt, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        txt = torch.chunk(
            txt, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]

        # pe cfg_usp
        txt_ids = torch.chunk(
            txt_ids, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        txt_ids = torch.chunk(
            txt_ids, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]

        img_ids = torch.chunk(
            img_ids, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        img_ids = torch.chunk(
            img_ids, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]

        landmark_ids = torch.chunk(
            landmark_ids, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        landmark_ids = torch.chunk(
            landmark_ids, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]
        # ---------------------------------------------------------------------

        ids = torch.cat((txt_ids, img_ids), dim=1)  
        pe_ti = self.pe_embedder(ids) 
        pe_lm = self.pe_embedder(landmark_ids)
        pe = torch.cat([pe_ti, pe_lm], dim=2)

        if not self.blocks_to_swap:
            for block in self.double_blocks:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

            img = torch.cat((txt, img), 1) 
            for block in self.single_blocks:
                img = block(img, vec=vec, pe=pe)
        else:
            for block_idx, block in enumerate(self.double_blocks):
                self.offloader_double.wait_for_block(block_idx)
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
                self.offloader_double.submit_move_blocks(self.double_blocks, block_idx)

            img = torch.cat((txt, img), 1)

            for block_idx, block in enumerate(self.single_blocks):
                self.offloader_single.wait_for_block(block_idx)
                img = block(img, vec=vec, pe=pe)
                self.offloader_single.submit_move_blocks(self.single_blocks, block_idx)
        img = img[:, txt.shape[1] :, ...]

        if self.training and self.cpu_offload_checkpointing:
            img = img.to(self.device)
            vec = vec.to(self.device)

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels) 

        # ---------------------------------------------------------------------
        img = get_sp_group().all_gather(img, dim=-2)
        img = get_cfg_group().all_gather(img, dim=0)
        # ---------------------------------------------------------------------
        
        return img

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward
