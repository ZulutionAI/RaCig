# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time
class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        masks=None,
        use_bg_prompt=False,
        use_bg_img=False,
        multi_ref_num=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            #print(self.attn_map.shape)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class IPAttnProcessor2_0_RP(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, 
                 hidden_size, 
                 cross_attention_dim=None, 
                 scale=1.0, 
                 num_tokens=4,
                 height=1344,
                 width=768,
                 ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.height = height
        self.width=width

    def split_dims(self, xs, height, width,):
        def repeat_div(x,y):
            while y > 0:
                x = math.ceil(x / 2)
                y = y - 1
            return x
        scale = math.ceil(math.log2(math.sqrt(height * width / xs)))
        dsh = repeat_div(height,scale)
        dsw = repeat_div(width,scale)

        return dsh, dsw

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        masks=None, # (num_people, 2(face+body), num_samples*2, h, w) or (num_people, 1(whole), num_samples*2, h, w)
        use_bg_prompt=False,
        use_bg_img=False,
        multi_ref_num=None, # multi_ref_num: [character_1_ref_num, character_2_ref_num, 1] (1 for bg)
        *args,
        **kwargs,
    ):  

        xs = hidden_states.size()[1]
        origin_bsz = hidden_states.size()[0]
        use_body_ref = masks.shape[1] == 2 # Each people has two ref mask

        (dsh,dsw) = self.split_dims(xs, self.height, self.width)

        num_ref_face = max(multi_ref_num) if multi_ref_num is not None else 1
        instance_len = 77 + self.num_tokens*(num_ref_face + int(use_body_ref))
 
        num_instance = (encoder_hidden_states.shape[1]) // instance_len 

        if num_instance == 2 and (use_bg_img or use_bg_prompt): # Makse sure Oner character or two character
            bg_index = 1
        elif num_instance == 3: # This means there must be 2 character + 1 background
            bg_index = 2

        encoder_hidden_states = torch.cat(encoder_hidden_states.chunk(num_instance, dim=1)) 
        hidden_states = hidden_states.repeat(num_instance, 1, 1)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        #################### Begin Attention ##########################
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens * (num_ref_face + int(use_body_ref))

            encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Turn to (batch_size, num_heads, h, w, head_dim)
        hidden_states = hidden_states.reshape(batch_size, attn.heads, dsh, dsw, head_dim)
        
        # For ip-adapter
        """
        ip_hidden_states.shape = [(character_num+bg)*(2*num_samples), head, seq_len, dim]
        seq_len = 16*(num_face_ref + use_body_ref)
        """

        # Prepare masks
        seg_select = F.interpolate(masks.view(masks.shape[0]*masks.shape[1], *masks.shape[2:]), size=(dsh, dsw), mode='nearest')
        seg_select = seg_select.view(
            len(masks), masks.shape[1], batch_size // num_instance,1,dsh,dsw,1).expand(
                len(masks), masks.shape[1], batch_size // num_instance, attn.heads, dsh, dsw, head_dim).to(device=hidden_states.device, dtype=hidden_states.dtype)
        
        seg_bg = 1 - (masks.sum(1).sum(0))
        seg_select_bg = F.interpolate(seg_bg.unsqueeze(1), size=(dsh, dsw), mode='nearest')
        seg_select_bg = seg_select_bg.view(
            batch_size // num_instance,1,dsh,dsw,1).expand(
                batch_size // num_instance, attn.heads, dsh, dsw, head_dim).to(device=hidden_states.device, dtype=hidden_states.dtype)

        hidden_states_final = torch.zeros((origin_bsz, *hidden_states.shape[1:]), device=hidden_states.device, dtype=hidden_states.dtype)
        if use_body_ref:
            # ip_hidden_states: [num_instance*2*num_samples, 16*num_face_ref + 16(use_body_ref), dim]
            ip_hidden_states_part_1, ip_hidden_states_part_2 = ip_hidden_states[:,:self.num_tokens*(num_ref_face), :], ip_hidden_states[:,self.num_tokens*(num_ref_face):, :]
            ip_hidden_states_parts = [] # [(num_instance*2*num_samples, 16*num_face_ref, dim), (num_instance*2*num_samples, 16, dim)]
            for i, ip_hidden_states in enumerate([ip_hidden_states_part_1, ip_hidden_states_part_2]):
                ip_key = self.to_k_ip(ip_hidden_states)
                ip_value = self.to_v_ip(ip_hidden_states)
                
                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                if i == 0:
                    attn_mask_ip = torch.zeros((origin_bsz, num_instance, attn.heads, dsh*dsw, self.num_tokens*num_ref_face), device=ip_key.device, dtype=ip_key.dtype)
                    for sample_idx in range(num_instance):
                        attn_mask_ip[:,sample_idx,:,:,multi_ref_num[sample_idx]*self.num_tokens:] = 1
                    attn_mask_ip = attn_mask_ip.view(origin_bsz*num_instance, attn.heads,  dsh*dsw, self.num_tokens*num_ref_face)
                else:
                    attn_mask_ip = None

                ip_hidden_states = F.scaled_dot_product_attention(
                    query, ip_key, ip_value, attn_mask=attn_mask_ip, dropout_p=0.0, is_causal=False
                )
                ip_hidden_states = ip_hidden_states.reshape(batch_size, attn.heads, dsh, dsw, head_dim)
                ip_hidden_states_parts.append(ip_hidden_states)

            for i, (hidden_states_chunked, ip_hidden_states_part_1, ip_hidden_states_part2) in enumerate(zip(
                hidden_states.chunk(num_instance, dim=0), 
                ip_hidden_states_parts[0].chunk(num_instance, dim=0), 
                ip_hidden_states_parts[1].chunk(num_instance, dim=0))):
                if i == bg_index:
                    if use_bg_img and use_bg_prompt:
                        hidden_states_chunked = hidden_states_chunked + self.scale * ip_hidden_states_part_1
                    else:
                        hidden_states_chunked = hidden_states_chunked if use_bg_prompt else self.scale * ip_hidden_states_part_1
                    hidden_states_chunked = hidden_states_chunked * seg_select_bg
                else:
                    hidden_states_chunked = (hidden_states_chunked + self.scale * ip_hidden_states_part_1) * seg_select[i,0,...] + \
                        (hidden_states_chunked + self.scale * ip_hidden_states_part2) * seg_select[i,1,...]

                hidden_states_final += hidden_states_chunked
        else:
            # ip_hidden_states: [num_instance*2*num_samples, head, 16*num_face_ref + 16(use_body_ref), dim]
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            attn_mask_ip = torch.zeros((origin_bsz, num_instance, attn.heads, dsh*dsw, self.num_tokens*num_ref_face), device=ip_key.device, dtype=ip_key.dtype)
            for sample_idx in range(num_instance):
                attn_mask_ip[:,sample_idx,:,:,multi_ref_num[sample_idx]*self.num_tokens:] = 1
            attn_mask_ip = attn_mask_ip.view(origin_bsz*num_instance, attn.heads,  dsh*dsw, self.num_tokens*num_ref_face)

            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=attn_mask_ip, dropout_p=0.0, is_causal=False
            )

            ip_hidden_states = ip_hidden_states.reshape(batch_size, attn.heads, dsh, dsw, head_dim)

            for i, (hidden_states_chunked, ip_hidden_states_chunked) in enumerate(zip(hidden_states.chunk(num_instance, dim=0), ip_hidden_states.chunk(num_instance, dim=0))):
                if i == bg_index:
                    if use_bg_img and use_bg_prompt:
                        hidden_states_chunked = hidden_states_chunked + self.scale * ip_hidden_states_chunked
                    else:
                        hidden_states_chunked = hidden_states_chunked if use_bg_prompt else self.scale * ip_hidden_states_chunked
                    hidden_states_chunked = hidden_states_chunked * seg_select_bg
                else:
                    hidden_states_chunked = hidden_states_chunked + self.scale * ip_hidden_states_chunked
                    hidden_states_chunked = hidden_states_chunked * seg_select[i,0,...]
                hidden_states_final += hidden_states_chunked
            
        batch_size = origin_bsz
        hidden_states = hidden_states_final.view(batch_size, attn.heads, dsh*dsw, head_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class IPAttnProcessor2_0_RP_Double_Adapter(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, 
                 hidden_size, 
                 cross_attention_dim=None, 
                 scale=1.0, 
                 num_tokens=4,
                 height=1344,
                 width=768,
                 ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.to_k_ip_body = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip_body = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.height = height
        self.width=width

    def split_dims(self, xs, height, width,):
        def repeat_div(x,y):
            while y > 0:
                x = math.ceil(x / 2)
                y = y - 1
            return x
        scale = math.ceil(math.log2(math.sqrt(height * width / xs)))
        dsh = repeat_div(height,scale)
        dsw = repeat_div(width,scale)

        return dsh, dsw

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        masks=None, # 有且仅有两种输入情况：1. 脸和衣服分开参考（用两个Adapter）：(num_people, 2(face+body), num_samples*2, h, w) 2.脸和衣服一起参考 (num_people, 1(whole), num_samples*2, h, w)
        use_bg_prompt=False,
        use_bg_img=False,
        multi_ref_num=None, # Inference: multi_ref_num（脸部参考图像数量，最后一个是背景）: [character_1_ref_num, character_2_ref_num, 1] (1 for bg)
        *args,
        **kwargs,
    ):  

        xs = hidden_states.size()[1]
        origin_bsz = hidden_states.size()[0] # 记录原始的batch_size
        use_body_ref = masks.shape[1] == 2 # Each people has two ref mask

        (dsh,dsw) = self.split_dims(xs, self.height, self.width)

        num_ref_face = max(multi_ref_num) if multi_ref_num is not None else 1
        instance_len = 77 + self.num_tokens * (num_ref_face + int(use_body_ref)) # 这个是注入的token数量 77是text prompt embedding，后面是stack起来的image embedding （n*16）
 
        num_instance = (encoder_hidden_states.shape[1]) // instance_len # 不管有没有bg prompt或者bg img，num_instance都是num_character + 1
        
        # 当num_instance==2时有两种情况，如果有背景，则说明只有一个character，bg_index=1。如果没背景则有两个character，不用管bg_index
        if num_instance == 2 and (use_bg_img or use_bg_prompt): # Makse sure Oner character or two character
            bg_index = 1
        else: # This means there must be 2 character + 1 background
            bg_index = 2

        # 这里encoder_hidden_states沿着seq_len维度拆开，然后在batch维度拼接起来：那么characters和bg在batch size维度将平行存在, 为了batch运算
        encoder_hidden_states = torch.cat(encoder_hidden_states.chunk(num_instance, dim=1))
        # 同理为了计算则需要复制三份 hidden_states
        hidden_states = hidden_states.repeat(num_instance, 1, 1)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        #################### 开始做 Attention ##########################
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # 此处把encoder_hidden_states的seq_len维度拆开成77（text embedding）和剩下的img embedding的token
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens * (num_ref_face + int(use_body_ref))

            encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Turn to (batch_size, num_heads, h, w, head_dim)
        hidden_states = hidden_states.reshape(batch_size, attn.heads, dsh, dsw, head_dim)
        
        # 开始做 ip-adapter 的 attention
        """
        ip_hidden_states.shape = [(character_num+bg)*(2*num_samples), head, seq_len, dim]
        seq_len = 16*(num_face_ref + use_body_ref)
        """

        # 准备 masks 用于分别对每个character取头和身体的mask
        seg_select = F.interpolate(masks.view(masks.shape[0]*masks.shape[1], *masks.shape[2:]), size=(dsh, dsw), mode='nearest')
        seg_select = seg_select.view(
            len(masks), masks.shape[1], batch_size // num_instance,1,dsh,dsw,1).expand(
                len(masks), masks.shape[1], batch_size // num_instance, attn.heads, dsh, dsw, head_dim).to(device=hidden_states.device, dtype=hidden_states.dtype)
        
        seg_bg = 1 - (masks.sum(1).sum(0))
        seg_select_bg = F.interpolate(seg_bg.unsqueeze(1), size=(dsh, dsw), mode='nearest')
        seg_select_bg = seg_select_bg.view(
            batch_size // num_instance,1,dsh,dsw,1).expand(
                batch_size // num_instance, attn.heads, dsh, dsw, head_dim).to(device=hidden_states.device, dtype=hidden_states.dtype)

        # hidden_states_final 作为最后ip-adapter的相加
        hidden_states_final = torch.zeros((origin_bsz, *hidden_states.shape[1:]), device=hidden_states.device, dtype=hidden_states.dtype)

        # ip_hidden_states 沿着seq_len维度拆开成头部和身体两个部分
        # ip_hidden_states: [num_instance*2*num_samples, 16*num_face_ref + 16(use_body_ref), dim]
        ip_hidden_states_part_1, ip_hidden_states_part_2 = ip_hidden_states[:,:self.num_tokens*(num_ref_face), :], ip_hidden_states[:,self.num_tokens*(num_ref_face):, :]
        ip_hidden_states_parts = [] # [(num_instance*2*num_samples, 16*num_face_ref, dim), (num_instance*2*num_samples, 16, dim)]
        
        # 1. 头部和身体的adapter需要分开计算，这里先计算attention之后再融合
        # Here we calculate ip_hidden_states of different part: face and body
        for i, ip_hidden_states in enumerate([ip_hidden_states_part_1, ip_hidden_states_part_2]):
            if i == 0: # 如果是头部的部分则使用to_xx_ip的adapter
                ip_key = self.to_k_ip(ip_hidden_states)
                ip_value = self.to_v_ip(ip_hidden_states)
            else: # 如果是身体的部分则使用to_xx_ip_body的adapter
                ip_key = self.to_k_ip_body(ip_hidden_states)
                ip_value = self.to_v_ip_body(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if i == 0:
                # 如果是头部的部分，因为做了padding，需要把padding的部分mask掉
                attn_mask_ip = torch.zeros((origin_bsz, num_instance, attn.heads, dsh*dsw, self.num_tokens*num_ref_face), device=ip_key.device, dtype=ip_key.dtype)
                for sample_idx in range(num_instance):
                    attn_mask_ip[:,sample_idx,:,:,multi_ref_num[sample_idx]*self.num_tokens:] = 1
                attn_mask_ip = attn_mask_ip.view(origin_bsz*num_instance, attn.heads,  dsh*dsw, self.num_tokens*num_ref_face)
            else:
                attn_mask_ip = None

            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=attn_mask_ip, dropout_p=0.0, is_causal=False
            )
            ip_hidden_states = ip_hidden_states.reshape(batch_size, attn.heads, dsh, dsw, head_dim)
            ip_hidden_states_parts.append(ip_hidden_states)

        # 2. 这里对计算的得到的attention进行融合，按照mask
        # Here we compose ip_hidden_states of different part: face and body
        # hidden_states_chunked: text对应的attention
        # ip_hidden_states_part_1: 脸部对应的attention
        # ip_hidden_states_part2: 身体对应的attention
        for i, (hidden_states_chunked, ip_hidden_states_part_1, ip_hidden_states_part2) in enumerate(zip(
            hidden_states.chunk(num_instance, dim=0), 
            ip_hidden_states_parts[0].chunk(num_instance, dim=0), 
            ip_hidden_states_parts[1].chunk(num_instance, dim=0))):
            if i == bg_index: # 处理背景，共4种情况：是否有bg text，是否有bg img，以及是否同时有和无
                if use_bg_img and use_bg_prompt:
                    hidden_states_chunked = hidden_states_chunked + self.scale * ip_hidden_states_part_1
                else:
                    hidden_states_chunked = hidden_states_chunked if use_bg_prompt else self.scale * ip_hidden_states_part_1
                hidden_states_chunked = hidden_states_chunked * seg_select_bg
            else: # 头部的ip attention 乘 头部的mask， 身体的ip attention 乘 身体的mask
                hidden_states_chunked = (hidden_states_chunked + self.scale * ip_hidden_states_part_1) * seg_select[i,0,...] + \
                    (hidden_states_chunked + self.scale * ip_hidden_states_part2) * seg_select[i,1,...]
                # seg_select第一维指subject的id，第二维指分别头和身体的mask，第三维为batch_size
            hidden_states_final += hidden_states_chunked

        batch_size = origin_bsz
        hidden_states = hidden_states_final.view(batch_size, attn.heads, dsh*dsw, head_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

## for controlnet
class CNAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, num_tokens=4):
        self.num_tokens = num_tokens

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs,):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class CNAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, num_tokens=4):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.num_tokens = num_tokens

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = 77
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
