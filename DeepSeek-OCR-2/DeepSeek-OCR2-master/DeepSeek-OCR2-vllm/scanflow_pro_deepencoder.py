"""ScanFlow Pro V2 dynamic recurrent visual encoder for the vLLM runtime.

It exactly preserves the trained V2 information flow: generate H recurrently
against immutable V, compute R=CrossAttention(V,H,H) once, apply the tokenwise
question-conditioned gate, and return [V + lambda_i(V,Q)R; H].
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import os
import math


from typing import Optional, Tuple

# from megatron.model import LayerNorm

import transformers


from typing import Optional, Tuple, Type
from functools import partial
import math

def scanflow_debug_tensor(name, x):
    if os.getenv("SCANFLOW_DEBUG_TENSORS", "0") != "1":
        return
    if x is None:
        print(f"[scanflow debug] {name}: None", flush=True)
        return

    try:
        t = x.detach()
        finite = torch.isfinite(t) if torch.is_floating_point(t) else torch.ones_like(t, dtype=torch.bool)
        finite_count = int(finite.sum().item())
        total = t.numel()

        msg = (
            f"[scanflow debug] {name}: "
            f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
            f"finite={finite_count}/{total}"
        )

        if torch.is_floating_point(t) and finite_count > 0:
            tf = t[finite]
            msg += (
                f" min={tf.min().item():.6g}"
                f" max={tf.max().item():.6g}"
                f" mean={tf.float().mean().item():.6g}"
            )

        print(msg, flush=True)
    except Exception as exc:
        print(f"[scanflow debug] {name}: failed to summarize: {exc}", flush=True)


def scanflow_is_main_process() -> bool:
    return not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    )


def scanflow_log_interval() -> int:
    return max(1, int(os.getenv("SCANFLOW_DIAGNOSTIC_LOG_INTERVAL", "10")))


def scanflow_env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def scanflow_env_float(name: str, default: str = "0.0") -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def scanflow_log_attention_outputs(
    *,
    forward_step: int,
    visual_attn_maps: Optional[torch.Tensor],
) -> None:
    """Log ScanFlow visual-attention maps used for attention-level fixation supervision."""
    if not scanflow_is_main_process():
        return
    if os.getenv("SCANFLOW_LOG_ATTENTION_OUTPUTS", "1") != "1":
        return
    if forward_step % scanflow_log_interval() != 0:
        return
    if visual_attn_maps is None:
        print(
            f"[scanflow attention outputs] forward_step={forward_step} maps=None",
            flush=True,
        )
        return

    with torch.no_grad():
        maps = visual_attn_maps.detach().float()
        row_sums = maps.sum(dim=-1)
        entropy = -(maps.clamp_min(1e-12) * maps.clamp_min(1e-12).log()).sum(dim=-1)
        max_probs = maps.max(dim=-1).values
        print(
            "[scanflow attention outputs] "
            f"forward_step={forward_step} "
            f"shape={tuple(maps.shape)} "
            f"mean={maps.mean().item():.8g} "
            f"std={maps.std().item():.8g} "
            f"min={maps.min().item():.8g} "
            f"max={maps.max().item():.8g} "
            f"row_sum_mean={row_sums.mean().item():.8g} "
            f"row_sum_min={row_sums.min().item():.8g} "
            f"row_sum_max={row_sums.max().item():.8g} "
            f"entropy_mean={entropy.mean().item():.8g} "
            f"max_prob_mean={max_probs.mean().item():.8g} "
            f"finite={bool(torch.isfinite(maps).all().item())}",
            flush=True,
        )


def scanflow_log_fixation_outputs(
    *,
    forward_step: int,
    fixation_logits: Optional[torch.Tensor],
    scanpath_hidden: Optional[torch.Tensor],
    visual_tokens: Optional[torch.Tensor],
    logit_scale: Optional[torch.Tensor],
) -> None:
    """Log whether the fixation decoder is escaping the uniform solution."""
    if not scanflow_is_main_process():
        return
    if os.getenv("SCANFLOW_LOG_FIXATION_OUTPUTS", "1") != "1":
        return
    if forward_step % scanflow_log_interval() != 0:
        return
    if fixation_logits is None:
        print(
            f"[scanflow fixation outputs] forward_step={forward_step} logits=None",
            flush=True,
        )
        return

    with torch.no_grad():
        logits = fixation_logits.detach().float()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        max_probs, argmax = probs.max(dim=-1)

        message = (
            "[scanflow fixation outputs] "
            f"forward_step={forward_step} "
            f"shape={tuple(logits.shape)} "
            f"logit_mean={logits.mean().item():.8g} "
            f"logit_std={logits.std().item():.8g} "
            f"logit_min={logits.min().item():.8g} "
            f"logit_max={logits.max().item():.8g} "
            f"logit_range={(logits.max() - logits.min()).item():.8g} "
            f"prob_max_mean={max_probs.mean().item():.8g} "
            f"prob_max_min={max_probs.min().item():.8g} "
            f"prob_max_max={max_probs.max().item():.8g} "
            f"prediction_entropy={entropy.mean().item():.8g} "
            f"first_argmax={int(argmax.reshape(-1)[0].item())} "
            f"finite={bool(torch.isfinite(logits).all().item())}"
        )

        if scanpath_hidden is not None:
            hidden = scanpath_hidden.detach().float()
            message += (
                f" hidden_mean={hidden.mean().item():.8g}"
                f" hidden_std={hidden.std().item():.8g}"
                f" hidden_norm={hidden.norm().item():.8g}"
                f" hidden_finite={bool(torch.isfinite(hidden).all().item())}"
            )

        if visual_tokens is not None:
            visual = visual_tokens.detach().float()
            message += (
                f" visual_mean={visual.mean().item():.8g}"
                f" visual_std={visual.std().item():.8g}"
                f" visual_norm={visual.norm().item():.8g}"
            )

        if logit_scale is not None:
            scale = logit_scale.detach().float()
            message += (
                f" logit_scale={scale.mean().item():.8g}"
                f" logit_scale_abs={scale.abs().mean().item():.8g}"
            )

        print(message, flush=True)


class MlpProjector(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)
        
        elif cfg.projector_type == "normlayer_downsample_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            mlp_ratio = cfg.get("mlp_ratio", 1)
            modules = [
                nn.LayerNorm(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio),
                nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio)
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)
        
        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            mlp_ratio = cfg.get("mlp_ratio", 1)
            modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio)]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            self.high_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)
            self.low_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "hybrid_split_feature_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            channel_div = cfg.get("channel_div", 0.5)
            self.high_up_proj = nn.Linear(cfg.input_dim[0], int(cfg.n_embed * channel_div))
            self.low_up_proj = nn.Linear(cfg.input_dim[1], cfg.n_embed - int(cfg.n_embed * channel_div))

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "low_high_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed // 2, cfg.n_embed // 2))
            modules = nn.Sequential(*modules)
            self.high_layers = nn.Sequential(*modules)
            self.low_layers = copy.deepcopy(modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        if cfg.get("token_pooling", False):
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

        if cfg.get("conv_fusion_high_low_features", False):
            self.fusion_layer = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.layers = modules

    def forward(self, x):
        if self.cfg.get("token_pooling", False):
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            # import ipdb; ipdb.set_trace()
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)
        
        if self.cfg.get("conv_fusion_high_low_features", False):
            x = self.fusion_layer(x[:, 0]) + x[:, 1]

        if self.cfg.projector_type == 'low_high_hybrid_split_mlp_gelu':
            high_x, low_x = x[0], x[1]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        
        if self.cfg.projector_type == 'hybrid_split_feature_mlp_gelu':
            high_x = x[...,:self.cfg.input_dim[0]]
            low_x = x[...,self.cfg.input_dim[0]:]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        
        if self.cfg.projector_type == 'low_high_split_mlp_gelu':
            high_x, low_x = x[0], x[1]
            high_x = self.high_layers(high_x)
            low_x = self.low_layers(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
            return x
        
        if self.cfg.projector_type == 'downsample_mlp_gelu' or self.cfg.projector_type == 'normlayer_downsample_mlp_gelu':
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(x, kernel_size=self.cfg.downsample_ratio, stride=self.cfg.downsample_ratio, padding=0) # B, C*4, HW // 4
            x = x.permute(0, 2, 1)
            
        return self.layers(x)

    @staticmethod
    def get_flops_per_sample(cfg):
        if cfg.projector_type == "linear":
            fwd = 2 * cfg.input_dim * cfg.n_embed

        elif "mlp_gelu" in cfg.projector_type :
            mlp_depth = cfg.get("depth", 1)
            downsample_ratio = cfg.get("downsample_ratio", 1)
            input_dim = sum(cfg.input_dim) if isinstance(cfg.input_dim, list) else cfg.input_dim
            input_dim = input_dim * downsample_ratio * downsample_ratio
            fwd = 2 * input_dim * cfg.n_embed + (mlp_depth - 1) * 2 * cfg.n_embed * cfg.n_embed
        else:
            fwd = 0

        return fwd * 3
    

#===================qwen2================================

class CustomQwen2Decoder(nn.Module):
    """Qwen2 visual/reasoning encoder used by recurrent Plan 1.

    token_type_ids:
        0 = bidirectional visual token
        1 = causal recurrent reasoning token

    After every pretrained Qwen decoder layer, question information is injected
    through a separate cross-attention residual on causal positions only. Visual
    positions are left unchanged by the text cross-attention branch.

    Gradient-checkpointing note
    ---------------------------
    The text cross-attention is executed *inside the checkpointed layer function*.
    text_context and token_type_ids are explicit checkpoint inputs, so backward
    recomputation follows exactly the same graph as the original forward. This
    intentionally avoids forward hooks and mutable outer-module "active context"
    state, which are not safe under activation checkpoint recomputation.
    """

    def __init__(
        self,
        decoder_layer: int = 24,
        max_position_embeddings: int = 131072,
        hidden_dimension: int = 896,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
        vocab_size: int = 151936,
        attn_implementation: str = "sdpa",
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        enable_text_cross_attention: bool = True,
    ):
        super().__init__()
        self.enable_text_cross_attention = bool(enable_text_cross_attention)
        self.hidden_dimension = hidden_dimension

        if attn_implementation == "flash_attention_2":
            raise ValueError(
                "CustomQwen2Decoder does not support flash_attention_2; "
                "the asymmetric visual/causal mask requires sdpa or eager."
            )

        Qwen2Model = getattr(transformers.models.qwen2.modeling_qwen2, "Qwen2Model")
        Qwen2Config = getattr(transformers, "Qwen2Config")
        config = Qwen2Config(
            hidden_size=hidden_dimension,
            num_hidden_layers=decoder_layer,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            _attn_implementation=attn_implementation,
        )

        # Preserve the original pretrained Qwen module/state-dict hierarchy.
        self.model = self._create_custom_model(Qwen2Model, config)

        # Keep the exact parameter/state-dict names used by the previous recurrent
        # Plan-1 implementation. Only the execution mechanism changes: these
        # modules are now called explicitly inside the checkpointed layer unit.
        self.text_cross_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dimension) for _ in range(decoder_layer)
        ])
        self.text_cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dimension,
                num_attention_heads,
                dropout=attention_dropout,
                batch_first=True,
            )
            for _ in range(decoder_layer)
        ])

        # inputs_embeds are always supplied by ScanFlow, so the unused embedding
        # table is removed exactly as before.
        del self.model.embed_tokens

    def _apply_text_cross_attention(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        text_context: Optional[torch.Tensor],
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the existing causal-only text residual after one Qwen layer."""
        if not self.enable_text_cross_attention or text_context is None:
            return hidden_states

        causal_mask = token_type_ids.eq(1)
        if causal_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                "token_type_ids must match hidden_states[:2], got "
                f"{tuple(causal_mask.shape)} vs {tuple(hidden_states.shape[:2])}."
            )

        causal_counts = causal_mask.sum(dim=1)
        if int(causal_counts.max().item()) == 0:
            return hidden_states
        if not torch.equal(causal_counts, causal_counts[:1].expand_as(causal_counts)):
            raise ValueError(
                "Recurrent Plan 1 requires the same active causal sequence length "
                "within a single Qwen call."
            )

        batch_size = hidden_states.size(0)
        num_causal = int(causal_counts[0].item())

        # This is intentionally identical to the former forward-hook branch:
        # normalize the complete layer output, query only causal positions, then
        # add the text-attention residual only at those causal positions.
        normed = self.text_cross_norms[layer_idx](hidden_states)
        causal_query = normed[causal_mask].view(
            batch_size, num_causal, hidden_states.size(-1)
        )
        text_context_local = text_context.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        text_residual, _ = self.text_cross_attn_layers[layer_idx](
            causal_query,
            text_context_local,
            text_context_local,
            need_weights=False,
        )

        updated = hidden_states.clone()
        updated[causal_mask] = (
            hidden_states[causal_mask]
            + text_residual.reshape(-1, hidden_states.size(-1))
        )
        return updated

    def _run_qwen_layer_with_text(
        self,
        *,
        layer_idx: int,
        decoder_layer: nn.Module,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values,
        output_attentions: bool,
        use_cache: bool,
        cache_position: torch.Tensor,
        position_embeddings,
        text_context: Optional[torch.Tensor],
        token_type_ids: torch.Tensor,
    ):
        """One architecture-faithful unit: Qwen layer -> text cross-attention."""
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        updated_hidden = self._apply_text_cross_attention(
            layer_idx=layer_idx,
            hidden_states=layer_outputs[0],
            text_context=text_context,
            token_type_ids=token_type_ids,
        )
        return (updated_hidden, *layer_outputs[1:])

    def _create_custom_model(self, Qwen2Model, config):
        class CustomQwen2ModelInner(Qwen2Model):
            """Qwen2Model retaining only ScanFlow's asymmetric-mask helper.

            No recurrent-step state is stored on this module. In particular,
            token_type_ids are passed explicitly by CustomQwen2Decoder.forward.
            """

            def build_scanflow_causal_mask(
                self,
                attention_mask,
                input_tensor,
                token_type_ids,
            ):
                dtype, device = input_tensor.dtype, input_tensor.device
                min_dtype = torch.finfo(dtype).min
                batch_size, sequence_length = input_tensor.shape[:2]

                if token_type_ids is None:
                    raise ValueError(
                        "Recurrent Plan 1 requires token_type_ids for its asymmetric "
                        "visual/causal attention mask."
                    )
                if token_type_ids.shape != (batch_size, sequence_length):
                    raise ValueError(
                        "token_type_ids must have shape [B,S] matching inputs_embeds, got "
                        f"{tuple(token_type_ids.shape)} vs {(batch_size, sequence_length)}."
                    )

                causal_mask = self._create_custom_4d_mask(
                    sequence_length=sequence_length,
                    dtype=dtype,
                    device=device,
                    batch_size=batch_size,
                    token_type_ids=token_type_ids,
                )
                if attention_mask is not None:
                    if attention_mask.dim() != 2:
                        raise ValueError(
                            "Recurrent Plan 1 expects an optional 2D attention_mask, got "
                            f"shape={tuple(attention_mask.shape)}."
                        )
                    padding_mask = attention_mask[:, None, None, :].to(dtype=dtype)
                    padding_mask = (1.0 - padding_mask) * min_dtype
                    causal_mask = causal_mask + padding_mask
                return causal_mask

            def _create_custom_4d_mask(
                self,
                sequence_length,
                dtype,
                device,
                batch_size,
                token_type_ids,
            ):
                min_dtype = torch.finfo(dtype).min
                masks = []
                for b in range(batch_size):
                    mask = torch.full(
                        (sequence_length, sequence_length),
                        fill_value=min_dtype,
                        dtype=dtype,
                        device=device,
                    )
                    type_ids = token_type_ids[b]
                    visual_positions = (type_ids == 0).nonzero(as_tuple=True)[0]
                    causal_positions = (type_ids == 1).nonzero(as_tuple=True)[0]

                    # Visual tokens remain a bidirectional visual context.
                    if len(visual_positions) > 0:
                        mask[visual_positions[:, None], visual_positions] = 0.0

                    # Recurrent reasoning tokens see all visual tokens and only
                    # themselves / earlier causal reasoning states.
                    for i, causal_pos in enumerate(causal_positions):
                        if len(visual_positions) > 0:
                            mask[causal_pos, visual_positions] = 0.0
                        mask[causal_pos, causal_positions[: i + 1]] = 0.0

                    masks.append(mask)
                return torch.stack(masks, dim=0).unsqueeze(1)

        return CustomQwen2ModelInner(config)

    def forward(
        self,
        inputs_embeds,
        token_type_ids,
        attention_mask=None,
        text_context: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run Qwen with checkpoint-safe per-layer text injection.

        This mirrors the layer loop of transformers==4.46.3 Qwen2Model.forward,
        but replaces the checkpoint target from a bare Qwen decoder layer with
        the architecture-faithful composite operation:

            QwenDecoderLayer -> causal-only text cross-attention residual

        Crucially, text_context and token_type_ids are explicit inputs to the
        checkpointed function. They are therefore available unchanged during
        backward recomputation.
        """
        from transformers.modeling_outputs import BaseModelOutputWithPast

        output_attentions = kwargs.pop("output_attentions", None)
        output_hidden_states = kwargs.pop("output_hidden_states", None)
        return_dict = kwargs.pop("return_dict", None)
        use_cache = kwargs.pop("use_cache", None)
        position_ids = kwargs.pop("position_ids", None)
        cache_position = kwargs.pop("cache_position", None)
        past_key_values = kwargs.pop("past_key_values", None)

        if kwargs:
            raise TypeError(
                "Unexpected CustomQwen2Decoder.forward keyword arguments: "
                f"{sorted(kwargs.keys())}"
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.model.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self.model.config.use_cache

        # Recurrent Plan 1 always calls this encoder with use_cache=False. Keeping
        # this explicit also avoids silently constructing a cache whose key length
        # would not match the custom asymmetric [S,S] mask.
        if use_cache or past_key_values is not None:
            raise ValueError(
                "Recurrent Plan-1 CustomQwen2Decoder does not use KV caching. "
                "Call it with use_cache=False and past_key_values=None."
            )

        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided for recurrent Plan 1.")
        if token_type_ids is None:
            raise ValueError("token_type_ids must be provided for recurrent Plan 1.")

        batch_size, sequence_length = inputs_embeds.shape[:2]
        if token_type_ids.shape != (batch_size, sequence_length):
            raise ValueError(
                "token_type_ids must match inputs_embeds[:2], got "
                f"{tuple(token_type_ids.shape)} vs {(batch_size, sequence_length)}."
            )

        if cache_position is None:
            cache_position = torch.arange(
                0,
                sequence_length,
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self.model.build_scanflow_causal_mask(
            attention_mask=attention_mask,
            input_tensor=inputs_embeds,
            token_type_ids=token_type_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        gradient_checkpointing = bool(
            getattr(self.model, "gradient_checkpointing", False) and self.training
        )
        capture_layer_indices = getattr(
            self,
            "scanflow_attention_layer_indices",
            None,
        )

        for layer_idx, decoder_layer in enumerate(self.model.layers):
            layer_output_attentions = bool(output_attentions) and (
                capture_layer_indices is None
                or layer_idx in capture_layer_indices
            )
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if gradient_checkpointing:
                # Do NOT checkpoint decoder_layer.__call__ alone. The text branch
                # is part of the layer-level computation and must be recomputed
                # together with it. The tensors that vary by recurrent step are
                # explicit checkpoint inputs rather than mutable module state.
                def checkpointed_layer(
                    layer_hidden_states,
                    layer_causal_mask,
                    layer_position_ids,
                    layer_cache_position,
                    layer_position_embeddings,
                    layer_text_context,
                    layer_token_type_ids,
                    *,
                    _layer_idx=layer_idx,
                    _decoder_layer=decoder_layer,
                    _output_attentions=layer_output_attentions,
                ):
                    return self._run_qwen_layer_with_text(
                        layer_idx=_layer_idx,
                        decoder_layer=_decoder_layer,
                        hidden_states=layer_hidden_states,
                        causal_mask=layer_causal_mask,
                        position_ids=layer_position_ids,
                        past_key_values=None,
                        output_attentions=_output_attentions,
                        use_cache=False,
                        cache_position=layer_cache_position,
                        position_embeddings=layer_position_embeddings,
                        text_context=layer_text_context,
                        token_type_ids=layer_token_type_ids,
                    )

                # text_context can legitimately be None, but recurrent Plan 1
                # normally provides it. For the no-text case use an empty tensor
                # as an explicit checkpoint input and map it back to None inside
                # the closure so PyTorch receives only stable positional args.
                if text_context is None:
                    checkpoint_text_context = hidden_states.new_empty(
                        batch_size, 0, self.hidden_dimension
                    )
                    text_context_is_none = True
                else:
                    checkpoint_text_context = text_context
                    text_context_is_none = False

                if text_context_is_none:
                    def checkpointed_layer_no_text(
                        layer_hidden_states,
                        layer_causal_mask,
                        layer_position_ids,
                        layer_cache_position,
                        layer_position_embeddings,
                        _checkpoint_text_context,
                        layer_token_type_ids,
                        *,
                        _layer_idx=layer_idx,
                        _decoder_layer=decoder_layer,
                        _output_attentions=layer_output_attentions,
                    ):
                        return self._run_qwen_layer_with_text(
                            layer_idx=_layer_idx,
                            decoder_layer=_decoder_layer,
                            hidden_states=layer_hidden_states,
                            causal_mask=layer_causal_mask,
                            position_ids=layer_position_ids,
                            past_key_values=None,
                            output_attentions=_output_attentions,
                            use_cache=False,
                            cache_position=layer_cache_position,
                            position_embeddings=layer_position_embeddings,
                            text_context=None,
                            token_type_ids=layer_token_type_ids,
                        )
                    checkpoint_fn = checkpointed_layer_no_text
                else:
                    checkpoint_fn = checkpointed_layer

                layer_outputs = self.model._gradient_checkpointing_func(
                    checkpoint_fn,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    cache_position,
                    position_embeddings,
                    checkpoint_text_context,
                    token_type_ids,
                )
            else:
                layer_outputs = self._run_qwen_layer_with_text(
                    layer_idx=layer_idx,
                    decoder_layer=decoder_layer,
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    output_attentions=layer_output_attentions,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    text_context=text_context,
                    token_type_ids=token_type_ids,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                # Preserve layer indexing while retaining attention tensors only
                # for the four recurrent Plan-1 supervision layers.
                all_self_attns += (
                    layer_outputs[1] if layer_output_attentions else None,
                )

        hidden_states = self.model.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in [hidden_states, None, all_hidden_states, all_self_attns]
                if value is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

# batch_size = 2
# inputs_embeds = torch.randn(batch_size, 512, 896).cuda()

# inputs_embeds = torch.randn(batch_size, 512, 896).cuda()
# token_type_ids = torch.cat([
#     torch.zeros(batch_size, 256, dtype=torch.long),
#     torch.ones(batch_size, 256, dtype=torch.long),
# ], dim=1).cuda()

# # start = time.time()
# with torch.no_grad():
#     outputs_sdpa = decoder_sdpa(inputs_embeds, token_type_ids)
#     print(outputs_sdpa[0].shape)
# print(f"SDPA time: {time.time() - start:.4f}s")



class Qwen2Decoder2Encoder(nn.Module):
    """ScanFlow Pro V2 dynamic spatial/intensity residual baseline.

    The checkpoint-compatible Plan-1 recurrent path that generates h_t is kept
    intact:

        h_0 = mean(V)
        h_t = Qwen(V, h_1, ..., h_{t-1}, seed_t)[-1]
        seed_1 = h_0
        seed_t = h_{t-1}  (t > 1)

    V2 does not change visual perception inside the recurrent loop. Only after
    the complete temporal reasoning sequence H=[h_1,...,h_T] has been generated,
    the immutable visual tokens query H once to obtain a global reasoning
    residual:

        R_i = CrossAttention(query=V, key=H, value=H)_i
        lambda_i = sigmoid(F_I([v_i, pool(V), pool(Q)]))
        V_v2_i = V_i + lambda_i * R_i

    The returned sequence is [V_v2; H]. This intentionally has no stepwise
    perceptual update, fixation-controlled write, or feedback from R/lambda into
    later h_t states. Its tokenwise intensity is dynamic across samples and
    questions, giving V2 spatial and reasoning-intensity axes while retaining no
    temporal perception-update axis.
    """

    def __init__(
        self,
        decoder_layer: int,
        hidden_dimension: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        max_query: int,
        text_hidden_dim: int = 1280,
        max_scanpath_steps: int = 32,
        scanpath_layers: int = 2,      # retained only for checkpoint/config compatibility
        scanpath_dropout: float = 0.0, # retained only for checkpoint/config compatibility
    ):
        super().__init__()
        del max_query, scanpath_layers, scanpath_dropout
        self.hidden_dimension = hidden_dimension
        self.text_hidden_dim = text_hidden_dim
        self.max_scanpath_steps = max_scanpath_steps

        self.enable_attention_supervision = (
            scanflow_env_flag("DEEPSEEK_OCR2_ENABLE_ATTENTION_SUPERVISION", "0")
            or scanflow_env_float("DEEPSEEK_OCR2_ATTENTION_LOSS_WEIGHT", "0.0") != 0.0
        )

        # Recurrent Plan-1 attention supervision:
        # supervise the four central Qwen layers. For a 24-layer encoder these
        # are 0-based layers [10, 11, 12, 13]. An explicit comma-separated
        # override is still available for controlled ablations.
        raw_attention_layers = os.getenv(
            "DEEPSEEK_OCR2_ATTENTION_LAYER_INDICES", ""
        ).strip()
        if raw_attention_layers:
            attention_layer_indices = [
                int(item.strip())
                for item in raw_attention_layers.split(",")
                if item.strip()
            ]
        else:
            if decoder_layer < 4:
                raise ValueError(
                    "Middle-four attention supervision requires at least 4 Qwen layers, "
                    f"got decoder_layer={decoder_layer}."
                )
            middle_start = (decoder_layer - 4) // 2
            attention_layer_indices = list(range(middle_start, middle_start + 4))

        if len(attention_layer_indices) != 4:
            raise ValueError(
                "DEEPSEEK_OCR2_ATTENTION_LAYER_INDICES must contain exactly four "
                f"layer indices, got {attention_layer_indices}."
            )
        if len(set(attention_layer_indices)) != 4:
            raise ValueError(
                "Attention supervision layer indices must be unique, got "
                f"{attention_layer_indices}."
            )
        for layer_idx in attention_layer_indices:
            if layer_idx < 0 or layer_idx >= decoder_layer:
                raise ValueError(
                    f"Attention supervision layer index {layer_idx} is out of range "
                    f"for decoder_layer={decoder_layer}."
                )

        self.attention_supervision_layer_indices = tuple(attention_layer_indices)

        # Keep eager attention for exact compatibility with the trained recurrent
        # Plan-1 attention path and its middle-four-layer supervision.
        qwen_attn_implementation = "eager"
        self.model = CustomQwen2Decoder(
            decoder_layer=decoder_layer,
            hidden_dimension=hidden_dimension,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            attn_implementation=qwen_attn_implementation,
            enable_text_cross_attention=True,
        )
        self.model.scanflow_attention_layer_indices = (
            self.attention_supervision_layer_indices
        )
        self.text_context_proj = nn.Linear(text_hidden_dim, hidden_dimension)

        # Version-2 global residual branch. It is registered under ``self.model``
        # so the existing MS-Swift trainable prefix
        # ``model.qwen2_model.model`` includes it without changing the selection
        # used by recurrent Plan-1 checkpoints.
        self.model.scanflow_v2_visual_norm = nn.LayerNorm(hidden_dimension)
        self.model.scanflow_v2_reasoning_norm = nn.LayerNorm(hidden_dimension)
        self.model.scanflow_v2_residual_attn = nn.MultiheadAttention(
            hidden_dimension,
            num_attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        # Tokenwise reasoning intensity lambda_i(V,Q). Local V_i supplies the
        # spatial axis, while pooled V and pooled question tokens supply global
        # image/question context. Register under ``self.model`` so the existing
        # MS-Swift trainable prefix includes every newly introduced parameter.
        self.model.scanflow_v2_intensity_visual_norm = nn.LayerNorm(
            hidden_dimension
        )
        self.model.scanflow_v2_intensity_question_norm = nn.LayerNorm(
            hidden_dimension
        )
        self.model.scanflow_v2_intensity_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, 1),
        )
        self.scanflow_v2_initial_intensity = scanflow_env_float(
            "SCANFLOW_V2_INITIAL_INTENSITY", "0.001"
        )
        if not 0.0 < self.scanflow_v2_initial_intensity < 1.0:
            raise ValueError(
                "SCANFLOW_V2_INITIAL_INTENSITY must be strictly between 0 and 1, "
                f"got {self.scanflow_v2_initial_intensity}."
            )
        self._reset_scanflow_v2_residual_branch()

        # Compatibility aliases for outer code. They no longer indicate an
        # external ScanFlow module; recurrence is intrinsic to this encoder.
        self.enable_scanflow = True
        self.enable_text_query_residual = True

        self.last_scanpath_hidden = None
        self.last_reasoning_residual = None
        self.last_reasoning_intensity = None
        self.last_residual_visual_tokens = None
        self.last_scanflow_output = None
        self.last_visual_attn_maps = None
        self.last_visual_attn_layer_indices = self.attention_supervision_layer_indices
        self.last_scanpath_mask = None
        self._scanflow_diagnostic_forward_step = 0

    def _reset_scanflow_v2_residual_branch(self):
        """Reset only the new R=CrossAttn(V,H,H) and lambda_i(V,Q) branch."""
        visual_norm = self.model.scanflow_v2_visual_norm
        reasoning_norm = self.model.scanflow_v2_reasoning_norm
        residual_attn = self.model.scanflow_v2_residual_attn
        intensity_visual_norm = self.model.scanflow_v2_intensity_visual_norm
        intensity_question_norm = self.model.scanflow_v2_intensity_question_norm
        intensity_mlp = self.model.scanflow_v2_intensity_mlp

        with torch.no_grad():
            nn.init.ones_(visual_norm.weight)
            nn.init.zeros_(visual_norm.bias)
            nn.init.ones_(reasoning_norm.weight)
            nn.init.zeros_(reasoning_norm.bias)
            nn.init.xavier_uniform_(residual_attn.in_proj_weight)
            if residual_attn.in_proj_bias is not None:
                nn.init.zeros_(residual_attn.in_proj_bias)
            nn.init.xavier_uniform_(residual_attn.out_proj.weight)
            if residual_attn.out_proj.bias is not None:
                nn.init.zeros_(residual_attn.out_proj.bias)
            nn.init.ones_(intensity_visual_norm.weight)
            nn.init.zeros_(intensity_visual_norm.bias)
            nn.init.ones_(intensity_question_norm.weight)
            nn.init.zeros_(intensity_question_norm.bias)
            nn.init.xavier_uniform_(intensity_mlp[0].weight)
            nn.init.zeros_(intensity_mlp[0].bias)

            # A zero output weight makes every token's initial gate exactly the
            # requested checkpoint-safe value. The output layer learns on the
            # first optimizer step; gradients reach the first layer once that
            # output weight becomes nonzero.
            nn.init.zeros_(intensity_mlp[2].weight)
            initial_logit = math.log(
                self.scanflow_v2_initial_intensity
                / (1.0 - self.scanflow_v2_initial_intensity)
            )
            nn.init.constant_(intensity_mlp[2].bias, initial_logit)

    def reset_scanflow_parameters(self):
        """Reset only the NEW recurrent-Plan-1 text-injection parameters.

        This is the original Plan-1 reset path and is intentionally kept
        independent from the ScanFlow Pro V2 reset path.
        """
        def _reset_module(module):
            for m in module.modules():
                if isinstance(m, nn.MultiheadAttention):
                    nn.init.xavier_uniform_(m.in_proj_weight)
                    if m.in_proj_bias is not None:
                        nn.init.zeros_(m.in_proj_bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    if m.weight is not None:
                        nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        _reset_module(self.model.text_cross_norms)
        _reset_module(self.model.text_cross_attn_layers)
        nn.init.xavier_uniform_(self.text_context_proj.weight)
        if self.text_context_proj.bias is not None:
            nn.init.zeros_(self.text_context_proj.bias)

        if scanflow_is_main_process():
            print(
                "[scanflow reset] recurrent Plan-1 new parameters reinitialized: "
                "text_cross_norms, text_cross_attn_layers, text_context_proj; "
                "pretrained Qwen weights preserved.",
                flush=True,
            )

    def reset_scanflow_v2_parameters(self):
        """Reset only modules newly introduced by ScanFlow Pro V2."""
        self._reset_scanflow_v2_residual_branch()

        if scanflow_is_main_process():
            print(
                "[scanflow v2 reset] global reasoning-residual cross-attention "
                "and dynamic lambda_i(V,Q) head reinitialized with "
                f"lambda_i={self.scanflow_v2_initial_intensity:g}; all recurrent Plan-1 "
                "checkpoint parameters preserved.",
                flush=True,
            )

    def _compute_global_reasoning_residual(
        self,
        visual_tokens: torch.Tensor,
        scanpath_hidden: torch.Tensor,
        scanpath_padding_mask: Optional[torch.Tensor],
        question_tokens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return R, lambda_i(V,Q), and V+lambda_i*R after recurrence.

        The cross-attention is intentionally evaluated once, outside the
        recurrence. Consequently R cannot affect the production of H and this
        baseline contains no stepwise perceptual feedback.
        """
        query = self.model.scanflow_v2_visual_norm(visual_tokens)
        reasoning = self.model.scanflow_v2_reasoning_norm(scanpath_hidden)
        residual, _ = self.model.scanflow_v2_residual_attn(
            query=query,
            key=reasoning,
            value=reasoning,
            key_padding_mask=scanpath_padding_mask,
            need_weights=False,
        )
        residual = residual.to(dtype=visual_tokens.dtype)

        intensity_visual = self.model.scanflow_v2_intensity_visual_norm(
            visual_tokens
        )
        pooled_visual = intensity_visual.mean(dim=1, keepdim=True).expand(
            -1, visual_tokens.size(1), -1
        )
        if question_tokens is None or question_tokens.size(1) == 0:
            pooled_question = torch.zeros_like(pooled_visual)
        else:
            normalized_question = self.model.scanflow_v2_intensity_question_norm(
                question_tokens
            )
            pooled_question = normalized_question.mean(dim=1, keepdim=True).expand(
                -1, visual_tokens.size(1), -1
            )

        intensity_features = torch.cat(
            [intensity_visual, pooled_visual, pooled_question], dim=-1
        )
        reasoning_intensity = torch.sigmoid(
            self.model.scanflow_v2_intensity_mlp(intensity_features)
        ).to(dtype=visual_tokens.dtype)
        residual_visual_tokens = visual_tokens + reasoning_intensity * residual
        return residual, reasoning_intensity, residual_visual_tokens

    def _project_text_tokens(self, text_context, device, dtype):
        if text_context is None:
            return None
        text_context = text_context.to(device=device)
        if text_context.dim() == 2:
            text_context = text_context.unsqueeze(1)
        elif text_context.dim() != 3:
            raise ValueError(
                f"text_context must have shape (B,D) or (B,L,D), got {tuple(text_context.shape)}."
            )
        if text_context.size(-1) == self.text_hidden_dim:
            text_context = self.text_context_proj(text_context)
        elif text_context.size(-1) != self.hidden_dimension:
            raise ValueError(
                f"Unsupported text_context hidden size={text_context.size(-1)}; "
                f"expected {self.text_hidden_dim} or {self.hidden_dimension}."
            )
        return text_context.to(dtype=dtype)

    def _resolve_scanpath_steps(self, scanpath_lengths, scanpath_steps, device):
        if scanpath_steps is not None:
            steps = int(scanpath_steps)
        elif scanpath_lengths is not None:
            if not torch.is_tensor(scanpath_lengths):
                scanpath_lengths = torch.tensor(scanpath_lengths, device=device)
            steps = int(scanpath_lengths.to(device=device).max().item())
        else:
            steps = self.max_scanpath_steps
        steps = max(1, steps)
        if steps > self.max_scanpath_steps:
            raise ValueError(
                f"scanpath_steps={steps} exceeds max_scanpath_steps={self.max_scanpath_steps}."
            )
        return steps

    def _build_scanpath_padding_mask(self, scanpath_lengths, batch_size, steps, device):
        if scanpath_lengths is None:
            return None
        if not torch.is_tensor(scanpath_lengths):
            scanpath_lengths = torch.tensor(scanpath_lengths, device=device)
        lengths = scanpath_lengths.to(device=device, dtype=torch.long).clamp(min=1, max=steps)
        step_ids = torch.arange(steps, device=device).unsqueeze(0)
        return step_ids.expand(batch_size, -1) >= lengths.unsqueeze(1)

    @staticmethod
    def _extract_last_query_visual_attention(
        outputs,
        n_visual: int,
        layer_indices,
    ):
        """Return selected-layer [B,L_sel,H,N_visual] attention for newest state.

        Qwen still has to compute attention weights for the forward when
        output_attentions=True, but recurrent Plan 1 retains only the four
        supervised middle layers across timesteps instead of retaining all
        decoder-layer attention tensors in the temporal supervision graph.
        """
        attentions = getattr(outputs, "attentions", None)
        if attentions is None:
            return None

        per_layer = []
        for layer_idx in layer_indices:
            layer_attention = attentions[layer_idx]
            if layer_attention is None:
                return None
            # Qwen attention: [B,H,Q,K]. The newest recurrent state is the final
            # query position; visual tokens occupy keys [0:n_visual].
            per_layer.append(layer_attention[:, :, -1, :n_visual])

        # list([B,H,N]) -> [B,L_sel,H,N]
        return torch.stack(per_layer, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
        scanpath_lengths: Optional[torch.Tensor] = None,
        scanpath_steps: Optional[int] = None,
        collect_visual_attention: Optional[bool] = None,
    ) -> torch.Tensor:
        """Generate H recurrently, then apply the dynamic one-shot V2 residual.

        x is the SAM feature map. Returned shape is [B,N+T,D] and is ordered as
        [V + lambda_i(V,Q)*R; H]. ``last_scanpath_hidden`` retains the checkpoint-
        compatible [B,T,D] reasoning sequence.
        """
        visual_tokens = x.flatten(2).transpose(1, 2)
        batch_size, n_visual, _ = visual_tokens.shape
        device, dtype = visual_tokens.device, visual_tokens.dtype
        text_tokens = self._project_text_tokens(text_context, device, dtype)
        num_steps = self._resolve_scanpath_steps(scanpath_lengths, scanpath_steps, device)
        padding_mask = self._build_scanpath_padding_mask(
            scanpath_lengths, batch_size, num_steps, device
        )

        # No learned causal query/token bank. The only bootstrap state is image
        # conditioned; every later seed is the actually generated previous state.
        prev_hidden = visual_tokens.mean(dim=1, keepdim=True)
        generated_states = []
        visual_attn_steps = []
        collect_attn = (
            self.enable_attention_supervision
            if collect_visual_attention is None
            else bool(collect_visual_attention)
        )

        for step_idx in range(num_steps):
            history = (
                torch.cat(generated_states, dim=1)
                if generated_states
                else visual_tokens.new_zeros(batch_size, 0, self.hidden_dimension)
            )
            current_seed = prev_hidden
            causal_inputs = torch.cat([history, current_seed], dim=1)
            n_causal = causal_inputs.size(1)
            # Preserve the original recurrent Plan-1 information flow exactly:
            # every h_t is generated against the same immutable V. The Version-2
            # residual is deliberately absent from this temporal loop.
            combined = torch.cat([visual_tokens, causal_inputs], dim=1)
            token_type_ids = torch.cat([
                torch.zeros(batch_size, n_visual, dtype=torch.long, device=device),
                torch.ones(batch_size, n_causal, dtype=torch.long, device=device),
            ], dim=1)

            outputs = self.model(
                combined,
                token_type_ids,
                text_context=text_tokens,
                output_attentions=collect_attn,
                return_dict=True,
                use_cache=False,
            )
            current_hidden = outputs.last_hidden_state[:, -1:, :]
            generated_states.append(current_hidden)
            prev_hidden = current_hidden

            if collect_attn:
                step_visual_attn = self._extract_last_query_visual_attention(
                    outputs,
                    n_visual=n_visual,
                    layer_indices=self.attention_supervision_layer_indices,
                )
                if step_visual_attn is None:
                    raise RuntimeError(
                        "ScanFlow Pro V2 attention supervision was requested, but "
                        "Qwen returned no visual self-attention weights."
                    )
                visual_attn_steps.append(step_visual_attn)

            scanflow_debug_tensor(f"recurrent_plan1_step_{step_idx}_hidden", current_hidden)

        scanpath_hidden = torch.cat(generated_states, dim=1)
        if padding_mask is not None:
            scanpath_hidden = scanpath_hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if visual_attn_steps:
            # Only the four supervised middle layers are retained:
            # list of [B,L_sel,H,N] -> [B,L_sel,T,H,N], L_sel=4.
            visual_attn_maps = torch.stack(visual_attn_steps, dim=2)
            if padding_mask is not None:
                visual_attn_maps = visual_attn_maps.masked_fill(
                    padding_mask[:, None, :, None, None], 0.0
                )
        else:
            visual_attn_maps = None

        reasoning_residual, reasoning_intensity, residual_visual_tokens = (
            self._compute_global_reasoning_residual(
                visual_tokens=visual_tokens,
                scanpath_hidden=scanpath_hidden,
                scanpath_padding_mask=padding_mask,
                # Do not let the new V2 loss path update the trained Plan-1 text
                # projection. The gate still observes its question features;
                # Plan-1 continues to train through its unchanged recurrent path.
                question_tokens=(
                    text_tokens.detach() if text_tokens is not None else None
                ),
            )
        )
        scanflow_output = torch.cat(
            [residual_visual_tokens, scanpath_hidden], dim=1
        )

        self.last_scanpath_hidden = scanpath_hidden
        self.last_reasoning_residual = reasoning_residual
        self.last_reasoning_intensity = reasoning_intensity
        self.last_residual_visual_tokens = residual_visual_tokens
        self.last_scanflow_output = scanflow_output
        self.last_visual_attn_maps = visual_attn_maps
        self.last_scanpath_mask = padding_mask
        self._scanflow_diagnostic_forward_step += 1
        scanflow_log_attention_outputs(
            forward_step=self._scanflow_diagnostic_forward_step,
            visual_attn_maps=visual_attn_maps,
        )
        scanflow_debug_tensor("recurrent_plan1_hidden", scanpath_hidden)
        scanflow_debug_tensor("scanflow_v2_reasoning_residual", reasoning_residual)
        scanflow_debug_tensor("scanflow_v2_reasoning_intensity", reasoning_intensity)
        scanflow_debug_tensor("scanflow_v2_residual_visual_tokens", residual_visual_tokens)
        scanflow_debug_tensor("scanflow_v2_output", scanflow_output)
        return scanflow_output



def build_qwen2_decoder_as_encoder(
    decoder_layer=24,
    hidden_dimension=896,
    num_attention_heads=14,
    num_key_value_heads=2,
    intermediate_size=4864,
    max_query = 400,
    text_hidden_dim=1280,
    max_scanpath_steps=32,
    scanpath_layers=2,
    scanpath_dropout=0.0,
    checkpoint=None,
):

    decoder_as_encoder = Qwen2Decoder2Encoder(
            decoder_layer=decoder_layer,
            hidden_dimension = hidden_dimension,
            num_attention_heads = num_attention_heads,
            num_key_value_heads = num_key_value_heads,
            intermediate_size = intermediate_size,
            max_query = max_query,
            text_hidden_dim=text_hidden_dim,
            max_scanpath_steps=max_scanpath_steps,
            scanpath_layers=scanpath_layers,
            scanpath_dropout=scanpath_dropout,
        )



    
    if checkpoint is not None:
        # with open(checkpoint, "rb") as f:
        state_dict = torch.load(checkpoint)

        load_result = decoder_as_encoder.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(
                "Missing keys when loading Qwen2Decoder2Encoder "
                f"({len(load_result.missing_keys)}): {load_result.missing_keys[:20]}"
            )
        if load_result.unexpected_keys:
            print(
                "Unexpected keys when loading Qwen2Decoder2Encoder "
                f"({len(load_result.unexpected_keys)}): {load_result.unexpected_keys[:20]}"
            )
        # tob
        print(checkpoint)
    return decoder_as_encoder




#=========================Sam-Vary=================================


def get_abs_pos_sam(abs_pos, tgt_size):

    dtype = abs_pos.dtype

    src_size = abs_pos.size(1)

    if src_size != tgt_size:
        old_pos_embed = abs_pos.permute(0, 3, 1, 2)
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode='bicubic',
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        return new_pos_embed
    else:
        return abs_pos




class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 896, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # x = x + self.pos_embed
            x = x + get_abs_pos_sam(self.pos_embed, x.size(1))

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        x2 = self.net_2(x)
        x3 = self.net_3(x2.clone())

        return x3


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = add_decomposed_rel_pos(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)

        if self.use_rel_pos:
            rel_h = rel_h.view(B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3))
            rel_w = rel_w.view(B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3))
            attn_bias = (rel_h + rel_w).view(B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4))
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            # x = _attention_rel_h_rel_w(q, k, v, rel_h, rel_w)
        else:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        dtype = rel_pos.dtype
        rel_pos = rel_pos.to(torch.float32)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).to(dtype)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_h = rel_h.unsqueeze(-1)
    rel_w = rel_w.unsqueeze(-2)
    rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
    rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)

    return rel_h, rel_w


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.proj(x)
    #     # B C H W -> B H W C
    #     x = x.permute(0, 2, 3, 1)
    #     return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.proj.weight.dtype
        x = x.to(device=self.proj.weight.device, dtype=dtype)
        if self.proj.bias is not None and self.proj.bias.dtype != dtype:
            self.proj.bias.data = self.proj.bias.data.to(dtype)
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_sam_fast_vit_b(checkpoint=None, compile_mode='max-autotune', dtype=torch.bfloat16):
    image_encoder = build_sam_vit_b(checkpoint).eval().to(dtype)
    # sam = _apply_eval_dtype_sam(sam, dtype)
    image_encoder = torch.compile(image_encoder, mode=compile_mode)
    return image_encoder


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    image_encoder.eval()
    if checkpoint is not None:
        # with open(checkpoint, "rb") as f:
        state_dict = torch.load(checkpoint)
        # print(state_dict.keys())
        # for key in state_dict:
        # image_encoder.load_state_dict({k[14:]: v for k, v in state_dict.items() if 'image_encoder' in k}, strict=False)
        # ocr-anyting
        # image_encoder.load_state_dict(state_dict, strict=True)
        # tob
        image_encoder.load_state_dict({k[30:]: v for k, v in state_dict.items() if 'vision_tower_high' in k}, strict=True)
        print(checkpoint)
    return image_encoder
