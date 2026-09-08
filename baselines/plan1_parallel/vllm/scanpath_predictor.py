from __future__ import annotations

"""
scanpath_predictor.py

Simple one-layer VLM-style scanpath predictor for ScanFlow pretraining.

The model receives a contextualized visual-language prefix from the existing
VLM. During teacher forcing, previous fixation targets are converted into
image-specific visual evidence tokens:

    e_t = sum_i target[t, i] * visual_token[i]

A single causal TransformerEncoderLayer produces scanpath hidden states.
Each hidden state is compared directly with the current image's actual visual
embeddings to produce the fixation distribution.

There is no learned fixation codebook, no fixation-step embedding, no learned
scanpath-start vector, and no separate multi-layer fixation decoder.

The original ``forward()`` remains teacher-forced for predictor pretraining.
``predict_autoregressive_from_prefix()`` is the frozen rollout path used by
Parallel Plan 1: it receives no ground-truth fixation sequence and recursively
feeds back its own soft fixation predictions as visual evidence.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_fixation_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Soft-target cross-entropy averaged per sample."""
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must match, got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}."
        )

    if valid_mask.shape != logits.shape[:2]:
        raise ValueError(
            f"valid_mask must have shape {tuple(logits.shape[:2])}, "
            f"got {tuple(valid_mask.shape)}."
        )

    targets = targets.float()
    targets = targets / targets.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)

    log_probs = F.log_softmax(
        logits.float(),
        dim=-1,
    )

    step_loss = -(
        targets * log_probs
    ).sum(dim=-1)

    valid_weights = valid_mask.float()

    sample_loss = (
        step_loss * valid_weights
    ).sum(dim=1) / valid_weights.sum(
        dim=1
    ).clamp_min(1.0)

    return sample_loss.mean()


class VLMScanpathPredictor(nn.Module):
    """
    One-layer decoder-only scanpath predictor.

    Teacher-forced sequence:
        [contextualized visual-language prefix,
         target-routed evidence for fixation 1,
         ...,
         target-routed evidence for fixation T-1]

    Prediction convention:
        last valid prefix state -> fixation 1
        evidence-1 state        -> fixation 2
        evidence-2 state        -> fixation 3

    Fixation logits:
        cosine(hidden_t, visual_i) / temperature
    """

    def __init__(
        self,
        prefix_hidden_dim: int,
        scanpath_hidden_dim: int,
        visual_dim: int,
        num_heads: int,
        intermediate_dim: Optional[int] = None,
        num_fixation_cells: int = 256,
        dropout: float = 0.0,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()

        if scanpath_hidden_dim % num_heads != 0:
            raise ValueError(
                f"scanpath_hidden_dim={scanpath_hidden_dim} must be "
                f"divisible by num_heads={num_heads}."
            )

        if intermediate_dim is None:
            intermediate_dim = 4 * scanpath_hidden_dim

        if temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {temperature}."
            )

        self.prefix_hidden_dim = int(prefix_hidden_dim)
        self.scanpath_hidden_dim = int(scanpath_hidden_dim)
        self.visual_dim = int(visual_dim)
        self.num_fixation_cells = int(num_fixation_cells)
        self.temperature = float(temperature)

        self.prefix_projection = (
            nn.Identity()
            if prefix_hidden_dim == scanpath_hidden_dim
            else nn.Linear(
                prefix_hidden_dim,
                scanpath_hidden_dim,
                bias=False,
            )
        )

        self.visual_projection = (
            nn.Identity()
            if visual_dim == scanpath_hidden_dim
            else nn.Linear(
                visual_dim,
                scanpath_hidden_dim,
                bias=False,
            )
        )

        self.scanpath_block = nn.TransformerEncoderLayer(
            d_model=scanpath_hidden_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.output_norm = nn.LayerNorm(
            scanpath_hidden_dim,
            elementwise_affine=False,
        )
    
    def reset_parameters(self) -> None:
        """
        Explicitly initialize all trainable predictor parameters.

        Use this for the first predictor-pretraining run when loading a
        checkpoint that does not yet contain scanpath_predictor weights.
        """
        with torch.no_grad():
            if isinstance(self.prefix_projection, nn.Linear):
                nn.init.xavier_uniform_(
                    self.prefix_projection.weight
                )

            if isinstance(self.visual_projection, nn.Linear):
                nn.init.xavier_uniform_(
                    self.visual_projection.weight
                )

            block = self.scanpath_block

            # Multi-head self-attention.
            nn.init.xavier_uniform_(
                block.self_attn.in_proj_weight
            )

            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(
                    block.self_attn.in_proj_bias
                )

            nn.init.xavier_uniform_(
                block.self_attn.out_proj.weight
            )

            if block.self_attn.out_proj.bias is not None:
                nn.init.zeros_(
                    block.self_attn.out_proj.bias
                )

            # Feed-forward network.
            nn.init.xavier_uniform_(
                block.linear1.weight
            )
            nn.init.zeros_(
                block.linear1.bias
            )

            nn.init.xavier_uniform_(
                block.linear2.weight
            )
            nn.init.zeros_(
                block.linear2.bias
            )

            # Transformer layer norms.
            nn.init.ones_(
                block.norm1.weight
            )
            nn.init.zeros_(
                block.norm1.bias
            )

            nn.init.ones_(
                block.norm2.weight
            )
            nn.init.zeros_(
                block.norm2.bias
            )

    def check_parameter_finiteness(self) -> None:
        """Raise immediately if any predictor parameter contains NaN or Inf."""
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if parameter.is_meta:
                    raise RuntimeError(
                        f"Predictor parameter {name} is still on the meta device."
                    )
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(
                        f"Non-finite values detected in predictor parameter: {name}"
                    )

    def log_parameter_statistics(self) -> None:
        """Print initialization statistics without changing model behavior."""
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                values = parameter.detach().float()
                finite = torch.isfinite(values)
                finite_count = int(finite.sum().item())
                total_count = values.numel()

                message = (
                    "[scanpath predictor parameter] "
                    f"name={name} "
                    f"shape={tuple(parameter.shape)} "
                    f"dtype={parameter.dtype} "
                    f"finite={finite_count}/{total_count}"
                )

                if finite_count > 0:
                    finite_values = values[finite]
                    message += (
                        f" mean={finite_values.mean().item():.8g}"
                        f" std={finite_values.std(unbiased=False).item():.8g}"
                        f" min={finite_values.min().item():.8g}"
                        f" max={finite_values.max().item():.8g}"
                        f" norm={finite_values.norm().item():.8g}"
                    )

                print(message, flush=True)

                if finite_count != total_count:
                    raise FloatingPointError(
                        f"Non-finite values detected in predictor parameter: {name}"
                    )

    @staticmethod
    def _causal_mask(
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """True entries are blocked."""
        return torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    @staticmethod
    def _normalize_fixation_targets(
        fixation_targets: torch.Tensor,
    ) -> torch.Tensor:
        targets = fixation_targets.float()
        return targets / targets.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

    def project_visual_tokens(
        self,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape [B, N, visual_dim]."
            )

        if visual_tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                f"Expected visual_dim={self.visual_dim}, got "
                f"{visual_tokens.shape[-1]}."
            )

        return self.visual_projection(
            visual_tokens
        )

    def build_teacher_forced_evidence(
        self,
        fixation_targets: torch.Tensor,
        projected_visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return T-1 weighted visual-evidence tokens.

        e_t = sum_i target[t, i] * projected_visual_token[i]
        """
        if fixation_targets.shape[0] != (
            projected_visual_tokens.shape[0]
        ):
            raise ValueError(
                "fixation_targets and visual tokens must share batch size."
            )

        if fixation_targets.shape[-1] != (
            projected_visual_tokens.shape[1]
        ):
            raise ValueError(
                "Fixation-cell count must equal visual-token count."
            )

        targets = self._normalize_fixation_targets(
            fixation_targets
        )

        previous_targets = targets[:, :-1]

        evidence = torch.einsum(
            "btn,bnd->btd",
            previous_targets,
            projected_visual_tokens.float(),
        )

        return evidence.to(
            projected_visual_tokens.dtype
        )

    def compute_fixation_logits(
        self,
        prediction_states: torch.Tensor,
        projected_visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Direct cosine similarity with actual image visual embeddings.
        """
        normalized_states = self.output_norm(
            prediction_states
        )

        queries = F.normalize(
            normalized_states.float(),
            dim=-1,
        )

        keys = F.normalize(
            projected_visual_tokens.float(),
            dim=-1,
        )

        if queries.ndim == 2:
            logits = torch.einsum(
                "bd,bnd->bn",
                queries,
                keys,
            )
        elif queries.ndim == 3:
            logits = torch.einsum(
                "btd,bnd->btn",
                queries,
                keys,
            )
        else:
            raise ValueError(
                "prediction_states must have shape [B, D] or [B, T, D]."
            )

        return logits / self.temperature

    def forward(
        self,
        prefix_hidden_states: torch.Tensor,
        prefix_valid_mask: torch.Tensor,
        prefix_lengths: torch.Tensor,
        visual_tokens: torch.Tensor,
        fixation_targets: torch.Tensor,
        fixation_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher-forced forward pass.

        Shapes:
            prefix_hidden_states: [B, P, prefix_hidden_dim]
            prefix_valid_mask:    [B, P]
            prefix_lengths:       [B]
            visual_tokens:        [B, 256, visual_dim]
            fixation_targets:     [B, T, 256]
            fixation_mask:        [B, T]
        """
        if prefix_hidden_states.ndim != 3:
            raise ValueError(
                "prefix_hidden_states must have shape [B, P, D]."
            )

        if prefix_valid_mask.shape != prefix_hidden_states.shape[:2]:
            raise ValueError(
                "prefix_valid_mask must match [B, P]."
            )

        if prefix_lengths.shape != (
            prefix_hidden_states.shape[0],
        ):
            raise ValueError(
                "prefix_lengths must have shape [B]."
            )

        if fixation_targets.ndim != 3:
            raise ValueError(
                "fixation_targets must have shape [B, T, N]."
            )

        if fixation_mask.shape != fixation_targets.shape[:2]:
            raise ValueError(
                "fixation_mask must match [B, T]."
            )

        if fixation_targets.shape[-1] != self.num_fixation_cells:
            raise ValueError(
                f"Expected {self.num_fixation_cells} fixation cells, got "
                f"{fixation_targets.shape[-1]}."
            )

        fixation_mask = fixation_mask.bool()
        prefix_valid_mask = prefix_valid_mask.bool()

        batch_size, max_prefix_length, _ = (
            prefix_hidden_states.shape
        )
        _, num_steps, _ = fixation_targets.shape

        prefix_states = self.prefix_projection(
            prefix_hidden_states
        )

        projected_visual_tokens = self.project_visual_tokens(
            visual_tokens
        )

        teacher_evidence = self.build_teacher_forced_evidence(
            fixation_targets=fixation_targets,
            projected_visual_tokens=projected_visual_tokens,
        )

        sequence = torch.cat(
            [
                prefix_states,
                teacher_evidence,
            ],
            dim=1,
        )

        sequence_valid_mask = torch.cat(
            [
                prefix_valid_mask,
                fixation_mask[:, :-1],
            ],
            dim=1,
        )

        causal_mask = self._causal_mask(
            sequence_length=sequence.shape[1],
            device=sequence.device,
        )

        transformed_states = self.scanpath_block(
            src=sequence,
            src_mask=causal_mask,
            src_key_padding_mask=~sequence_valid_mask,
        )

        batch_indices = torch.arange(
            batch_size,
            device=sequence.device,
        )

        first_prediction_indices = (
            prefix_lengths.long() - 1
        ).clamp_min(0)

        first_prediction_state = transformed_states[
            batch_indices,
            first_prediction_indices,
        ]

        later_prediction_states = transformed_states[
            :,
            max_prefix_length:,
        ]

        prediction_states = torch.cat(
            [
                first_prediction_state.unsqueeze(1),
                later_prediction_states,
            ],
            dim=1,
        )

        if prediction_states.shape[1] != num_steps:
            raise RuntimeError(
                f"Expected {num_steps} prediction states, got "
                f"{prediction_states.shape[1]}."
            )

        fixation_logits = self.compute_fixation_logits(
            prediction_states=prediction_states,
            projected_visual_tokens=projected_visual_tokens,
        )

        fixation_loss = soft_fixation_cross_entropy(
            logits=fixation_logits,
            targets=fixation_targets,
            valid_mask=fixation_mask,
        )

        return {
            "loss": fixation_loss,
            "fixation_loss": fixation_loss,
            "fixation_logits": fixation_logits,
            "fixation_probs": torch.softmax(
                fixation_logits.float(),
                dim=-1,
            ),
            "prediction_states": prediction_states,
        }

    def predict_autoregressive_from_prefix(
        self,
        prefix_hidden_states: torch.Tensor,
        prefix_valid_mask: torch.Tensor,
        prefix_lengths: torch.Tensor,
        visual_tokens: torch.Tensor,
        max_steps: int,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressively decode a scanpath from a contextual prefix.

        This is the frozen-inference path used by Parallel Plan 1. Unlike
        ``forward()``, it NEVER receives ground-truth fixation targets and does
        not construct teacher-forced evidence.

        The causal prefix is the only contextual/reasoning input. ``visual_tokens``
        are used as the fixed image-specific spatial basis for two operations:

          1. convert each predicted fixation distribution into the next evidence
             token, e_t = sum_i p_t[i] * v_i;
          2. score each prediction state against the 256 spatial visual keys.

        No ``torch.no_grad()`` and no detach are used internally. When this module
        is frozen by the outer model, gradients still propagate through the full
        autoregressive rollout into ``prefix_hidden_states``.

        Args:
            prefix_hidden_states: [B, P, prefix_hidden_dim]
            prefix_valid_mask:    [B, P]
            prefix_lengths:       [B]
            visual_tokens:        [B, 256, visual_dim]
            max_steps:            number of fixation steps to generate

        Returns:
            fixation_logits:   [B, max_steps, 256]
            fixation_probs:    [B, max_steps, 256]
            prediction_states: [B, max_steps, scanpath_hidden_dim]
            predicted_evidence:[B, max_steps, scanpath_hidden_dim]
        """
        if prefix_hidden_states.ndim != 3:
            raise ValueError(
                "prefix_hidden_states must have shape [B, P, D]."
            )
        if prefix_valid_mask.shape != prefix_hidden_states.shape[:2]:
            raise ValueError(
                "prefix_valid_mask must match prefix_hidden_states[:2]."
            )
        if prefix_lengths.shape != (prefix_hidden_states.shape[0],):
            raise ValueError(
                "prefix_lengths must have shape [B]."
            )
        if int(max_steps) < 1:
            raise ValueError(
                f"max_steps must be positive, got {max_steps}."
            )

        prefix_valid_mask = prefix_valid_mask.bool()
        max_steps = int(max_steps)
        batch_size, max_prefix_length, _ = prefix_hidden_states.shape

        prefix_lengths = prefix_lengths.to(
            device=prefix_hidden_states.device,
            dtype=torch.long,
        )
        if torch.any(prefix_lengths < 1):
            raise ValueError("Every prefix length must be at least 1.")
        if torch.any(prefix_lengths > max_prefix_length):
            raise ValueError(
                "prefix_lengths cannot exceed the padded prefix length."
            )

        # Both projections are frozen during Parallel Plan-1 training, but the
        # prefix projection remains differentiable with respect to its input.
        prefix_states = self.prefix_projection(prefix_hidden_states)
        projected_visual_tokens = self.project_visual_tokens(visual_tokens)

        sequence = prefix_states
        sequence_valid_mask = prefix_valid_mask

        batch_indices = torch.arange(
            batch_size,
            device=prefix_hidden_states.device,
        )
        first_prediction_indices = (prefix_lengths - 1).clamp_min(0)

        logits_per_step = []
        probs_per_step = []
        states_per_step = []
        evidence_per_step = []

        for step_idx in range(max_steps):
            causal_mask = self._causal_mask(
                sequence_length=sequence.size(1),
                device=sequence.device,
            )
            transformed_states = self.scanpath_block(
                src=sequence,
                src_mask=causal_mask,
                src_key_padding_mask=~sequence_valid_mask,
            )

            if step_idx == 0:
                # Same convention as teacher-forced pretraining: the final valid
                # prefix state predicts fixation 1.
                prediction_state = transformed_states[
                    batch_indices,
                    first_prediction_indices,
                ]
            else:
                # The most recently appended self-generated evidence token predicts
                # the next fixation.
                prediction_state = transformed_states[:, -1, :]

            fixation_logits = self.compute_fixation_logits(
                prediction_states=prediction_state,
                projected_visual_tokens=projected_visual_tokens,
            )
            fixation_probs = torch.softmax(
                fixation_logits.float(),
                dim=-1,
            )

            # Soft autoregressive evidence. Crucially this uses the predictor's
            # OWN p_t, never the ground-truth fixation target. Keep it attached so
            # later-step losses backpropagate through earlier predicted steps.
            predicted_evidence = torch.einsum(
                "bn,bnd->bd",
                fixation_probs,
                projected_visual_tokens.float(),
            ).to(projected_visual_tokens.dtype)

            logits_per_step.append(fixation_logits)
            probs_per_step.append(fixation_probs)
            states_per_step.append(prediction_state)
            evidence_per_step.append(predicted_evidence)

            # No need to append evidence after the final requested prediction.
            if step_idx + 1 < max_steps:
                sequence = torch.cat(
                    [sequence, predicted_evidence.unsqueeze(1)],
                    dim=1,
                )
                new_valid = torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=sequence.device,
                )
                sequence_valid_mask = torch.cat(
                    [sequence_valid_mask, new_valid],
                    dim=1,
                )

        return {
            "fixation_logits": torch.stack(logits_per_step, dim=1),
            "fixation_probs": torch.stack(probs_per_step, dim=1),
            "prediction_states": torch.stack(states_per_step, dim=1),
            "predicted_evidence": torch.stack(evidence_per_step, dim=1),
        }

    def predict_one_step(
        self,
        scanflow_hidden_state: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        One-step use for the later frozen ScanFlow fixation probe.

        Freeze scanpath_block, visual_projection, and output_norm before
        ScanFlow training. Do not use torch.no_grad(), because gradients
        still need to reach scanflow_hidden_state.
        """
        squeeze_step_dimension = False

        if scanflow_hidden_state.ndim == 2:
            scanflow_hidden_state = (
                scanflow_hidden_state.unsqueeze(1)
            )
            squeeze_step_dimension = True
        elif scanflow_hidden_state.ndim != 3:
            raise ValueError(
                "scanflow_hidden_state must have shape [B, D] "
                "or [B, 1, D]."
            )

        if scanflow_hidden_state.shape[1] != 1:
            raise ValueError(
                "predict_one_step expects one hidden state per sample."
            )

        if scanflow_hidden_state.shape[-1] != (
            self.scanpath_hidden_dim
        ):
            raise ValueError(
                f"Expected hidden dimension "
                f"{self.scanpath_hidden_dim}, got "
                f"{scanflow_hidden_state.shape[-1]}."
            )

        valid_mask = torch.ones(
            scanflow_hidden_state.shape[:2],
            dtype=torch.bool,
            device=scanflow_hidden_state.device,
        )

        transformed_state = self.scanpath_block(
            src=scanflow_hidden_state,
            src_mask=self._causal_mask(
                sequence_length=1,
                device=scanflow_hidden_state.device,
            ),
            src_key_padding_mask=~valid_mask,
        )

        projected_visual_tokens = self.project_visual_tokens(
            visual_tokens
        )

        fixation_logits = self.compute_fixation_logits(
            prediction_states=transformed_state,
            projected_visual_tokens=projected_visual_tokens,
        )

        fixation_probs = torch.softmax(
            fixation_logits.float(),
            dim=-1,
        )

        if squeeze_step_dimension:
            fixation_logits = fixation_logits[:, 0]
            fixation_probs = fixation_probs[:, 0]
            transformed_state = transformed_state[:, 0]

        return {
            "fixation_logits": fixation_logits,
            "fixation_probs": fixation_probs,
            "probe_hidden_state": transformed_state,
        }

    def freeze_for_scanflow(self) -> None:
        """
        Freeze the pretrained one-step probe components.

        prefix_projection is only used during standalone predictor training.
        """
        self.scanpath_block.requires_grad_(False)
        self.visual_projection.requires_grad_(False)
        self.output_norm.requires_grad_(False)

        self.scanpath_block.eval()
        self.visual_projection.eval()
        self.output_norm.eval()