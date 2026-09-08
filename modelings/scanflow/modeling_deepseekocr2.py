from .modeling_deepseekv2 import DeepseekV2Model, DeepseekV2ForCausalLM
from .configuration_deepseek_v2 import DeepseekV2Config
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
import requests
from PIL import Image, ImageOps, ImageDraw, ImageFont
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
import os
from .deepencoderv2 import build_sam_vit_b, build_qwen2_decoder_as_encoder, MlpProjector
from addict import Dict
from transformers import TextStreamer
from .conversation import get_conv_template
from abc import ABC
import math
import re
from tqdm import tqdm
import numpy as np
# import time

def scanflow_debug_tensor(name, x):
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


def load_image(image_path):

    try:
        image = Image.open(image_path)
        
        corrected_image = ImageOps.exif_transpose(image)
        
        return corrected_image
        
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    # pattern1 = r'<\|ref\|>.*?<\|/ref\|>\n'
    # new_text1 = re.sub(pattern1, '', text, flags=re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):

    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, ouput_path):

    image_width, image_height = image.size
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    # try:
    # except IOError:
    #     try:
    #         font = ImageFont.truetype("DejaVuSans.ttf", 20) 
    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{ouput_path}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_path):

    result_image = draw_bounding_boxes(image, ref_texts, output_path)
    
    return result_image





def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=6, image_size=768, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # print(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # print(target_aspect_ratio)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio



def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform



def format_messages(
        conversations: List[Dict[str, str]],
        sft_format: str = "deepseek",
        system_prompt: str = "",
):
    """
    Applies the SFT template to conversation.

    Args:
        conversations (List[Dict]): A List of messages.
        sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
        system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

    Returns:
        sft_prompt (str): The formatted text.
    """

    conv = get_conv_template(sft_format)
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    sft_prompt = conv.get_prompt().strip()

    return sft_prompt


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = 0
    eos_id = 1
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]

    return t

def load_pil_images(conversations: List[Dict[str, str]]) -> List[Image.Image]:
    """

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            # print('----------------')
            # print(image_path)
            # print('----------------')
            # exit()
            
            # pil_img = Image.open(image_path)
            pil_img = load_image(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images


class BaseTransform(ABC):

    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError


class BasicImageTransform(BaseTransform):
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        self.mean = mean
        self.std = std
    
        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        x = self.transform(x)
        return x

class NoEOSTextStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):

        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        print(text, flush=True, end="")


class DeepseekOCR2Config(DeepseekV2Config):
    model_type = "DeepseekOCR2"


def scanpath_fixation_loss(
    fixation_logits: torch.Tensor,
    fixation_targets: torch.Tensor,
    fixation_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes soft-label cross entropy over visual patch fixation distributions.

    fixation_logits:  [B, T, N]
    fixation_targets: [B, T_max, N]
    fixation_mask:    [B, T_max]
    """
    if fixation_logits.size(-1) != fixation_targets.size(-1):
        raise ValueError(
            f"Fixation token mismatch: logits have {fixation_logits.size(-1)} tokens, "
            f"targets have {fixation_targets.size(-1)} tokens."
        )

    steps = fixation_logits.size(1)
    fixation_targets = fixation_targets[:, :steps, :].to(
        device=fixation_logits.device,
        dtype=torch.float32,
    )
    fixation_mask = fixation_mask[:, :steps].to(
        device=fixation_logits.device,
        dtype=torch.float32,
    )

    log_probs = F.log_softmax(fixation_logits.float(), dim=-1)
    per_step_loss = -(fixation_targets * log_probs).sum(dim=-1)

    return (per_step_loss * fixation_mask).sum() / fixation_mask.sum().clamp_min(1.0)




def scanpath_resize_distribution_to_target_tokens(
    distribution: torch.Tensor,
    target_tokens: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Resize a flattened spatial distribution to the target token count.

    distribution: [B, T, N]
    target_tokens: usually 256 for the SalChartQA 16x16 global fixation grid.

    If N already matches target_tokens, this returns the normalized input. If both
    N and target_tokens are square grids, it bilinearly resamples the distribution
    spatially and renormalizes. This keeps attention supervision compatible with
    144-token and 256-token visual grids without changing the original model path.
    """
    distribution = distribution.float()
    distribution = distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(eps)

    source_tokens = distribution.size(-1)
    if source_tokens == target_tokens:
        return distribution

    source_side = int(math.sqrt(source_tokens))
    target_side = int(math.sqrt(target_tokens))
    if source_side * source_side != source_tokens or target_side * target_side != target_tokens:
        raise ValueError(
            f"Cannot resize attention distribution from {source_tokens} to {target_tokens} tokens; "
            "both token counts must be square grids."
        )

    batch_size, steps, _ = distribution.shape
    distribution_2d = distribution.reshape(batch_size * steps, 1, source_side, source_side)
    resized = F.interpolate(
        distribution_2d,
        size=(target_side, target_side),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.reshape(batch_size, steps, target_tokens).clamp_min(0.0)
    return resized / resized.sum(dim=-1, keepdim=True).clamp_min(eps)


def scanpath_attention_alignment_loss(
    visual_attn_maps: torch.Tensor,
    fixation_targets: torch.Tensor,
    fixation_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Align the four supervised middle recurrent-Qwen layers with human fixations.

    visual_attn_maps: [B, L_sel, T, H, N]
        L_sel = 4 selected middle Qwen layers (10,11,12,13 for a 24-layer model).
        H = all query attention heads (14 in the current Qwen configuration).

    Supervision is intentionally applied in two stages:
      1. Within EACH selected layer, average over ALL attention heads:
             p_l,t = mean_h A_l,t,h
      2. Compute the fixation soft cross-entropy independently for each layer,
         then average the four layer losses.

    This means every selected middle layer must itself exhibit temporally aligned
    visual attention; averaging layers before the loss cannot let one layer hide
    another layer's mismatch.
    """
    if visual_attn_maps is None:
        raise ValueError("visual_attn_maps is None.")
    if visual_attn_maps.dim() != 5:
        raise ValueError(
            f"Expected visual_attn_maps to have shape [B, L_sel, T, H, N], "
            f"got {tuple(visual_attn_maps.shape)}."
        )
    if visual_attn_maps.size(1) != 4:
        raise ValueError(
            "Middle-four attention supervision expects exactly 4 retained layers, "
            f"got {visual_attn_maps.size(1)}."
        )

    # Mean across ALL heads independently within each selected layer.
    # [B,4,T,H,N] -> [B,4,T,N]
    pred_by_layer = visual_attn_maps.float().mean(dim=3)
    pred_by_layer = pred_by_layer / pred_by_layer.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)

    steps = pred_by_layer.size(2)
    targets = fixation_targets[:, :steps, :].to(
        device=pred_by_layer.device,
        dtype=torch.float32,
    )
    mask = fixation_mask[:, :steps].to(
        device=pred_by_layer.device,
        dtype=torch.float32,
    )
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(eps)

    layer_losses = []
    for layer_slot in range(pred_by_layer.size(1)):
        pred = pred_by_layer[:, layer_slot]  # [B,T,N]

        if pred.size(-1) != targets.size(-1):
            pred = scanpath_resize_distribution_to_target_tokens(
                pred,
                target_tokens=targets.size(-1),
                eps=eps,
            )

        per_step_loss = -(
            targets * pred.clamp_min(eps).log()
        ).sum(dim=-1)
        layer_loss = (
            (per_step_loss * mask).sum()
            / mask.sum().clamp_min(1.0)
        )
        layer_losses.append(layer_loss)

    # Equal supervision strength for each of the four middle layers.
    return torch.stack(layer_losses).mean()


def scanpath_attention_entropy_metrics(
    visual_attn_maps: torch.Tensor,
    fixation_targets: torch.Tensor,
    fixation_mask: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Diagnostics matching the middle-four/all-head attention objective.

    Prediction entropy and excess cross entropy are computed independently for
    each supervised layer after averaging all heads, then averaged across the
    four layers. Target entropy is shared by all layers.
    """
    if visual_attn_maps is None:
        raise ValueError("visual_attn_maps is None.")
    if visual_attn_maps.dim() != 5:
        raise ValueError(
            f"Expected visual_attn_maps to have shape [B, L_sel, T, H, N], "
            f"got {tuple(visual_attn_maps.shape)}."
        )
    if visual_attn_maps.size(1) != 4:
        raise ValueError(
            "Middle-four attention diagnostics expect exactly 4 retained layers, "
            f"got {visual_attn_maps.size(1)}."
        )

    pred_by_layer = visual_attn_maps.float().mean(dim=3)  # [B,4,T,N]
    pred_by_layer = pred_by_layer / pred_by_layer.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)

    steps = pred_by_layer.size(2)
    targets = fixation_targets[:, :steps, :].to(
        device=pred_by_layer.device,
        dtype=torch.float32,
    )
    mask = fixation_mask[:, :steps].to(
        device=pred_by_layer.device,
        dtype=torch.float32,
    )
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(eps)

    target_entropy_per_step = -(
        targets * targets.clamp_min(eps).log()
    ).sum(dim=-1)
    normalizer = mask.sum().clamp_min(1.0)
    target_entropy = (
        target_entropy_per_step * mask
    ).sum() / normalizer

    layer_prediction_entropies = []
    layer_excess_cross_entropies = []

    for layer_slot in range(pred_by_layer.size(1)):
        pred = pred_by_layer[:, layer_slot]

        if pred.size(-1) != targets.size(-1):
            pred = scanpath_resize_distribution_to_target_tokens(
                pred,
                target_tokens=targets.size(-1),
                eps=eps,
            )

        log_pred = pred.clamp_min(eps).log()
        pred_entropy_per_step = -(pred * log_pred).sum(dim=-1)
        cross_entropy_per_step = -(targets * log_pred).sum(dim=-1)
        excess_ce_per_step = (
            cross_entropy_per_step - target_entropy_per_step
        ).clamp_min(0.0)

        layer_prediction_entropies.append(
            (pred_entropy_per_step * mask).sum() / normalizer
        )
        layer_excess_cross_entropies.append(
            (excess_ce_per_step * mask).sum() / normalizer
        )

    prediction_entropy = torch.stack(layer_prediction_entropies).mean()
    excess_cross_entropy = torch.stack(layer_excess_cross_entropies).mean()
    return target_entropy, prediction_entropy, excess_cross_entropy

def scanpath_fixation_entropy_metrics(
    fixation_logits: torch.Tensor,
    fixation_targets: torch.Tensor,
    fixation_mask: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes masked mean entropy diagnostics for ScanFlow fixation prediction.

    Returns:
        target_entropy:
            H(y) = -sum_i y_i log y_i. This is the theoretical minimum
            soft-label cross entropy for the supplied target distributions.

        prediction_entropy:
            H(p) = -sum_i p_i log p_i. Lower values indicate sharper model
            predictions; higher values indicate broader or more uniform maps.

        excess_cross_entropy:
            CE(y, p) - H(y) = KL(y || p). This is the portion of the fixation
            cross entropy caused by prediction-target mismatch rather than the
            irreducible entropy of the Gaussian target itself.

    Shapes:
        fixation_logits:  [B, T, N]
        fixation_targets: [B, T_max, N]
        fixation_mask:    [B, T_max]
    """
    if fixation_logits.size(-1) != fixation_targets.size(-1):
        raise ValueError(
            f"Entropy token mismatch: logits have {fixation_logits.size(-1)} tokens, "
            f"targets have {fixation_targets.size(-1)} tokens."
        )

    steps = fixation_logits.size(1)
    targets = fixation_targets[:, :steps, :].to(
        device=fixation_logits.device,
        dtype=torch.float32,
    )
    mask = fixation_mask[:, :steps].to(
        device=fixation_logits.device,
        dtype=torch.float32,
    )

    # Normalize defensively in case preprocessing introduces small numerical
    # deviations from an exact row sum of one.
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(eps)

    log_probs = F.log_softmax(fixation_logits.float(), dim=-1)
    probs = log_probs.exp()

    per_step_target_entropy = -(
        targets * targets.clamp_min(eps).log()
    ).sum(dim=-1)

    per_step_prediction_entropy = -(
        probs * log_probs
    ).sum(dim=-1)

    per_step_cross_entropy = -(
        targets * log_probs
    ).sum(dim=-1)

    per_step_excess_cross_entropy = (
        per_step_cross_entropy - per_step_target_entropy
    ).clamp_min(0.0)

    normalizer = mask.sum().clamp_min(1.0)

    target_entropy = (
        per_step_target_entropy * mask
    ).sum() / normalizer

    prediction_entropy = (
        per_step_prediction_entropy * mask
    ).sum() / normalizer

    excess_cross_entropy = (
        per_step_excess_cross_entropy * mask
    ).sum() / normalizer

    return target_entropy, prediction_entropy, excess_cross_entropy


def scanpath_diversity_loss(
    fixation_logits: torch.Tensor,
    fixation_targets: torch.Tensor,
    fixation_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Target-conditioned pairwise diversity loss.

    The loss penalizes predicted fixation distributions for being similar when
    the corresponding ground-truth fixation distributions are different.

    fixation_logits:  [B, T, N]
    fixation_targets: [B, T_max, N]
    fixation_mask:    [B, T_max]

    This is intentionally applied to predicted fixation distributions rather
    than raw hidden states so that diversity must affect the fixation output.
    """
    if fixation_logits.size(-1) != fixation_targets.size(-1):
        raise ValueError(
            f"Diversity token mismatch: logits have {fixation_logits.size(-1)} tokens, "
            f"targets have {fixation_targets.size(-1)} tokens."
        )

    steps = fixation_logits.size(1)
    if steps <= 1:
        return fixation_logits.sum() * 0.0

    predictions = F.softmax(fixation_logits.float(), dim=-1)
    targets = fixation_targets[:, :steps, :].to(
        device=fixation_logits.device,
        dtype=torch.float32,
    )
    mask = fixation_mask[:, :steps].to(
        device=fixation_logits.device,
        dtype=torch.float32,
    )

    pred_norm = F.normalize(predictions, p=2, dim=-1, eps=eps)
    target_norm = F.normalize(targets, p=2, dim=-1, eps=eps)

    pred_similarity = torch.matmul(pred_norm, pred_norm.transpose(1, 2))
    target_similarity = torch.matmul(target_norm, target_norm.transpose(1, 2))

    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
    diagonal = torch.eye(
        steps,
        device=fixation_logits.device,
        dtype=pair_mask.dtype,
    ).unsqueeze(0)
    pair_mask = pair_mask * (1.0 - diagonal)

    # Pairs with similar target maps receive little or no diversity penalty.
    target_difference = (1.0 - target_similarity).clamp_min(0.0).detach()
    weights = pair_mask * target_difference

    weighted_similarity = weights * pred_similarity
    normalizer = weights.sum().clamp_min(eps)
    return weighted_similarity.sum() / normalizer


def scanflow_loss_gradients(
    loss: Optional[torch.Tensor],
    parameters: list[torch.nn.Parameter],
) -> list[Optional[torch.Tensor]]:
    """Return gradients for one loss without modifying parameter.grad."""
    if loss is None or not torch.is_tensor(loss) or not loss.requires_grad:
        return [None] * len(parameters)
    return list(torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    ))


def scanflow_gradient_norm(
    gradients: list[Optional[torch.Tensor]],
) -> torch.Tensor:
    total = None
    for grad in gradients:
        if grad is None:
            continue
        value = grad.detach().float().pow(2).sum()
        total = value if total is None else total + value
    return torch.tensor(0.0) if total is None else total.sqrt()


def scanflow_gradient_dot(
    gradients_a: list[Optional[torch.Tensor]],
    gradients_b: list[Optional[torch.Tensor]],
) -> torch.Tensor:
    total = None
    for grad_a, grad_b in zip(gradients_a, gradients_b):
        if grad_a is None or grad_b is None:
            continue
        value = (grad_a.detach().float() * grad_b.detach().float()).sum()
        total = value if total is None else total + value
    return torch.tensor(0.0) if total is None else total


def scanflow_gradient_cosine(
    gradients_a: list[Optional[torch.Tensor]],
    gradients_b: list[Optional[torch.Tensor]],
    eps: float = 1e-12,
) -> torch.Tensor:
    norm_a = scanflow_gradient_norm(gradients_a)
    norm_b = scanflow_gradient_norm(gradients_b)
    denominator = norm_a * norm_b
    if float(denominator.detach().cpu()) <= eps:
        return torch.tensor(float("nan"))
    return scanflow_gradient_dot(gradients_a, gradients_b) / denominator


def scanflow_weighted_combined_gradient_norm(
    gradient_groups: list[tuple[float, list[Optional[torch.Tensor]]]],
) -> torch.Tensor:
    if not gradient_groups:
        return torch.tensor(0.0)
    total_sq = None
    for parameter_index in range(len(gradient_groups[0][1])):
        combined = None
        for weight, gradients in gradient_groups:
            grad = gradients[parameter_index]
            if grad is None or weight == 0.0:
                continue
            term = grad.detach().float() * float(weight)
            combined = term if combined is None else combined + term
        if combined is None:
            continue
        value = combined.pow(2).sum()
        total_sq = value if total_sq is None else total_sq + value
    return torch.tensor(0.0) if total_sq is None else total_sq.sqrt()


def scanflow_is_main_process() -> bool:
    return not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    )


def scanflow_diagnostic_patterns() -> tuple[str, ...]:
    raw = os.getenv(
        "DEEPSEEK_OCR2_UPDATE_PARAM_FILTER",
        "qwen2_model.model.model,qwen2_model.model.text_cross_norms,"
        "qwen2_model.model.text_cross_attn_layers,qwen2_model.text_context_proj",
    )
    return tuple(part.strip() for part in raw.split(",") if part.strip())


class DeepseekOCR2Model(DeepseekV2Model):
    """DeepSeek-OCR2 outer model for dynamic ScanFlow Pro V2.

    The visual encoder preserves the recurrent Plan-1 generation of h_1...h_T,
    all against the immutable visual tokens V. After H is complete, V queries H
    once to produce R. A tokenwise reasoning-intensity gate lambda_i(V,Q) then
    produces [V + lambda(V,Q)*R; H]. Human scanpath
    supervision remains attached to the same Qwen self-attention h_t -> V maps
    on the GLOBAL image branch.
    """

    config_class = DeepseekOCR2Config

    def __init__(self, config: DeepseekV2Config):
        super(DeepseekOCR2Model, self).__init__(config)
        self.sam_model = build_sam_vit_b()
        self.qwen2_model = build_qwen2_decoder_as_encoder(max_scanpath_steps=32)
        n_embed = 1280
        self.projector = MlpProjector(
            Dict(projector_type="linear", input_dim=896, n_embed=n_embed)
        )
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

        # Compatibility attributes used by older training/evaluation scripts.
        # Recurrence and the V2 global residual are intrinsic to qwen2_model.
        self.enable_scanflow = True
        self.enable_text_query_residual = True
        self.last_visual_attn_maps = None
        self.last_visual_attn_layer_indices = tuple(
            getattr(self.qwen2_model, "attention_supervision_layer_indices", ())
        )

        # Reset is intentionally deferred until the first real forward pass.
        # Hugging Face/MS-Swift constructs the module first and loads checkpoint
        # tensors afterward; resetting inside __init__ would therefore be
        # overwritten by from_pretrained().
        self._scanflow_new_params_reset_done = False
        self._scanflow_v2_new_params_reset_done = False

    def _select_text_context(
        self,
        inputs_embeds: torch.Tensor,
        sample_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images_seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The question context conditions the visual/recurrent encoder, but we do
        # not need gradients back into the outer LLM token embeddings through this
        # auxiliary branch.
        token_source = inputs_embeds[sample_idx].detach()
        mask = torch.ones(token_source.shape[0], dtype=torch.bool, device=token_source.device)
        if attention_mask is not None:
            mask = mask & attention_mask[sample_idx].bool()
        if images_seq_mask is not None:
            mask = mask & (~images_seq_mask[sample_idx].bool())
        if labels is not None:
            prompt_mask = mask & labels[sample_idx].eq(-100)
            if prompt_mask.sum() > 0:
                mask = prompt_mask
        if mask.sum() == 0:
            mask = torch.ones_like(mask)
        return token_source[mask].unsqueeze(0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        scanpath_lengths: Optional[torch.Tensor] = None,
        scanpath_steps: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # Original recurrent Plan-1 reset mechanism. Keep this switch off when
        # starting V2 from an already-trained Plan-1 recurrent checkpoint.
        if (
            not self._scanflow_new_params_reset_done
            and os.getenv("SCANFLOW_RESET_NEW_PARAMS", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        ):
            qwen2_model_for_reset = getattr(self, "qwen2_model", None)
            if qwen2_model_for_reset is None or not hasattr(
                qwen2_model_for_reset, "reset_scanflow_parameters"
            ):
                raise RuntimeError(
                    "SCANFLOW_RESET_NEW_PARAMS=1, but qwen2_model does not expose "
                    "reset_scanflow_parameters()."
                )
            qwen2_model_for_reset.reset_scanflow_parameters()
            self._scanflow_new_params_reset_done = True

        # Independent Version-2 reset mechanism. For the first transition from
        # a Plan-1 checkpoint, this initializes only R=CrossAttn(V,H,H) and its
        # question-conditioned tokenwise intensity head, leaving every trained
        # Plan-1 parameter untouched.
        if (
            not self._scanflow_v2_new_params_reset_done
            and os.getenv("SCANFLOW_V2_RESET_NEW_PARAMS", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        ):
            qwen2_model_for_reset = getattr(self, "qwen2_model", None)
            if qwen2_model_for_reset is None or not hasattr(
                qwen2_model_for_reset, "reset_scanflow_v2_parameters"
            ):
                raise RuntimeError(
                    "SCANFLOW_V2_RESET_NEW_PARAMS=1, but qwen2_model does not "
                    "expose reset_scanflow_v2_parameters()."
                )
            qwen2_model_for_reset.reset_scanflow_v2_parameters()
            self._scanflow_v2_new_params_reset_done = True

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        sam_model = getattr(self, "sam_model", None)
        qwen2_model = getattr(self, "qwen2_model", None)
        self.last_visual_attn_maps = None
        self.last_visual_attn_layer_indices = tuple(
            getattr(qwen2_model, "attention_supervision_layer_indices", ())
        )
        batch_visual_attn_maps = []

        if (
            sam_model is not None
            and qwen2_model is not None
            and (input_ids.shape[1] != 1 or self.training)
            and torch.sum(images[0][1]).item() != 0
        ):
            idx = 0
            for image, crop_shape in zip(images, images_spatial_crop):
                del crop_shape
                patches = image[0]
                image_ori = image[1]

                sample_scanpath_lengths = None
                if scanpath_lengths is not None:
                    sample_scanpath_lengths = scanpath_lengths[idx:idx + 1]

                # ScanFlow Pro V2 uses a genuinely dynamic number of causal states.
                # During training use the sample's true length; at inference, None
                # lets qwen2_model fall back to max_scanpath_steps=32.
                if scanpath_steps is not None:
                    sample_scanpath_steps = int(scanpath_steps)
                elif sample_scanpath_lengths is not None:
                    sample_scanpath_steps = int(sample_scanpath_lengths.max().item())
                else:
                    sample_scanpath_steps = None

                sample_text_context = self._select_text_context(
                    inputs_embeds,
                    idx,
                    attention_mask=attention_mask,
                    labels=labels,
                    images_seq_mask=images_seq_mask,
                )

                if torch.sum(patches).item() != 0:
                    with torch.no_grad():
                        local_features_1 = sam_model(patches)
                        global_features_1 = sam_model(image_ori)

                    local_text_context = sample_text_context.expand(
                        local_features_1.shape[0], -1, -1
                    )
                    global_text_context = sample_text_context.expand(
                        global_features_1.shape[0], -1, -1
                    )
                    local_scanpath_lengths = (
                        sample_scanpath_lengths.expand(local_features_1.shape[0])
                        if sample_scanpath_lengths is not None else None
                    )
                    global_scanpath_lengths = (
                        sample_scanpath_lengths.expand(global_features_1.shape[0])
                        if sample_scanpath_lengths is not None else None
                    )

                    # Every view returns [V + lambda(V,Q)*R; H]. The residual and
                    # tokenwise intensity are computed once after H; only the
                    # global branch retains Qwen attention maps for scanpath loss.
                    local_features_2 = qwen2_model(
                        local_features_1,
                        text_context=local_text_context,
                        scanpath_lengths=local_scanpath_lengths,
                        scanpath_steps=sample_scanpath_steps,
                        collect_visual_attention=False,
                    )
                    local_features = self.projector(local_features_2)

                    global_features_2 = qwen2_model(
                        global_features_1,
                        text_context=global_text_context,
                        scanpath_lengths=global_scanpath_lengths,
                        scanpath_steps=sample_scanpath_steps,
                        collect_visual_attention=getattr(
                            qwen2_model, "enable_attention_supervision", False
                        ),
                    )
                    global_visual_attn_maps = getattr(
                        qwen2_model, "last_visual_attn_maps", None
                    )
                    if global_visual_attn_maps is not None:
                        batch_visual_attn_maps.append(global_visual_attn_maps)
                    global_features = self.projector(global_features_2)

                    local_features = local_features.reshape(-1, local_features.size(-1))
                    global_features = global_features.reshape(-1, global_features.size(-1))
                    global_local_features = torch.cat(
                        [local_features, global_features, self.view_seperator[None, :]],
                        dim=0,
                    )
                else:
                    with torch.no_grad():
                        global_features_1 = sam_model(image_ori)
                    global_text_context = sample_text_context.expand(
                        global_features_1.shape[0], -1, -1
                    )
                    global_scanpath_lengths = (
                        sample_scanpath_lengths.expand(global_features_1.shape[0])
                        if sample_scanpath_lengths is not None else None
                    )
                    global_features_2 = qwen2_model(
                        global_features_1,
                        text_context=global_text_context,
                        scanpath_lengths=global_scanpath_lengths,
                        scanpath_steps=sample_scanpath_steps,
                        collect_visual_attention=getattr(
                            qwen2_model, "enable_attention_supervision", False
                        ),
                    )
                    global_visual_attn_maps = getattr(
                        qwen2_model, "last_visual_attn_maps", None
                    )
                    if global_visual_attn_maps is not None:
                        batch_visual_attn_maps.append(global_visual_attn_maps)
                    global_features = self.projector(global_features_2)
                    global_features = global_features.reshape(-1, global_features.size(-1))
                    global_local_features = torch.cat(
                        [global_features, self.view_seperator[None, :]], dim=0
                    )

                image_slot_mask = images_seq_mask[idx].bool().to(inputs_embeds.device)
                num_slots = int(image_slot_mask.sum().item())
                num_features = int(global_local_features.size(0))
                if num_slots != num_features:
                    raise ValueError(
                        "ScanFlow Pro V2 image-token count mismatch for sample "
                        f"{idx}: template reserved {num_slots} <image> positions, "
                        f"but the visual encoder produced {num_features} features. "
                        "The template must reserve (N_visual + T) slots per encoded "
                        "view plus the existing view-separator slot."
                    )

                inputs_embeds[idx].masked_scatter_(
                    image_slot_mask.unsqueeze(-1),
                    global_local_features.to(
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    ),
                )
                idx += 1

            self.last_visual_attn_maps = (
                torch.cat(batch_visual_attn_maps, dim=0)
                if batch_visual_attn_maps else None
            )

        return super(DeepseekOCR2Model, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class DeepseekOCR2ForCausalLM(DeepseekV2ForCausalLM):

    config_class = DeepseekOCR2Config
    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(DeepseekV2ForCausalLM, self).__init__(config)
        self.model = DeepseekOCR2Model(config)

        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Cross-forward snapshots reveal whether an optimizer step changed the
        # selected ScanFlow parameters. During gradient accumulation many
        # forwards will correctly report no change; a change should appear on
        # the first forward after optimizer.step().
        self._scanflow_parameter_snapshot = {}
        self._scanflow_snapshot_forward_step = 0
        self._scanflow_optimizer_membership_logged = False

    def get_model(self):
        return self.model


    def _scanflow_selected_named_parameters(self):
        patterns = scanflow_diagnostic_patterns()
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and any(pattern in name for pattern in patterns)
        ]

    def _scanflow_log_parameter_changes(self) -> None:
        if not self.training or not scanflow_is_main_process():
            return
        if os.getenv("DEEPSEEK_OCR2_LOG_PARAMETER_UPDATES", "1") != "1":
            return

        self._scanflow_snapshot_forward_step += 1
        interval = max(
            1, int(os.getenv("DEEPSEEK_OCR2_UPDATE_LOG_INTERVAL", "10"))
        )
        selected = self._scanflow_selected_named_parameters()

        # Compare current values with values saved at the end of the preceding
        # forward. A nonzero delta means an optimizer update occurred between
        # the two forwards. Zero deltas are expected during accumulation.
        if self._scanflow_parameter_snapshot and (
            self._scanflow_snapshot_forward_step % interval == 0
        ):
            changed_tensors = 0
            total_update_sq = 0.0
            total_parameter_sq = 0.0

            for name, parameter in selected:
                previous = self._scanflow_parameter_snapshot.get(name)
                if previous is None:
                    continue
                current = parameter.detach().float()
                delta = current - previous.to(device=current.device)
                update_norm = delta.norm().item()
                parameter_norm = current.norm().item()
                relative_update = update_norm / (parameter_norm + 1e-12)
                changed = update_norm > 0.0
                changed_tensors += int(changed)
                total_update_sq += update_norm * update_norm
                total_parameter_sq += parameter_norm * parameter_norm

                print(
                    "[scanflow parameter update] "
                    f"forward_step={self._scanflow_snapshot_forward_step} "
                    f"name={name} "
                    f"parameter_norm={parameter_norm:.8g} "
                    f"update_norm={update_norm:.8g} "
                    f"max_update={delta.abs().max().item():.8g} "
                    f"relative_update={relative_update:.8g} "
                    f"changed={changed} "
                    f"finite={bool(torch.isfinite(delta).all().item())}",
                    flush=True,
                )

            total_update_norm = total_update_sq ** 0.5
            total_parameter_norm = total_parameter_sq ** 0.5
            print(
                "[scanflow parameter update summary] "
                f"forward_step={self._scanflow_snapshot_forward_step} "
                f"selected_tensors={len(selected)} "
                f"changed_tensors={changed_tensors} "
                f"total_update_norm={total_update_norm:.8g} "
                f"total_parameter_norm={total_parameter_norm:.8g} "
                f"relative_update={total_update_norm / (total_parameter_norm + 1e-12):.8g}",
                flush=True,
            )

    def _scanflow_save_parameter_snapshot(self) -> None:
        if not self.training:
            return
        if os.getenv("DEEPSEEK_OCR2_LOG_PARAMETER_UPDATES", "1") != "1":
            return
        self._scanflow_parameter_snapshot = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in self._scanflow_selected_named_parameters()
        }


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        scanpath_lengths: Optional[torch.Tensor] = None,
        scanpath_steps: Optional[int] = None,
        fixation_targets: Optional[torch.Tensor] = None,
        fixation_mask: Optional[torch.Tensor] = None,
        fixation_loss_weight: Optional[float] = None,
        diversity_loss_weight: Optional[float] = None,
        attention_loss_weight: Optional[float] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self._scanflow_log_parameter_changes()

        requested_attention_weight = attention_loss_weight
        if requested_attention_weight is None:
            requested_attention_weight = float(
                os.getenv("DEEPSEEK_OCR2_ATTENTION_LOSS_WEIGHT", "0.0")
            )
        requested_attention_weight = float(requested_attention_weight)

        # Enable recurrent Qwen self-attention collection whenever the attention
        # objective or explicit attention-supervision switch is active.
        collect_attention_maps = (
            requested_attention_weight != 0.0
            or os.getenv("DEEPSEEK_OCR2_ENABLE_ATTENTION_SUPERVISION", "0") == "1"
        )
        qwen_for_attention = getattr(self.model, "qwen2_model", None)
        if collect_attention_maps and qwen_for_attention is not None and hasattr(
            qwen_for_attention, "enable_attention_supervision"
        ):
            qwen_for_attention.enable_attention_supervision = True

        outputs  = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            images_seq_mask = images_seq_mask,
            images_spatial_crop = images_spatial_crop,
            labels=labels,
            return_dict=return_dict,
            scanpath_lengths=scanpath_lengths,
            scanpath_steps=scanpath_steps,
            
        )


        
        # print(transformer_outputs)

        hidden_states = outputs[0]

        if not getattr(self, "_scanflow_printed_tensor_stats", False):
            scanflow_debug_tensor("hidden_states_before_lm_head", hidden_states)
            scanflow_debug_tensor("labels", labels)
            if labels is not None:
                valid_labels = labels[..., 1:] != -100
                print(
                    "[scanflow debug] valid shifted labels:",
                    int(valid_labels.sum().item()),
                    "/",
                    valid_labels.numel(),
                    flush=True,
                )

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if not getattr(self, "_scanflow_printed_tensor_stats", False):
            scanflow_debug_tensor("logits_after_lm_head", logits)

        # logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        lm_loss = loss
        # Recurrent Plan 1 deliberately supervises ONLY the Qwen self-attention
        # from each generated h_t to visual tokens. The old decoded-fixation and
        # fixation-diversity objectives are disabled by architecture, not merely
        # assigned small weights.
        fixation_loss = None
        diversity_loss = None
        attention_loss = None
        fixation_weight = 0.0
        diversity_weight = 0.0
        attention_weight = requested_attention_weight
        target_entropy = None
        prediction_entropy = None
        excess_fixation_ce = None
        attention_target_entropy = None
        attention_prediction_entropy = None
        attention_excess_ce = None

        # Keep these legacy keyword arguments accepted so existing ms-swift data
        # plumbing does not break, but they have no effect in recurrent Plan 1.
        del fixation_loss_weight, diversity_loss_weight

        if fixation_targets is not None and fixation_mask is not None:
            visual_attn_maps = getattr(self.model, "last_visual_attn_maps", None)
            if visual_attn_maps is None:
                qwen = getattr(self.model, "qwen2_model", None)
                visual_attn_maps = getattr(qwen, "last_visual_attn_maps", None)

            needs_attention_maps = (
                attention_weight != 0.0
                or os.getenv("DEEPSEEK_OCR2_COMPUTE_ATTENTION_DIAGNOSTICS", "0") == "1"
            )
            if visual_attn_maps is None and attention_weight != 0.0:
                raise RuntimeError(
                    "DEEPSEEK_OCR2_ATTENTION_LOSS_WEIGHT is nonzero, but recurrent "
                    "Plan 1 produced no Qwen self-attention maps. Ensure attention "
                    "supervision is enabled before the recurrent encoder forward."
                )

            if not getattr(self, "_scanflow_printed_tensor_stats", False):
                scanflow_debug_tensor("recurrent_plan1_visual_attn_maps", visual_attn_maps)
                scanflow_debug_tensor("fixation_targets", fixation_targets)
                scanflow_debug_tensor("fixation_mask", fixation_mask)
                self._scanflow_printed_tensor_stats = True

            if visual_attn_maps is not None and needs_attention_maps:
                # deepencoderv2 retains only the four selected middle layers.
                # Within each layer the loss averages ALL 14 query heads, aligns
                # that layer-level map to the temporal fixation target, and then
                # averages the four independent layer losses.
                attention_loss = scanpath_attention_alignment_loss(
                    visual_attn_maps=visual_attn_maps,
                    fixation_targets=fixation_targets,
                    fixation_mask=fixation_mask,
                )
                (
                    attention_target_entropy,
                    attention_prediction_entropy,
                    attention_excess_ce,
                ) = scanpath_attention_entropy_metrics(
                    visual_attn_maps=visual_attn_maps,
                    fixation_targets=fixation_targets,
                    fixation_mask=fixation_mask,
                )

            if attention_loss is not None:
                attention_term = attention_weight * attention_loss
                loss = attention_term if lm_loss is None else lm_loss + attention_term

        # Count forward micro-steps before optional diagnostics.
        self._scanflow_forward_step = getattr(self, "_scanflow_forward_step", 0) + 1

        # Per-loss gradient diagnostics. torch.autograd.grad leaves .grad buffers
        # untouched, so the Trainer can still backpropagate the combined loss.
        # This is expensive, so logging is interval-controlled.
        self.last_gradient_metrics = None
        grad_logging_enabled = os.getenv("DEEPSEEK_OCR2_LOG_GRADIENTS", "1") == "1"
        grad_log_interval = max(
            1, int(os.getenv("DEEPSEEK_OCR2_GRAD_LOG_INTERVAL", "10"))
        )
        should_log_gradients = (
            self.training
            and torch.is_grad_enabled()
            and grad_logging_enabled
            and self._scanflow_forward_step % grad_log_interval == 0
        )

        if should_log_gradients:
            filter_text = os.getenv("DEEPSEEK_OCR2_GRAD_PARAM_FILTER", "").strip()
            filters = [item.strip() for item in filter_text.split(",") if item.strip()]
            selected_named_parameters = [
                (name, parameter)
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
                and (not filters or any(token in name for token in filters))
            ]
            selected_parameters = [parameter for _, parameter in selected_named_parameters]

            if selected_parameters:
                lm_gradients = scanflow_loss_gradients(lm_loss, selected_parameters)
                fixation_gradients = scanflow_loss_gradients(
                    fixation_loss, selected_parameters
                )
                diversity_gradients = scanflow_loss_gradients(
                    diversity_loss, selected_parameters
                )
                attention_gradients = scanflow_loss_gradients(
                    attention_loss, selected_parameters
                )

                lm_grad_norm = scanflow_gradient_norm(lm_gradients)
                fixation_grad_norm = scanflow_gradient_norm(fixation_gradients)
                diversity_grad_norm = scanflow_gradient_norm(diversity_gradients)
                attention_grad_norm = scanflow_gradient_norm(attention_gradients)
                weighted_fixation_grad_norm = abs(float(fixation_weight)) * fixation_grad_norm
                weighted_diversity_grad_norm = abs(float(diversity_weight)) * diversity_grad_norm
                weighted_attention_grad_norm = abs(float(attention_weight)) * attention_grad_norm
                combined_grad_norm = scanflow_weighted_combined_gradient_norm([
                    (1.0, lm_gradients),
                    (float(fixation_weight), fixation_gradients),
                    (float(diversity_weight), diversity_gradients),
                    (float(attention_weight), attention_gradients),
                ])

                lm_fixation_cosine = scanflow_gradient_cosine(
                    lm_gradients, fixation_gradients
                )
                lm_diversity_cosine = scanflow_gradient_cosine(
                    lm_gradients, diversity_gradients
                )
                fixation_diversity_cosine = scanflow_gradient_cosine(
                    fixation_gradients, diversity_gradients
                )
                lm_attention_cosine = scanflow_gradient_cosine(
                    lm_gradients, attention_gradients
                )
                fixation_attention_cosine = scanflow_gradient_cosine(
                    fixation_gradients, attention_gradients
                )
                diversity_attention_cosine = scanflow_gradient_cosine(
                    diversity_gradients, attention_gradients
                )

                def _grad_value(value):
                    return float(value.detach().float().cpu().item())

                self.last_gradient_metrics = {
                    "num_trainable_tensors": len(selected_named_parameters),
                    "num_trainable_elements": sum(
                        parameter.numel() for _, parameter in selected_named_parameters
                    ),
                    "lm_grad_norm": _grad_value(lm_grad_norm),
                    "fixation_grad_norm": _grad_value(fixation_grad_norm),
                    "diversity_grad_norm": _grad_value(diversity_grad_norm),
                    "attention_grad_norm": _grad_value(attention_grad_norm),
                    "weighted_fixation_grad_norm": _grad_value(weighted_fixation_grad_norm),
                    "weighted_diversity_grad_norm": _grad_value(weighted_diversity_grad_norm),
                    "weighted_attention_grad_norm": _grad_value(weighted_attention_grad_norm),
                    "combined_grad_norm": _grad_value(combined_grad_norm),
                    "lm_fixation_grad_cosine": _grad_value(lm_fixation_cosine),
                    "lm_diversity_grad_cosine": _grad_value(lm_diversity_cosine),
                    "fixation_diversity_grad_cosine": _grad_value(fixation_diversity_cosine),
                    "lm_attention_grad_cosine": _grad_value(lm_attention_cosine),
                    "fixation_attention_grad_cosine": _grad_value(fixation_attention_cosine),
                    "diversity_attention_grad_cosine": _grad_value(diversity_attention_cosine),
                }

                print(
                    "[scanflow gradients] "
                    f"forward_step={self._scanflow_forward_step} "
                    f"trainable_tensors={self.last_gradient_metrics['num_trainable_tensors']} "
                    f"trainable_elements={self.last_gradient_metrics['num_trainable_elements']} "
                    f"lm_norm={self.last_gradient_metrics['lm_grad_norm']:.6g} "
                    f"fixation_norm={self.last_gradient_metrics['fixation_grad_norm']:.6g} "
                    f"diversity_norm={self.last_gradient_metrics['diversity_grad_norm']:.6g} "
                    f"attention_norm={self.last_gradient_metrics['attention_grad_norm']:.6g} "
                    f"weighted_fixation_norm={self.last_gradient_metrics['weighted_fixation_grad_norm']:.6g} "
                    f"weighted_diversity_norm={self.last_gradient_metrics['weighted_diversity_grad_norm']:.6g} "
                    f"weighted_attention_norm={self.last_gradient_metrics['weighted_attention_grad_norm']:.6g} "
                    f"combined_norm={self.last_gradient_metrics['combined_grad_norm']:.6g} "
                    f"cos_lm_fixation={self.last_gradient_metrics['lm_fixation_grad_cosine']:.6g} "
                    f"cos_lm_diversity={self.last_gradient_metrics['lm_diversity_grad_cosine']:.6g} "
                    f"cos_fixation_diversity={self.last_gradient_metrics['fixation_diversity_grad_cosine']:.6g} "
                    f"cos_lm_attention={self.last_gradient_metrics['lm_attention_grad_cosine']:.6g} "
                    f"cos_fixation_attention={self.last_gradient_metrics['fixation_attention_grad_cosine']:.6g} "
                    f"cos_diversity_attention={self.last_gradient_metrics['diversity_attention_grad_cosine']:.6g}",
                    flush=True,
                )


                if os.getenv("DEEPSEEK_OCR2_LOG_NAMED_GRADIENTS", "1") == "1":
                    named_limit = max(
                        1,
                        int(os.getenv("DEEPSEEK_OCR2_NAMED_GRADIENT_LIMIT", "80")),
                    )
                    for parameter_index, (
                        parameter_name,
                        parameter,
                    ) in enumerate(selected_named_parameters[:named_limit]):
                        lm_grad = lm_gradients[parameter_index]
                        fix_grad = fixation_gradients[parameter_index]
                        div_grad = diversity_gradients[parameter_index]
                        attn_grad = attention_gradients[parameter_index]

                        def _named_norm(grad):
                            if grad is None:
                                return None
                            return grad.detach().float().norm().item()

                        def _named_max(grad):
                            if grad is None:
                                return None
                            return grad.detach().float().abs().max().item()

                        fix_norm = _named_norm(fix_grad)
                        attn_norm = _named_norm(attn_grad)
                        weighted_fix_norm = (
                            None
                            if fix_norm is None
                            else abs(float(fixation_weight)) * fix_norm
                        )
                        weighted_attn_norm = (
                            None
                            if attn_norm is None
                            else abs(float(attention_weight)) * attn_norm
                        )

                        print(
                            "[scanflow named component gradient] "
                            f"forward_step={self._scanflow_forward_step} "
                            f"name={parameter_name} "
                            f"shape={tuple(parameter.shape)} "
                            f"parameter_norm={parameter.detach().float().norm().item():.8g} "
                            f"lm_norm={_named_norm(lm_grad)} "
                            f"fixation_norm={fix_norm} "
                            f"weighted_fixation_norm={weighted_fix_norm} "
                            f"diversity_norm={_named_norm(div_grad)} "
                            f"attention_norm={attn_norm} "
                            f"weighted_attention_norm={weighted_attn_norm} "
                            f"fixation_max={_named_max(fix_grad)} "
                            f"attention_max={_named_max(attn_grad)} "
                            f"fixation_grad_none={fix_grad is None} "
                            f"attention_grad_none={attn_grad is None}",
                            flush=True,
                        )

        self.last_lm_loss = lm_loss.detach() if lm_loss is not None else None
        self.last_fixation_loss = fixation_loss.detach() if fixation_loss is not None else None
        self.last_diversity_loss = diversity_loss.detach() if diversity_loss is not None else None
        self.last_attention_loss = attention_loss.detach() if attention_loss is not None else None
        self.last_target_entropy = target_entropy.detach() if target_entropy is not None else None
        self.last_prediction_entropy = prediction_entropy.detach() if prediction_entropy is not None else None
        self.last_excess_fixation_ce = excess_fixation_ce.detach() if excess_fixation_ce is not None else None
        self.last_attention_target_entropy = attention_target_entropy.detach() if attention_target_entropy is not None else None
        self.last_attention_prediction_entropy = attention_prediction_entropy.detach() if attention_prediction_entropy is not None else None
        self.last_attention_excess_ce = attention_excess_ce.detach() if attention_excess_ce is not None else None

        if os.getenv("DEEPSEEK_OCR2_LOG_LOSSES", "1") == "1":
            def _loss_value(value):
                if value is None:
                    return None
                return float(value.detach().float().item())

            print(
                "[scanflow losses] "
                f"forward_step={self._scanflow_forward_step} "
                f"total={_loss_value(loss)} "
                f"lm={_loss_value(lm_loss)} "
                f"fixation={_loss_value(fixation_loss)} "
                f"target_entropy={_loss_value(target_entropy)} "
                f"prediction_entropy={_loss_value(prediction_entropy)} "
                f"excess_fixation_ce={_loss_value(excess_fixation_ce)} "
                f"diversity={_loss_value(diversity_loss)} "
                f"attention={_loss_value(attention_loss)} "
                f"attention_target_entropy={_loss_value(attention_target_entropy)} "
                f"attention_prediction_entropy={_loss_value(attention_prediction_entropy)} "
                f"attention_excess_ce={_loss_value(attention_excess_ce)} "
                f"fixation_weight={float(fixation_weight):.6g} "
                f"diversity_weight={float(diversity_weight):.6g} "
                f"attention_weight={float(attention_weight):.6g}",
                flush=True,
            )

        self._scanflow_save_parameter_snapshot()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if self.generation_config.cache_implementation == "static":
        #     # generation with static cache
        #     cache_position = kwargs.get("cache_position", None)
        #     if cache_position is None:
        #         past_length = 0
        #     else:
        #         past_length = cache_position[-1] + 1
        #     input_ids = input_ids[:, past_length:]
        #     position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "images_seq_mask": kwargs.get("images_seq_mask", None),
                "images_spatial_crop": kwargs.get("images_spatial_crop", None),
                "scanpath_lengths": kwargs.get("scanpath_lengths", None),
                "scanpath_steps": kwargs.get("scanpath_steps", None),
            }
        )
        return model_inputs
    

    def disable_torch_init(self):
        """
        Disable the redundant torch default initialization to accelerate model creation.
        """
        import torch
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)



    def infer(self, tokenizer, prompt='', image_file='', output_path = '', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False, scanpath_steps=32):
        self.disable_torch_init()

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/images', exist_ok=True)

        if prompt and image_file:
            conversation = [
                {
                    "role": "<|User|>",
                    # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                    "content": f'{prompt}',
                    # "content": "君不见黄河之水天上来的下一句是什么？",
                    # "content": "<image>\nFree OCR. ",
                    # "content": "<image>\nParse the figure. ",
                    # "content": "<image>\nExtract the text in the image. ",
                    "images": [f'{image_file}'],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        
        elif prompt:
            conversation = [
                {
                    "role": "<|User|>",
                    # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                    "content": f'{prompt}',
                    # "content": "君不见黄河之水天上来的下一句是什么？",
                    # "content": "<image>\nFree OCR. ",
                    # "content": "<image>\nParse the figure. ",
                    # "content": "<image>\nExtract the text in the image. ",
                    # "images": [f'{image_file}'],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        else:
            assert False, f'prompt is none!'
        
        prompt = format_messages(conversations=conversation, sft_format='plain', system_prompt='')

        patch_size = 16
        downsample_ratio = 4
        images = load_pil_images(conversation)

        valid_img_tokens = 0
        ratio = 1

        image_draw = images[0].copy()

        w,h = image_draw.size
        # print(w, h)
        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
    

        image_transform=BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        images_seq_mask = []

        image_token = '<image>'
        image_token_id = 128815
        text_splits = prompt.split(image_token)

        images_list, images_crop_list, images_seq_mask = [], [], []
        tokenized_str = []
        images_spatial_crop = []
        for text_sep, image in zip(text_splits, images):

            tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            if crop_mode:

                if image.size[0] <= 768 and image.size[1] <= 768:
                    crop_ratio = [1, 1]

                else:
                    if crop_mode:
                        # best_width, best_height = select_best_resolution(image.size, self.candidate_resolutions)
                        images_crop_raw, crop_ratio = dynamic_preprocess(image)
                    else:
                        # best_width, best_height = self.image_size, self.image_size
                        crop_ratio = [1, 1]
                
                """process the global view"""
                # image = image.resize((base_size, base_size))
                global_view = ImageOps.pad(image, (base_size, base_size),
                                        color=tuple(int(x * 255) for x in image_transform.mean))
                
                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                # elif base_size == 640:
                #     valid_img_tokens += int(100 * ratio)
                



                
                images_list.append(image_transform(global_view).to(torch.bfloat16))

                # global_view_tensor = image_transform(global_view).to(torch.bfloat16)

                width_crop_num, height_crop_num = crop_ratio

                images_spatial_crop.append([width_crop_num, height_crop_num])
                
                
                if width_crop_num > 1 or height_crop_num > 1:
                    """process the local views"""
                    
                    for i in range(len(images_crop_raw)):
                        images_crop_list.append(image_transform(images_crop_raw[i]).to(torch.bfloat16))
                
                if image_size == 768:
                    valid_img_tokens += len(images_crop_list) * 144

                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)



                """add image tokens"""

                

                # ScanFlow Pro V2 returns [V + lambda(V,Q)*R; H] for the global view
                # and every local crop: N_visual residual-modified visual tokens
                # plus T recurrent reasoning states per view.
                num_local_views = (
                    width_crop_num * height_crop_num
                    if (width_crop_num > 1 or height_crop_num > 1)
                    else 0
                )
                global_v2_tokens = num_queries_base ** 2 + int(scanpath_steps)
                local_v2_tokens = num_queries ** 2 + int(scanpath_steps)
                num_scanflow_v2_image_tokens = (
                    global_v2_tokens
                    + num_local_views * local_v2_tokens
                    + 1
                )  # +1 existing view separator
                tokenized_image = [image_token_id] * num_scanflow_v2_image_tokens
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                # num_image_tokens.append(len(tokenized_image))

            else:
                # best_width, best_height = self.image_size, self.image_size
                # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

                """process the global view"""
                if image_size <= 768:
                    print('directly resize')
                    image = image.resize((image_size, image_size))
                # else:
                global_view = ImageOps.pad(image, (image_size, image_size),
                                        color=tuple(int(x * 255) for x in image_transform.mean))
                images_list.append(image_transform(global_view).to(torch.bfloat16))

                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                elif base_size == 640:
                    valid_img_tokens += int(100 * 1)
                elif base_size == 512:
                    valid_img_tokens += int(64 * 1)
                elif base_size == 768:
                    valid_img_tokens += int(144 * 1)

                width_crop_num, height_crop_num = 1, 1

                images_spatial_crop.append([width_crop_num, height_crop_num])


                """add image tokens"""
                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)

                # ScanFlow Pro V2: [V + lambda(V,Q)*R; H] for one global view,
                # plus the existing separator.
                tokenized_image = [image_token_id] * (
                    num_queries ** 2 + int(scanpath_steps) + 1
                )
                # tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                #             num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                # num_image_tokens.append(len(tokenized_image))
        

        """process the last text split"""
        tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos tokens"""
        bos_id = 0
        tokenized_str = [bos_id] + tokenized_str 
        images_seq_mask = [False] + images_seq_mask



        input_ids = torch.LongTensor(tokenized_str)


        

        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)


        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, image_size, image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, base_size, base_size))

        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, base_size, base_size))



        if not eval_mode:
            streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    output_ids = self.generate(
                        input_ids.unsqueeze(0).cuda(),
                        images=[(images_crop.cuda(), images_ori.cuda())],
                        images_seq_mask = images_seq_mask.unsqueeze(0).cuda(),
                        images_spatial_crop = images_spatial_crop,
                        scanpath_steps = int(scanpath_steps),
                        # do_sample=False,
                        # num_beams = 1,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        streamer=streamer,
                        max_new_tokens=8192,
                        no_repeat_ngram_size = 20,
                        use_cache = True
                        )

        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    output_ids = self.generate(
                        input_ids.unsqueeze(0).cuda(),
                        images=[(images_crop.cuda(), images_ori.cuda())],
                        images_seq_mask = images_seq_mask.unsqueeze(0).cuda(),
                        images_spatial_crop = images_spatial_crop,
                        scanpath_steps = int(scanpath_steps),
                        # do_sample=False,
                        # num_beams = 1,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=8192,
                        no_repeat_ngram_size = 35,
                        use_cache = True
                        )
                

        if '<image>' in conversation[0]['content'] and eval_mode:
                outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
                stop_str = '<｜end▁of▁sentence｜>'
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                # re_match
                outputs = outputs.strip()

                return outputs
        
        if '<image>' in conversation[0]['content'] and test_compress:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
            pure_texts_outputs_token_length = len(text_encode(tokenizer, outputs, bos=False, eos=False))
            print('='*50)
            print('image size: ', (w, h))
            print('valid image tokens: ', int(valid_img_tokens))
            print('output texts tokens (valid): ', pure_texts_outputs_token_length)
            print('compression ratio: ', round(pure_texts_outputs_token_length/valid_img_tokens, 2))
            print('='*50)


        if '<image>' in conversation[0]['content'] and save_results:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
            stop_str = '<｜end▁of▁sentence｜>'

            print('='*15 + 'save results:' + '='*15)
            
            # # # # conv.messages[-1][-1] = outputs
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            matches_ref, matches_images, mathes_other = re_match(outputs)
            # print(matches_ref)
            result = process_image_with_refs(image_draw, matches_ref, output_path)


            for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
                outputs = outputs.replace(a_match_image, '![](images/' + str(idx) + '.jpg)\n')
            
            for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
                outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')


            # if 'structural formula' in conversation[0]['content']:
            #     outputs = '<smiles>' + outputs + '</smiles>'
            with open(f'{output_path}/result.mmd', 'w', encoding = 'utf-8') as afile:
                afile.write(outputs)

            if 'line_type' in outputs:
                import matplotlib.pyplot as plt
                lines = eval(outputs)['Line']['line']

                line_type = eval(outputs)['Line']['line_type']
                # print(lines)

                endpoints = eval(outputs)['Line']['line_endpoint']

                fig, ax = plt.subplots(figsize=(3,3), dpi=200)
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)

                for idx, line in enumerate(lines):
                    try:
                        p0 = eval(line.split(' -- ')[0])
                        p1 = eval(line.split(' -- ')[-1])

                        if line_type[idx] == '--':
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                        else:
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth = 0.8, color = 'k')

                        ax.scatter(p0[0], p0[1], s=5, color = 'k')
                        ax.scatter(p1[0], p1[1], s=5, color = 'k')
                    except:
                        pass

                for endpoint in endpoints:

                    label = endpoint.split(': ')[0]
                    (x, y) = eval(endpoint.split(': ')[1])
                    ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', 
                                fontsize=5, fontweight='light')
                

                plt.savefig(f'{output_path}/geo.jpg')
                plt.close()

            result.save(f"{output_path}/result_with_boxes.jpg")
