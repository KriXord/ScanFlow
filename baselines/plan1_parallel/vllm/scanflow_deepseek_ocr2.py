# SPDX-License-Identifier: Apache-2.0

# adapted from https://github.com/deepseek-ai/DeepSeek-VL2/blob/faf18023f24b962b32d9f0a2d89e402a8d383a78/deepseek_vl2/models/modeling_deepseek_vl_v2.py
"""Inference-only DeepSeek-OCR2 vLLM wrapper for Parallel Plan 1.

The vLLM integration remains unchanged, while the visual reasoning path mirrors
the trained fixed-query Parallel Plan-1 checkpoint.
"""
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import List, Literal, Optional, Set, Tuple, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.deepseek_vl2 import (DeepseekVLV2Config,
                                                          MlpProjectorConfig,
                                                          VisionEncoderConfig)
from process.image_process import (
    DeepseekOCR2Processor, count_tiles)
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config
# from vllm.utils import is_list_of

from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)

from deepencoderv2.sam_vary_sdpa import build_sam_vit_b
# from deepencoder.clip_sdpa import build_clip_l
# from deepencoderv2.qwen2_d2e import build_qwen2_decoder_as_encoder
from scanflow_deepencoder import build_qwen2_decoder_as_encoder
from scanpath_predictor import VLMScanpathPredictor
from deepencoderv2.build_linear import MlpProjector
from addict import Dict
import time
from pathlib import Path
from config import IMAGE_SIZE, BASE_SIZE, CROP_MODE, PRINT_NUM_VIS_TOKENS, PROMPT
# The image token id may be various
_IMAGE_TOKEN = "<image>"


# Optional request metadata queue for sample-wise hidden-state saving.
# The eval script can push one metadata dict per request immediately before
# llm.generate(...). The model runtime pops one metadata item whenever it saves
# one sample's hidden states. This makes the .pt files directly traceable to
# sample_id / image_id / question_id when running with request_batch_size=1 and
# max_num_seqs=1.
SCANFLOW_REQUEST_METADATA_QUEUE = []


def push_scanflow_request_metadata(metadata):
    """Append one request metadata dict for the next hidden-state save."""
    if metadata is None:
        SCANFLOW_REQUEST_METADATA_QUEUE.append(None)
    else:
        SCANFLOW_REQUEST_METADATA_QUEUE.append(dict(metadata))


def push_scanflow_request_metadata_many(metadatas):
    """Append a sequence of request metadata dicts for upcoming saves."""
    for metadata in metadatas:
        push_scanflow_request_metadata(metadata)


def clear_scanflow_request_metadata_queue():
    """Clear stale request metadata, useful before a fresh eval run."""
    SCANFLOW_REQUEST_METADATA_QUEUE.clear()


def _pop_scanflow_request_metadata():
    if SCANFLOW_REQUEST_METADATA_QUEUE:
        return SCANFLOW_REQUEST_METADATA_QUEUE.pop(0)
    return None


def build_scanflow_qwen2_encoder():
    """Build the fixed-query Parallel Plan-1 Qwen visual encoder."""
    return build_qwen2_decoder_as_encoder()


class DeepseekOCR2ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(DeepseekVLV2Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(DeepseekOCR2Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(self,
                             *,
                             image_width: int,
                             image_height: int,
                             cropping: bool = True) -> int:
        hf_processor = self.get_hf_processor()


        # image_size = hf_processor.image_size
        # patch_size = hf_processor.patch_size
        # downsample_ratio = hf_processor.downsample_ratio

        image_size = IMAGE_SIZE
        base_size = BASE_SIZE
        patch_size = 16
        downsample_ratio = 4

        if CROP_MODE:
            if image_width <= 768 and image_height <= 768:
                crop_ratio = [1, 1]
            else:
                # images_crop_raw, crop_ratio = hf_processor.dynamic_preprocess(image)

                # find the closest aspect ratio to the target
                crop_ratio = count_tiles(image_width, image_height, image_size=IMAGE_SIZE)

                # print('===========')
                # print('crop_ratio ', crop_ratio)
                # print('============')
                
            num_width_tiles, num_height_tiles = crop_ratio
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((base_size // patch_size) / downsample_ratio)

        h2 = w2 = math.ceil((image_size // patch_size) / downsample_ratio)

        global_views_tokens = h * (w)
        if num_width_tiles >1 or num_height_tiles>1:
            local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2)
        else:
            local_views_tokens = 0


        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:

        if IMAGE_SIZE == 1024 and BASE_SIZE == 1280:
            return ImageSize(width=1024*2, height=1024*2)
        return ImageSize(width=768*2, height=768*2)


class DeepseekOCR2DummyInputsBuilder(
        BaseDummyInputsBuilder[DeepseekOCR2ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        max_image_size = self.info.get_image_size_with_most_features()

        if '<image>' in PROMPT:
            return {
                "image":
                DeepseekOCR2Processor().tokenize_with_images(images = self._get_dummy_images(width=max_image_size.width,
                                    height=max_image_size.height,
                                    num_images=num_images), conversation = PROMPT, bos=True, eos=True, cropping=CROP_MODE)
            }
        else:
            return {
                "image": []
            }




class DeepseekOCR2MultiModalProcessor(
        BaseMultiModalProcessor[DeepseekOCR2ProcessingInfo]):
    

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        
        
        # print(mm_data)
        if mm_data:
            processed_outputs = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**mm_kwargs),
                dict(prompt=prompt, **mm_data),
                mm_kwargs,
            )

        else:
            tokenizer = self.info.get_tokenizer()
            processed_outputs = tokenizer(prompt,
                                          add_special_tokens=True,
                                          return_tensors="pt")

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            images_spatial_crop=MultiModalFieldConfig.batched("image"),
            # image_embeds=MultiModalFieldConfig.batched("image2"),
            images_crop=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement_deepseek_vl2(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))



            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:

                
                width = images[0][-1][0][0]
                height = images[0][-1][0][1]

                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=width,
                    image_height=height,
                    # flag = True,
                    cropping=CROP_MODE,
                )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_deepseek_vl2,
            )
        ]

    def _cached_apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> tuple[list[int], MultiModalKwargs, bool]:
        # The processor logic is different for len(images) <= 2 vs > 2
        # Since the processing cache assumes that the processor output is
        # invariant of how many images are passed per prompt, we only
        # perform caching for the most common case
        if mm_data_items.get_count("image", strict=False) > 2:
            # This code path corresponds to the cache being disabled
            return self._apply_hf_processor_main(
                prompt=prompt,
                mm_items=mm_data_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                enable_hf_prompt_update=True,
            )

        return super()._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCR2MultiModalProcessor,
    info=DeepseekOCR2ProcessingInfo,
    dummy_inputs=DeepseekOCR2DummyInputsBuilder)
class DeepseekOCR2ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "language.": "language_model.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: DeepseekVLV2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        # config.model_type ='deepseek_vl_v2'

        self.config = config
        self.multimodal_config = multimodal_config


        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        # self.text_config = config.text_config

        self.text_config = getattr(config, "text_config", None)
        if self.text_config is None:
            self.text_config = getattr(config, "language_config", None)

        if self.text_config is None:
            raise AttributeError(
                "Expected config.text_config or config.language_config for DeepSeek-OCR2 language model."
            )

        if isinstance(self.text_config, dict):
            text_config_dict = dict(self.text_config)

            # language_config in this checkpoint's config.json is a partial dict.
            # Some missing fields happen to match DeepseekV2Config's own defaults
            # (hidden_act, rms_norm_eps, rope_theta, ...) but two do NOT:
            # pad_token_id (class default None vs. this model's 2) and use_cache
            # (class default True vs. this model's False). Backfill explicitly from
            # the parent config rather than trusting the class defaults for anything
            # that's actually specified at the top level.
            for key in (
                "hidden_act", "rms_norm_eps", "rope_theta", "rope_scaling",
                "attention_bias", "attention_dropout", "initializer_range",
                "pretraining_tp", "routed_scaling_factor", "scoring_func",
                "seq_aux", "norm_topk_prob", "aux_loss_alpha", "moe_layer_freq",
                "pad_token_id", "tie_word_embeddings", "use_cache",
            ):
                if key not in text_config_dict and hasattr(config, key):
                    text_config_dict[key] = getattr(config, key)

            from configuration_deepseek_v2 import DeepseekV2Config
            self.text_config = DeepseekV2Config(**text_config_dict)

        model_config = vllm_config.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        self.image_token_id = tokenizer.vocab[_IMAGE_TOKEN]

        self.sam_model = build_sam_vit_b()
        self.qwen2_model = build_scanflow_qwen2_encoder()

        # Parallel Plan 1 has no legacy text-query residual switch. Question
        # conditioning happens inside qwen2_model after every Qwen layer.
        self.enable_text_query_residual = False

        n_embed = 1280
        self.projector = MlpProjector(
            Dict(projector_type="linear", input_dim=896, n_embed=n_embed)
        )

        # Frozen pretrained autoregressive scanpath predictor. It does not alter
        # QA generation, but it is part of the trained checkpoint and must exist
        # so model.scanpath_predictor.* weights load into the correct module.
        self.scanpath_predictor = VLMScanpathPredictor(
            prefix_hidden_dim=int(getattr(self.text_config, "hidden_size", n_embed)),
            scanpath_hidden_dim=896,
            visual_dim=896,
            num_heads=14,
            intermediate_dim=4864,
            num_fixation_cells=256,
            dropout=float(os.getenv("SCANPATH_PREDICTOR_DROPOUT", "0.0")),
            temperature=float(os.getenv("SCANPATH_PREDICTOR_TEMPERATURE", "0.2")),
        )

        self.qwen2_model.query_768.weight.requires_grad_(False)
        self.qwen2_model.query_1024.weight.requires_grad_(False)
        for parameter in self.scanpath_predictor.parameters():
            parameter.requires_grad_(False)

        # Architecture-faithful caches matching the HF Plan 1 Parallel wrapper.
        self.last_scanpath_predictor_visual_tokens = None
        self.last_parallel_plan1_causal_tokens = None
        self.last_scanpath_predictor_logits = None
        self.last_scanpath_predictor_probs = None
        self.last_scanpath_predictor_states = None
        self.last_scanpath_predictor_evidence = None
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos
    
        # self.sam_model = torch.compile(self.sam_model, mode="reduce-overhead")
        # self.vision_model = torch.compile(self.vision_model, mode="reduce-overhead")
        # self.projector = torch.compile(self.projector, mode="max-autotune")




        # special token for image token sequence format
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            # self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )
        
        topk_method = getattr(self.text_config, "topk_method", "greedy")
        use_mla = getattr(self.text_config, "use_mla", False)

        if self.text_config.topk_method == "noaux_tc":
            architectures = ["DeepseekV3ForCausalLM"]
            # architectures = ["DeepseekForCausalLM"]
        elif not self.text_config.use_mla:
            architectures = ["DeepseekForCausalLM"]
        else:
            architectures = ["DeepseekV2ForCausalLM"]

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=self.text_config,
            prefix=maybe_prefix(prefix, "language"),
            architectures=architectures,
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)
        


        self.sam_model.to(dtype=torch.bfloat16)



    def _parse_and_validate_image_input(
            self, **kwargs: object):
        
        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)


        if pixel_values is None or torch.sum(pixel_values).item() == 0:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(images_spatial_crop, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(images_spatial_crop)}")
            
            if not isinstance(images_crop, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image crop. "
                                 f"Got type: {type(images_crop)}")

            return [pixel_values, images_crop, images_spatial_crop]


        raise AssertionError("This line should be unreachable.")
    
    @staticmethod
    def _prepare_scanpath_predictor_visual_tokens(
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return the ordered global 16x16 SAM grid as [B, 256, 896]."""
        if global_features.dim() != 4:
            raise ValueError(
                "Expected global SAM features [B,D,H,W], got "
                f"{tuple(global_features.shape)}."
            )
        visual_tokens = global_features.flatten(2).transpose(1, 2)
        if visual_tokens.size(1) != 256:
            raise ValueError(
                "Parallel Plan 1 requires 256 global SAM visual tokens, got "
                f"{visual_tokens.size(1)}."
            )
        if visual_tokens.size(-1) != 896:
            raise ValueError(
                "Parallel Plan 1 predictor expects visual_dim=896, got "
                f"{visual_tokens.size(-1)}."
            )
        return visual_tokens

    def _select_text_context(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if inputs_embeds is None:
            return None

        token_source = inputs_embeds.detach()
        if token_source.dim() == 3:
            token_source = token_source.reshape(-1, token_source.size(-1))
        elif token_source.dim() != 2:
            return None

        if input_ids is not None:
            flat_input_ids = input_ids.reshape(-1).to(device=token_source.device)
            if flat_input_ids.numel() == token_source.size(0):
                mask = flat_input_ids.ne(self.image_token_id)
                if mask.sum() > 0:
                    token_source = token_source[mask]

        if token_source.numel() == 0:
            return None
        return token_source.unsqueeze(0)

    @staticmethod
    def _expand_text_context(
        text_context: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if text_context is None:
            return None
        return text_context.expand(batch_size, -1, -1)

    def _run_qwen2_encoder(
        self,
        visual_features: torch.Tensor,
        text_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the fixed-query Parallel Plan-1 causal encoder."""
        try:
            return self.qwen2_model(
                visual_features,
                text_context=text_context,
            )
        except TypeError as exc:
            raise TypeError(
                "scanflow_deepencoder.py is not the Parallel Plan-1 runtime. "
                "Expected qwen2_model(visual_features, text_context=...)."
            ) from exc

    def _scanflow_get_rank(self) -> int:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank())
        except Exception:
            pass

        for key in ("RANK", "LOCAL_RANK"):
            value = os.environ.get(key)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        return 0


    def _scanflow_should_save_hidden_states(self) -> bool:
        if os.getenv("SCANFLOW_SAVE_HIDDEN_STATES", "0") != "1":
            return False

        if os.getenv("SCANFLOW_SAVE_ALL_RANKS", "0") == "1":
            return True

        return self._scanflow_get_rank() == 0


    @staticmethod
    def _scanflow_to_cpu_tensor(x, *, float_dtype=torch.float16):
        if x is None:
            return None

        x = x.detach()
        if torch.is_floating_point(x):
            x = x.to(dtype=float_dtype)
        return x.cpu()


    @staticmethod
    def _scanflow_batch_size_from_tensors(*tensors) -> int:
        """Infer sample count from tensors shaped [B, ...]."""
        batch_size = 1
        for tensor in tensors:
            if torch.is_tensor(tensor) and tensor.dim() > 0:
                batch_size = max(batch_size, int(tensor.shape[0]))
        return batch_size

    @staticmethod
    def _scanflow_select_batch_item(x, batch_idx: int, batch_size: int):
        """Select one sample from a [B, ...] tensor when possible."""
        if x is None:
            return None
        if torch.is_tensor(x):
            # Only split along dim 0 when it clearly represents the same batch.
            if x.dim() > 0 and int(x.shape[0]) == batch_size:
                return x[batch_idx : batch_idx + 1]
            return x
        if isinstance(x, (list, tuple)) and len(x) == batch_size:
            return x[batch_idx]
        return x

    def _maybe_save_scanflow_hidden_states(
        self,
        *,
        view: str,
        qwen_visual_output: torch.Tensor | None = None,
        crop_shape: torch.Tensor | None = None,
    ) -> None:
        """Optionally save architecture-faithful Parallel Plan-1 runtime tensors."""
        if not self._scanflow_should_save_hidden_states():
            return

        save_dir = os.getenv("SCANFLOW_HIDDEN_STATE_DIR", "")
        if not save_dir:
            return

        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        causal_tokens = self.last_parallel_plan1_causal_tokens
        predictor_visual_tokens = self.last_scanpath_predictor_visual_tokens

        batch_size = self._scanflow_batch_size_from_tensors(
            causal_tokens,
            predictor_visual_tokens,
            qwen_visual_output,
            crop_shape,
        )

        rank = self._scanflow_get_rank()
        pid = os.getpid()
        require_metadata = os.getenv("SCANFLOW_SAVE_ONLY_WITH_METADATA", "0") == "1"

        for batch_idx in range(batch_size):
            request_metadata = _pop_scanflow_request_metadata()
            if require_metadata and request_metadata is None:
                continue

            counter = getattr(self, "_scanflow_hidden_save_counter", 0)
            self._scanflow_hidden_save_counter = counter + 1
            timestamp_ms = int(time.time() * 1000)

            def clean_name(value):
                if value is None:
                    return None
                value = str(value)
                return "".join(
                    ch if (ch.isalnum() or ch in "._-") else "_"
                    for ch in value
                )[:120]

            sample_id = None
            image_id = None
            question_id = None
            if isinstance(request_metadata, dict):
                sample_id = request_metadata.get("sample_id")
                image_id = request_metadata.get("image_id")
                question_id = request_metadata.get("question_id")

            name_parts = [f"{counter:06d}", f"rank{rank}", f"pid{pid}", view]
            if clean_name(sample_id):
                name_parts.append(f"sample-{clean_name(sample_id)}")
            else:
                if clean_name(image_id):
                    name_parts.append(f"image-{clean_name(image_id)}")
                if clean_name(question_id):
                    name_parts.append(f"question-{clean_name(question_id)}")
            name_parts.append(str(timestamp_ms))
            save_path = save_dir_path / ("_".join(name_parts) + ".pt")

            crop_shape_item = self._scanflow_select_batch_item(
                crop_shape, batch_idx, batch_size
            )
            if torch.is_tensor(crop_shape_item):
                crop_shape_item = crop_shape_item.detach().cpu().tolist()

            payload = {
                "architecture": "plan1-parallel",
                "view": view,
                "counter": counter,
                "rank": rank,
                "pid": pid,
                "timestamp_ms": timestamp_ms,
                "request_metadata": request_metadata,
                "parallel_plan1_causal_tokens": self._scanflow_to_cpu_tensor(
                    self._scanflow_select_batch_item(
                        causal_tokens, batch_idx, batch_size
                    )
                ),
                "scanpath_predictor_visual_tokens": self._scanflow_to_cpu_tensor(
                    self._scanflow_select_batch_item(
                        predictor_visual_tokens, batch_idx, batch_size
                    )
                ),
                "qwen_visual_output": self._scanflow_to_cpu_tensor(
                    self._scanflow_select_batch_item(
                        qwen_visual_output, batch_idx, batch_size
                    )
                ),
                "crop_shape": crop_shape_item,
            }
            torch.save(payload, save_path)
            print(f"[plan1 parallel hidden save] {save_path}", flush=True)

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
    ) -> NestedTensors:
        """Encode images with the exact Parallel Plan-1 visual reasoning path."""

        images_in_this_batch = []

        self.last_scanpath_predictor_visual_tokens = None
        self.last_parallel_plan1_causal_tokens = None
        self.last_scanpath_predictor_logits = None
        self.last_scanpath_predictor_probs = None
        self.last_scanpath_predictor_states = None
        self.last_scanpath_predictor_evidence = None

        batch_predictor_visual_tokens = []
        batch_parallel_plan1_causal_tokens = []

        # vLLM inference never trains the visual stack.
        with torch.no_grad():
            for jdx in range(images_spatial_crop.size(0)):
                patches = images_crop[jdx][0].to(torch.bfloat16)
                image_ori = pixel_values[jdx]
                crop_shape = images_spatial_crop[jdx][0]

                if torch.sum(patches).item() != 0:
                    local_features_1 = self.sam_model(patches)
                    global_features_1 = self.sam_model(image_ori)

                    batch_predictor_visual_tokens.append(
                        self._prepare_scanpath_predictor_visual_tokens(
                            global_features_1
                        ).detach()
                    )

                    local_text_context = self._expand_text_context(
                        text_context,
                        local_features_1.shape[0],
                    )
                    global_text_context = self._expand_text_context(
                        text_context,
                        global_features_1.shape[0],
                    )

                    local_features_2 = self._run_qwen2_encoder(
                        local_features_1,
                        text_context=local_text_context,
                    )
                    local_features = self.projector(local_features_2)

                    global_features_2 = self._run_qwen2_encoder(
                        global_features_1,
                        text_context=global_text_context,
                    )
                    global_features = self.projector(global_features_2)

                    # Exact projected global causal prefix supervised during
                    # Parallel Plan-1 representation learning.
                    batch_parallel_plan1_causal_tokens.append(
                        global_features.detach()
                    )

                    # Populate caches before optional saving.
                    self.last_scanpath_predictor_visual_tokens = torch.cat(
                        batch_predictor_visual_tokens, dim=0
                    )
                    self.last_parallel_plan1_causal_tokens = torch.cat(
                        batch_parallel_plan1_causal_tokens, dim=0
                    )

                    self._maybe_save_scanflow_hidden_states(
                        view="global",
                        qwen_visual_output=global_features_2,
                        crop_shape=crop_shape,
                    )

                    if PRINT_NUM_VIS_TOKENS:
                        print("=====================")
                        print("BASE: ", global_features.shape)
                        print("PATCHES: ", local_features.shape)
                        print("=====================")

                    _, _, n_dim = global_features.shape
                    _, _, n_dim_local = local_features.shape
                    global_features = global_features.view(-1, n_dim)
                    local_features = local_features.view(-1, n_dim_local)

                    global_local_features = torch.cat(
                        [
                            local_features,
                            global_features,
                            self.view_seperator[None, :],
                        ],
                        dim=0,
                    )
                else:
                    global_features_1 = self.sam_model(image_ori)

                    batch_predictor_visual_tokens.append(
                        self._prepare_scanpath_predictor_visual_tokens(
                            global_features_1
                        ).detach()
                    )

                    global_text_context = self._expand_text_context(
                        text_context,
                        global_features_1.shape[0],
                    )
                    global_features_2 = self._run_qwen2_encoder(
                        global_features_1,
                        text_context=global_text_context,
                    )
                    global_features = self.projector(global_features_2)

                    batch_parallel_plan1_causal_tokens.append(
                        global_features.detach()
                    )

                    self.last_scanpath_predictor_visual_tokens = torch.cat(
                        batch_predictor_visual_tokens, dim=0
                    )
                    self.last_parallel_plan1_causal_tokens = torch.cat(
                        batch_parallel_plan1_causal_tokens, dim=0
                    )

                    self._maybe_save_scanflow_hidden_states(
                        view="global",
                        qwen_visual_output=global_features_2,
                        crop_shape=crop_shape,
                    )

                    if PRINT_NUM_VIS_TOKENS:
                        print("=====================")
                        print("BASE: ", global_features.shape)
                        print("NO PATCHES")
                        print("=====================")

                    _, _, n_dim = global_features.shape
                    global_features = global_features.view(-1, n_dim)
                    global_local_features = torch.cat(
                        [global_features, self.view_seperator[None, :]],
                        dim=0,
                    )

                images_in_this_batch.append(global_local_features)

        if batch_predictor_visual_tokens:
            self.last_scanpath_predictor_visual_tokens = torch.cat(
                batch_predictor_visual_tokens,
                dim=0,
            )
        if batch_parallel_plan1_causal_tokens:
            self.last_parallel_plan1_causal_tokens = torch.cat(
                batch_parallel_plan1_causal_tokens,
                dim=0,
            )

        return images_in_this_batch

    def _process_image_input(
            self,
            image_input,
            text_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        

        # image_input: [pixel_values, images_crop, images_spatial_crop]
    
        pixel_values = image_input[0].to(torch.bfloat16)

        # images_crop = image_input[1].to(torch.bfloat16)
        images_crop = image_input[1]
        # images_crop = image_input[1]
        images_spatial_crop = image_input[2].to(dtype=torch.long)

        # local_start = time.time()
        vision_features = self._pixel_values_to_embedding(
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
            text_context=text_context,
        )


        return vision_features

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self,
            text_context: Optional[torch.Tensor] = None,
            **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(
            image_input,
            text_context=text_context,
        )
        return vision_embeddings
    


    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        


        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        # input_ids.to(torch.bfloat16)
        # self.image_token_id.to(torch.bfloat16)

        if multimodal_embeddings is not None:
            # multimodal_embeddings = multimodal_embeddings.to(torch.bfloat16)
            # multimodal_embeddings = [emb.to(torch.bfloat16) for emb in multimodal_embeddings]
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.image_token_id)

            
        return inputs_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object):

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility
        elif inputs_embeds is None:
            text_inputs_embeds = self.language_model.get_input_embeddings(input_ids)
            text_context = self._select_text_context(input_ids, text_inputs_embeds)
            vision_embeddings = self.get_multimodal_embeddings(
                text_context=text_context,
                **kwargs,
            )
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            intermediate_tensors,
                                            inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        processed_weights = []
        
        for name, tensor in weights:
            if (
                'sam_model' in name
                or 'qwen2_model' in name
                or 'projector' in name
                or 'view_seperator' in name
                or 'scanpath_predictor' in name
            ):
                new_name = name.replace('model.', '', 1)
            else:
                new_name = 'language.' + name
            
            # tensor = tensor.to(torch.bfloat16)

            processed_weights.append((new_name, tensor))
        
        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(processed_weights, mapper=self.hf_to_vllm_mapper)





        return autoloaded_weights