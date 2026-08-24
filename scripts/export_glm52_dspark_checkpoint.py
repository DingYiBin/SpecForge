# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export a standalone GLM-5.2 DSpark draft Hugging Face directory.

Writes trained ``mtp.*`` tensors plus a serving ``config.json`` and tokenizer
files hardlinked from the target (copy fallback). Does not overlay weights
onto GLM-5.2-NVFP4.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from specforge.export.checkpoint_io import resolve_training_state

DEFAULT_AUX_LAYER_IDS = [75, 76, 77]
GLM_HIDDEN_SIZE = 6144
TOKENIZER_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
    "tokenization_glm.py",
    "tokenization_glm_moe_dsa.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _copy_or_link_file(source: Path, target: Path, *, use_hardlink: bool) -> None:
    """Hardlink like DSV4 export; fall back to copy across filesystems."""
    if use_hardlink:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _copy_tokenizer(
    target_dir: Path, output_dir: Path, *, copy_tokenizer: bool
) -> None:
    copied = False
    for name in TOKENIZER_NAMES:
        source = target_dir / name
        if source.is_file():
            _copy_or_link_file(
                source, output_dir / name, use_hardlink=not copy_tokenizer
            )
            copied = True
    if not copied:
        raise FileNotFoundError(
            f"no tokenizer files found under {target_dir}; "
            "pass --target-model pointing at the GLM tokenizer directory"
        )


def _infer_num_layers(tensors: dict[str, torch.Tensor]) -> int:
    indices = set()
    for key in tensors:
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "mtp" and parts[1].isdecimal():
            indices.add(int(parts[1]))
    if not indices:
        raise ValueError("checkpoint contains no mtp.{i}.* tensors")
    return max(indices) + 1


def _drop_target_layer_lists(config: dict[str, Any], *, num_layers: int) -> None:
    """Drop 78-wide GLM target lists; draft is 3 dense SWA layers."""
    original_layers = int(config.get("num_hidden_layers") or 0)
    for key, value in list(config.items()):
        if not isinstance(value, list):
            continue
        drop = False
        if key.endswith("layer_types") or key == "moe_layer_freq":
            drop = len(value) != num_layers
        elif original_layers and original_layers != num_layers and len(value) == original_layers:
            drop = True
        if drop:
            config.pop(key, None)
    config["layer_types"] = ["sliding_attention"] * num_layers
    config["mlp_layer_types"] = ["dense"] * num_layers


def _load_mtp_tensors(checkpoint: str) -> dict[str, torch.Tensor]:
    state = resolve_training_state(checkpoint)
    draft_state = state.get("draft_state_dict")
    if not isinstance(draft_state, dict):
        raise ValueError(f"checkpoint has no draft_state_dict: {checkpoint}")
    tensors: dict[str, torch.Tensor] = {}
    for key, value in draft_state.items():
        if not key.startswith("mtp.") or not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16)
        tensors[key] = tensor
    if not tensors:
        raise ValueError(f"checkpoint contains no mtp.* tensor: {checkpoint}")
    return tensors


def _build_config(
    *,
    target_config: dict[str, Any],
    draft_config: dict[str, Any] | None,
    num_layers: int,
    main_proj_in: int,
) -> dict[str, Any]:
    config = dict(target_config)
    if draft_config:
        for key in (
            "dflash_config",
            "sliding_window",
            "draft_vocab_size",
            "attention_chunk_size",
            "rope_interleave",
        ):
            if key in draft_config:
                config[key] = draft_config[key]
        config.setdefault("dflash_config", draft_config.get("dflash_config", {}))

    method = dict(config.get("dflash_config") or {})
    aux_ids = list(method.get("target_layer_ids") or DEFAULT_AUX_LAYER_IDS)
    if len(aux_ids) != 3:
        raise ValueError(
            "GLM-5.2 DSpark serving expects 3 aux layers "
            f"(got dflash_config.target_layer_ids={aux_ids})"
        )
    expected_in = len(aux_ids) * int(config.get("hidden_size") or GLM_HIDDEN_SIZE)
    if main_proj_in != expected_in:
        raise ValueError(
            "mtp.0.main_proj in-features must equal "
            f"{len(aux_ids)} * hidden_size ({expected_in}), got {main_proj_in}"
        )
    method.setdefault("num_layers", num_layers)
    method.setdefault("target_layer_ids", aux_ids)
    method.setdefault("mlp_type", "dense")
    if method.get("mask_token_id") is None:
        raise ValueError(
            "dflash_config.mask_token_id is required for DSpark serving"
        )

    hidden_size = int(config["hidden_size"])
    num_layers = int(method["num_layers"])
    # Target GLM-5.2 config still has 78-layer fields (layer_types,
    # mlp_layer_types, NVFP4 quantization, native MTP). A 3-layer dense
    # draft cannot keep them: transformers rejects mismatched *layer_types.
    for key in (
        "quantization_config",
        "quantization",
        "num_nextn_predict_layers",
        "auto_map",
    ):
        config.pop(key, None)
    _drop_target_layer_lists(config, num_layers=num_layers)
    config["architectures"] = ["Glm52DSparkDraftModel"]
    config["model_type"] = "glm52_dspark"
    config["dflash_config"] = method
    config["eagle_aux_hidden_state_layer_ids"] = aux_ids
    config["target_layer_ids"] = aux_ids
    config["num_target_layers"] = len(aux_ids)
    config["target_hidden_size"] = hidden_size
    config["num_hidden_layers"] = num_layers
    config["n_routed_experts"] = 0
    config["n_shared_experts"] = 0
    config["num_experts_per_tok"] = 0
    config["draft_vocab_size"] = int(
        config.get("draft_vocab_size") or config["vocab_size"]
    )
    config["markov_rank"] = int(
        method.get("markov_rank") or config.get("markov_rank") or 0
    )
    if config["markov_rank"] <= 0:
        raise ValueError("dflash_config.markov_rank must be positive")
    config["n_predict"] = int(
        method.get("block_size") or config.get("n_predict") or 5
    )
    config["sliding_window"] = int(config.get("sliding_window") or 128)
    config["torch_dtype"] = "bfloat16"
    config["dtype"] = "bfloat16"
    return config


def export_glm52_dspark_checkpoint(
    *,
    checkpoint: str,
    output_dir: str,
    target_model: str,
    draft_config: str | None,
    overwrite: bool,
    copy_tokenizer: bool = False,
) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    target_dir = Path(target_model).expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    tensors = _load_mtp_tensors(checkpoint)
    main_proj = tensors.get("mtp.0.main_proj.weight")
    if main_proj is None:
        raise ValueError("checkpoint is missing mtp.0.main_proj.weight")
    num_layers = _infer_num_layers(tensors)

    target_config_path = target_dir / "config.json"
    if not target_config_path.is_file():
        raise FileNotFoundError(f"missing target config.json: {target_config_path}")
    target_config = _load_json(target_config_path)

    extra_config = None
    if draft_config:
        extra_config = _load_json(Path(draft_config).expanduser().resolve())
    else:
        for candidate in (
            Path(checkpoint).expanduser() / "config.json",
            Path(checkpoint).expanduser().parent / "config.json",
        ):
            if candidate.is_file():
                extra_config = _load_json(candidate)
                break

    config = _build_config(
        target_config=target_config,
        draft_config=extra_config,
        num_layers=num_layers,
        main_proj_in=int(main_proj.shape[-1]),
    )
    with (output_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    save_file(tensors, str(output_path / "model.safetensors"))
    _copy_tokenizer(target_dir, output_path, copy_tokenizer=copy_tokenizer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="SpecForge run or step dir")
    parser.add_argument("--output-dir", required=True, help="Standalone draft HF dir")
    parser.add_argument(
        "--target-model",
        required=True,
        help="GLM-5.2 tokenizer/config directory (not overlayed)",
    )
    parser.add_argument(
        "--draft-config",
        help="Optional SpecForge GLM DSpark config.json (for dflash_config)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-tokenizer",
        action="store_true",
        help="Copy tokenizer files instead of hardlinking them (DSV4 --copy-base-weights).",
    )
    args = parser.parse_args()
    export_glm52_dspark_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        target_model=args.target_model,
        draft_config=args.draft_config,
        overwrite=args.overwrite,
        copy_tokenizer=args.copy_tokenizer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
