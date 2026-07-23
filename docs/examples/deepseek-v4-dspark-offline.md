# DeepSeek-V4 DSpark offline training

The offline path stores target features once and trains only the three DSpark
layers. The trainer additionally loads the frozen target embedding and LM head;
it does not construct the 43-layer target decoder.

## Prepare target features

```bash
python -m torch.distributed.run --nproc_per_node=8 \
  scripts/prepare_hidden_states.py \
  --strategy dspark \
  --target-model-path /path/to/DeepSeek-V4-Flash-DSpark \
  --draft-model-config configs/deepseek-v4-flash-dspark.json \
  --data-path /path/to/train.jsonl \
  --output-path /path/to/dspark-hidden-states \
  --max-length 4096 \
  --tp-size 8 \
  --batch-size 8
```

The configured target layers are `40`, `41`, and `42`. Each record contains
`input_ids`, `loss_mask`, their concatenated `aux_hidden_state`, and the target
`hidden_state` before the LM head.

## Train

Use the following fields in the training YAML:

```yaml
model:
  target_model_path: /path/to/DeepSeek-V4-Flash-DSpark
  draft_model_config: configs/deepseek-v4-flash-dspark.json
  draft_checkpoint_path: /path/to/DeepSeek-V4-Flash-DSpark
  torch_dtype: bfloat16

data:
  hidden_states_path: /path/to/dspark-hidden-states
  max_length: 4096

training:
  strategy: dspark
  attention_backend: flex_attention
  batch_size: 1
  num_anchors: 512
  dspark_ce_loss_alpha: 0.1
  dspark_l1_loss_alpha: 0.9
  dspark_confidence_head_alpha: 1.0

deployment:
  mode: local_colocated
  trainer:
    nnodes: 1
    nproc_per_node: 8
```

The DSv4 loader reads only `mtp.*` tensors from the full HF checkpoint. Packed
FP4 expert tensors and FP8 tensors are dequantized to the configured training
dtype while loading. The checkpoint, tokenizer, captured layer IDs, and hidden
state convention must come from the same model release.

## Train on Ascend NPU

DeepSeek-V4 DSpark attention hard-requires `flex_attention` on the CUDA path,
which is unavailable on Ascend NPU. Set `attention_backend: sdpa` to select the
dense fallback (`DeepseekV4DSparkAttention._dense_attention`): it consumes the
boolean mask from `create_dflash_sdpa_mask` with no Q/KV length constraint, and
splits the softmax at the `[context | draft]` boundary so the block-diagonal
draft region is computed as `N` independent `bs x bs` attentions instead of the
wasteful full `Q x Q` product. The learned attention sink reuses the same
logsumexp as a sigmoid gate, matching the flex path numerically.

A ready-made recipe is
[`examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml`](../../examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml).
Install the vendor-matched PyTorch and `torch_npu` packages first, then launch:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

specforge train -c examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml
```

The runtime auto-detects NPU via `torch.npu.is_available()` (or honor
`SPECFORGE_DEVICE=npu`). The `prepare_hidden_states.py` capture step is a
separate SGLang-backed pipeline; run it on an NPU-compatible SGLang service
(or capture on CUDA and point `data.hidden_states_path` at the exported
features) before training.

