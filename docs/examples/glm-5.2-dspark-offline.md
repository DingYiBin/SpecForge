# GLM-5.2 MLA+SWA DSpark offline training

This recipe trains a randomly initialized, three-stage GLM-5.2 draft. Each
stage uses GLM MLA projections, interleaved RoPE, a dense SwiGLU MLP, and the
DSpark fixed sliding-window mask. It does not use GLM-5.2's DSA indexer.

The draft consumes hidden states captured from target layers 75, 76, and 77.
Its DSpark block size is 5, the target-context window is 128 tokens, and its
noise token is `[MASK]` (token ID 154821).

## Capture target hidden states

Prepare preformatted GLM-5.2 trajectories, then capture the target features:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  scripts/prepare_hidden_states.py \
  --strategy dspark \
  --target-model-path /path/to/GLM-5.2 \
  --draft-model-config configs/glm-5.2-dspark.json \
  --data-path /path/to/glm-5.2-trajectories.jsonl \
  --output-path /path/to/glm-5.2-hidden-states \
  --is-preformatted \
  --max-length 32768 \
  --response-only \
  --response-context-tokens 128 \
  --sglang-context-length 32768 \
  --tp-size 8 \
  --batch-size 1
```

Use the same GLM-5.2 checkpoint and tokenizer for feature capture and training.
The output must contain `input_ids`, `loss_mask`, the concatenated
`aux_hidden_state` from layers 75–77, and the final target `hidden_state`.

## Train from scratch

The smoke recipe intentionally omits `model.draft_checkpoint_path`, so draft
parameters are initialized from `configs/glm-5.2-dspark.json`. Only the frozen
target embedding and LM head are loaded from the target checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run specforge train \
  --config examples/configs/glm-5.2-dspark-offline-smoke.yaml \
  model.target_model_path=/path/to/GLM-5.2 \
  data.hidden_states_path=/path/to/glm-5.2-hidden-states
```

The configured dense draft is substantially larger than a tiny smoke model.
Keep `FULL_SHARD` and one draft stage per FSDP unit for the first GPU run.

## Train on Ascend NPU

GPU training uses `flex_attention`. Ascend NPU does not provide that kernel,
and the GLM draft must also import without Flex: `Glm52DSparkAttention` now
guards the Flex import the same way DeepSeek-V4 DSpark does. Set
`attention_backend: eager` so the trainer builds a boolean mask with
`create_dflash_sdpa_mask` and runs `Glm52DSparkAttention._dense_attention`
(MLA expanded to 64-head MHA, then split softmax over context `Q×S` and
draft `5×5` blocks). Do not set `flex_attention` on NPU.

A ready-made recipe is
[`examples/configs/glm-5.2-dspark-offline-npu.yaml`](../../examples/configs/glm-5.2-dspark-offline-npu.yaml).
It uses `eager`, `model.attention_chunk_size: 64`, and
`training.activation_checkpointing: attention`. Install the vendor-matched
PyTorch and `torch_npu` packages first, then launch:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

specforge train -c examples/configs/glm-5.2-dspark-offline-npu.yaml \
  model.target_model_path=/path/to/GLM-5.2 \
  data.hidden_states_path=/path/to/glm-5.2-hidden-states
```

The runtime auto-detects NPU via `torch.npu.is_available()` (or honor
`SPECFORGE_DEVICE=npu`). Capture hidden states on an NPU-compatible SGLang
service, or capture on CUDA and point `data.hidden_states_path` at the
exported features. GLM's expanded K/V is larger than DeepSeek-V4's shared-KV
MQA; if attention OOMs, lower `model.attention_chunk_size`,
`training.num_anchors`, or `data.max_length`.

## Resume a complete training run

Resume from a SpecForge checkpoint to restore draft weights, optimizer,
scheduler, data position, and random-number state:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run specforge train \
  --config examples/configs/glm-5.2-dspark-offline-smoke.yaml \
  model.target_model_path=/path/to/GLM-5.2 \
  data.hidden_states_path=/path/to/glm-5.2-hidden-states \
  training.resume_from=./outputs/glm-5.2-dspark-offline-smoke/glm-5.2-dspark-offline-smoke-latest
```

Resume validation rejects changes to the draft block size, stage count, fixed
SWA window, dense-MLP mode, mask token, target layers, or objective semantics.
HF draft warm-start, checkpoint merging, and serving export are outside this
recipe's scope.
