# GLM-5.2 windowed online DSpark

This recipe streams the same samples produced by the repository's
`loop_curl.py` workflow without writing offline `.ckpt` batches.

## Data contract

`data.parent_bin_path` is a flat little-endian int32 token file. It is treated
as a ring. One capture parent contains

```text
1024 + 47 * (1024 - 128) = 43136 tokens
```

The parent is sent to the external capture server as one prompt. The server
writes full-length `hidden_states` and `target_last_hidden_states` tensors into
Mooncake and returns their normal `spec_capture` metadata. SpecForge then reads
the parent once and writes 48 independent child samples:

```text
child i = parent[i * 896 : i * 896 + 1024]
```

Child zero has an all-one loss mask. Every later child masks its first 128
overlap tokens and supervises its remaining 896 tokens.

## Runtime requirements

- Configure exactly one external capture server. The initial implementation
  keeps one parent capture in flight and publishes children in order.
- The server, producer, and consumer must use the same Mooncake metadata/master
  endpoints and store id.
- The producer needs a positive Mooncake segment because it owns the child
  objects. For GLM-5.2 BF16 with three auxiliary layers, one 48-child group is
  about 2.25 GiB before allocator overhead.
- `runtime.producer_lease` must be `1`.
- `save_interval * dp_size * batch_size * accumulation_steps` and the configured
  final step must be divisible by `children_per_parent` (48).

See
`examples/configs/glm-5.2-dspark-windowed-online-npu.yaml` for a complete
configuration.

## Shuffle and resume

Parent blocks, not child windows, are shuffled. Epoch `e` uses NumPy PCG64 with
`training.seed + e`; children inside a parent remain ordered.

A consumer checkpoint stores the dataset fingerprint and window/shuffle
contract. To resume, start both producer and consumer with that checkpoint and
fresh control/consumer-state directories. The producer derives the number of
fully consumed parents from:

```text
global_step * dp_size * batch_size * accumulation_steps / 48
```

It recreates the epoch permutation and starts at the next parent. A retained
consumer SQLite database is intentionally not reused for this coordinated
windowed replay.
