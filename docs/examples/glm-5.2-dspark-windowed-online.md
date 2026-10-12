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

## Data plane (A': Mooncake parents + spill-file children)

Only the **parent** travels through Mooncake (capture server -> producer, one
~2.8 GiB read per parent). Children are written by the producer as plain
`.ckpt` files under `deployment.disaggregated.spill_root`:

```text
{spill_root}/{run_id}/{parent_id}/w{i:04d}.ckpt
```

and published as `file://` sample refs. Trainer ranks load them with
`torch.load(mmap=True)` through the same `file://` path the offline reader
uses: same-node ranks share the OS page cache, so the physical read happens
once per child per node and cross-machine child traffic is zero. Physical
release at the optimizer-boundary durable ack is a file delete — no Mooncake
read lease, no `-706` retry window for children.

Steady-state spill occupancy is the in-flight (published-but-unacked) child
bytes: with the suggested watermarks (192/96 in-flight refs) that is a few GiB
of tmpfs. `runtime.feature_store_max_resident_bytes` bounds the directory and
turns consumer-lag into a loud producer-side `MemoryError` (backpressure /
hard cap) instead of a full disk.

## Runtime requirements

- Configure exactly one external capture server. The initial implementation
  keeps one parent capture in flight and publishes children in order.
- The server, producer, and consumer must use the same Mooncake metadata/master
  endpoints and store id.
- `deployment.disaggregated.spill_root` is required and must be shared by the
  producer and every trainer rank — with `nnodes=1` a node-local `/tmp` (e.g.
  a 1 TB tmpfs) is the intended choice. Multi-node deployments need the same
  path to resolve to the same physical storage.
- All peers must agree on one transport: set `deployment.disaggregated.
  mooncake_protocol` (and the capture server's `MOONCAKE_PROTOCOL`) to the same
  value. `tcp` is the default and the verified bring-up path on NPU wheels —
  SpecForge sets the engine-level `MC_FORCE_TCP` escape hatch automatically,
  so the trainer roles need no ACL device context. `ascend` requires every
  peer to hold an NPU context (plus `HCCL_INTRA_ROCE_ENABLE=1` cross-machine)
  and is only worth it after the RoCE fabric is verified.
- The producer's Mooncake segment only needs to hold one parent (~2.8 GiB)
  plus headroom — `producer_segment_size: 4294967296` (4 GiB) is the
  recommended value now that children bypass Mooncake.
- `runtime.producer_lease` must be `1`.
- `save_interval * dp_size * batch_size * accumulation_steps` and the configured
  final step must be divisible by `children_per_parent` (48).

## Fixed issues (windowed online hardening)

- **Child generation stamp**: the producer no longer copies the parent's
  Mooncake `generation` into child refs (`w0001+` were previously unreadable
  with `KeyError ... gen 1 ... not available`).
- **EOF fail-fast**: a terminally failed parent capture now fails the channel
  immediately (`prompts_failed` participates in pool-drain), so consumers see
  a failure sentinel instead of hanging with a frozen producer.
- **Terminal drain window**: `drain_pending_removals` waits past Mooncake's
  10 s read-lease TTL (3 attempts x 12 s) instead of 8 x 0.25 s, so a
  lease-blocked parent remove succeeds on the first post-expiry retry.

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
