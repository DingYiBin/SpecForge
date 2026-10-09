"""Fan out one long server capture into overlapping Mooncake samples."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List

import torch

from specforge.inference.capture import CaptureConfig, verify_capture_specs
from specforge.runtime.contracts import PromptTask, SampleRef, SampleRefGroup


class WindowedServerCaptureAdapter:
    """Wrap a server ref source with producer-side parent materialization.

    The server writes one long parent capture. This adapter fetches that parent,
    writes one independent Mooncake object set per training window, and returns
    one :class:`SampleRefGroup` per prompt. No child shares parent storage.
    """

    def __init__(
        self,
        parent_source,
        store,
        *,
        child_length: int,
        overlap: int,
        children_per_parent: int,
    ) -> None:
        self.parent_source = parent_source
        self.store = store
        self.child_length = int(child_length)
        self.overlap = int(overlap)
        self.children_per_parent = int(children_per_parent)
        self.stride = self.child_length - self.overlap
        self.parent_length = self.child_length + (
            self.children_per_parent - 1
        ) * self.stride
        if self.child_length < 2 or not 0 <= self.overlap < self.child_length:
            raise ValueError("invalid window geometry")
        if self.children_per_parent < 1:
            raise ValueError("children_per_parent must be positive")

        schema = getattr(parent_source, "schema", None)
        self.aux_feature = getattr(schema, "aux_feature", None)
        self.last_hidden_feature = getattr(schema, "last_hidden_feature", None)
        if not self.aux_feature or not self.last_hidden_feature:
            raise ValueError(
                "windowed capture requires aux and last-hidden server features"
            )

    def _validate_parent(
        self, task: PromptTask, parent: SampleRef, tensors: dict[str, torch.Tensor]
    ) -> None:
        expected = len(task.payload["input_ids"])
        if expected != self.parent_length:
            raise ValueError(
                f"windowed parent {task.task_id} has {expected} tokens; "
                f"expected {self.parent_length}"
            )
        for name in (self.aux_feature, self.last_hidden_feature):
            value = tensors[name]
            if value.ndim < 2 or value.shape[0] != 1 or value.shape[1] != expected:
                raise ValueError(
                    f"parent {parent.sample_id} feature {name!r} must have shape "
                    f"[1, {expected}, ...], got {tuple(value.shape)}"
                )

    def _child_ref(
        self,
        *,
        task: PromptTask,
        parent: SampleRef,
        tensors: dict[str, torch.Tensor],
        window_index: int,
        capture: CaptureConfig,
    ) -> SampleRef:
        start = window_index * self.stride
        end = start + self.child_length
        tokens = torch.tensor(
            task.payload["input_ids"][start:end], dtype=torch.long
        ).unsqueeze(0)
        loss_mask = torch.ones((1, self.child_length), dtype=torch.long)
        if window_index:
            loss_mask[:, : self.overlap] = 0
        child_tensors = {
            "input_ids": tokens,
            "loss_mask": loss_mask,
            self.aux_feature: tensors[self.aux_feature][:, start:end].contiguous(),
            self.last_hidden_feature: tensors[self.last_hidden_feature][
                :, start:end
            ].contiguous(),
        }
        child_id = f"{parent.sample_id}:w{window_index:04d}"
        metadata = {
            **dict(parent.metadata),
            "run_id": parent.run_id,
            "source_task_id": task.task_id,
            "strategy": parent.strategy,
            "target_repr": capture.target_repr,
            "vocab_map_version": capture.vocab_map_version,
            "target_model_version": parent.target_model_version,
            "num_tokens": self.child_length,
            "windowed_parent_sample_id": parent.sample_id,
            "window_index": window_index,
            "window_start": start,
            "window_end": end,
            "child_length": self.child_length,
            "overlap": self.overlap,
            "stride": self.stride,
            "children_per_parent": self.children_per_parent,
        }
        ref = self.store.put(child_tensors, sample_id=child_id, metadata=metadata)
        try:
            specs = dict(ref.feature_specs)
            target_spec = specs[self.last_hidden_feature]
            target_meta = (
                {"vocab_map_version": capture.vocab_map_version}
                if capture.vocab_map_version
                else {}
            )
            specs[self.last_hidden_feature] = replace(
                target_spec,
                target_repr=capture.target_repr,
                target_meta=target_meta,
            )
            ref = replace(
                ref,
                source_task_id=task.task_id,
                target_model_version=parent.target_model_version,
                tokenizer_version=parent.tokenizer_version,
                draft_weight_version=parent.draft_weight_version,
                feature_specs=specs,
                metadata={**ref.metadata, **metadata},
            )
            verify_capture_specs(
                ref.feature_specs,
                capture,
                sample_id=ref.sample_id,
                recorded_aux_layer_ids=capture.aux_hidden_state_layer_ids,
                aux_feature_name=self.aux_feature,
                target_feature_name=self.last_hidden_feature,
            )
            return ref
        except BaseException:
            # The put is already visible at this point.  If metadata/spec
            # validation fails before the caller can append the ref, clean it
            # here so the outer group rollback cannot miss it.
            self.store.abort(child_id, reason="window-child-validation-failed")
            raise

    def produce_refs(
        self, tasks: List[PromptTask], *, capture: CaptureConfig
    ) -> List[Any]:
        parent_results = self.parent_source.produce_refs(tasks, capture=capture)
        if len(parent_results) != len(tasks):
            raise RuntimeError(
                "parent capture returned "
                f"{len(parent_results)} records for {len(tasks)} tasks"
            )
        results: List[Any] = []
        for task, parent in zip(tasks, parent_results):
            if not isinstance(parent, SampleRef):
                results.append(parent)
                continue
            children: list[SampleRef] = []
            handle = None
            try:
                tensors, handle = self.store.get(
                    parent,
                    device="cpu",
                    names=[self.aux_feature, self.last_hidden_feature],
                )
                self._validate_parent(task, parent, tensors)
                for window_index in range(self.children_per_parent):
                    children.append(
                        self._child_ref(
                            task=task,
                            parent=parent,
                            tensors=tensors,
                            window_index=window_index,
                            capture=capture,
                        )
                    )
            except BaseException:
                for child in children:
                    self.store.abort(child.sample_id, reason="window-fanout-failed")
                self.store.abort(parent.sample_id, reason="window-fanout-failed")
                raise
            finally:
                if handle is not None:
                    self.store.release(handle, reason="window-parent-fanned-out")
            results.append(SampleRefGroup(task.task_id, tuple(children)))
        return results

    def health(self):
        health = getattr(self.parent_source, "health", None)
        parent = health() if callable(health) else {"healthy": True}
        return {**parent, "windowed_fanout": True}


__all__ = ["WindowedServerCaptureAdapter"]
