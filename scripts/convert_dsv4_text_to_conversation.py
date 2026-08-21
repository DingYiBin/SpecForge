# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert DeepSeek-V4 templated text into role/content conversations.

Parses ``<｜User｜>`` / ``<｜Assistant｜>`` / ``<｜end▁of▁sentence｜>`` out of
a request-log ``prompt`` (or already-split ``text``) and writes one JSONL row
per complete assistant prefix. Duplicate conversation prefixes are dropped by
SHA-256. Chat templates are not applied; render later with DSV4 or GLM
``apply_chat_template``.

Pass ``--no-explode-turns`` when the input was already split by
``convert_deepseek_request_logs.py``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BOS_MARKER = "<｜begin▁of▁sentence｜>"
DEFAULT_USER_MARKER = "<｜User｜>"
DEFAULT_ASSISTANT_MARKER = "<｜Assistant｜>"
DEFAULT_END_MARKER = "<｜end▁of▁sentence｜>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--bos-marker", default=DEFAULT_BOS_MARKER)
    parser.add_argument("--user-marker", default=DEFAULT_USER_MARKER)
    parser.add_argument("--assistant-marker", default=DEFAULT_ASSISTANT_MARKER)
    parser.add_argument("--end-marker", default=DEFAULT_END_MARKER)
    parser.add_argument(
        "--input-format",
        choices=("json", "python-literal"),
        default="json",
        help="Input record encoding. Use python-literal for lines such as {'prompt': '...'}.",
    )
    parser.add_argument(
        "--explode-turns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one row per complete assistant turn (default: true).",
    )
    return parser.parse_args()


def validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise ValueError(f"input file does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")
    if output_path.suffix != ".jsonl":
        raise ValueError(f"output path must end in .jsonl: {output_path}")
    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing output: {output_path}")


def validate_markers(
    bos_marker: str,
    user_marker: str,
    assistant_marker: str,
    end_marker: str,
) -> None:
    for name, value in (
        ("bos-marker", bos_marker),
        ("user-marker", user_marker),
        ("assistant-marker", assistant_marker),
        ("end-marker", end_marker),
    ):
        if not value:
            raise ValueError(f"{name} must not be empty")
    if user_marker == assistant_marker:
        raise ValueError("user and assistant markers must differ")


def parse_record(line: str, input_format: str) -> object:
    if input_format == "json":
        return json.loads(line)
    if input_format == "python-literal":
        return ast.literal_eval(line)
    raise ValueError(f"unsupported input format: {input_format}")


def _next_marker(
    text: str, start: int, markers: Sequence[str]
) -> tuple[int, str] | None:
    found: list[tuple[int, str]] = []
    for marker in markers:
        index = text.find(marker, start)
        if index >= 0:
            found.append((index, marker))
    if not found:
        return None
    found.sort(key=lambda item: (item[0], -len(item[1])))
    return found[0]


def parse_deepseek_messages(
    prompt: str,
    *,
    bos_marker: str = DEFAULT_BOS_MARKER,
    user_marker: str = DEFAULT_USER_MARKER,
    assistant_marker: str = DEFAULT_ASSISTANT_MARKER,
    end_marker: str = DEFAULT_END_MARKER,
) -> list[dict[str, str]]:
    """Parse a DeepSeek-templated prompt into OpenAI-style messages.

    Incomplete trailing assistant turns are dropped, along with a hanging
    user turn after the last complete assistant.
    """

    text = prompt
    if text.startswith(bos_marker):
        text = text[len(bos_marker) :]

    role_markers = (user_marker, assistant_marker)
    first = _next_marker(text, 0, role_markers)
    messages: list[dict[str, str]] = []
    if first is None:
        return messages

    first_index, _ = first
    system = text[:first_index].strip()
    if system:
        messages.append({"role": "system", "content": system})

    pending_user: str | None = None
    cursor = first_index
    while True:
        nxt = _next_marker(text, cursor, role_markers)
        if nxt is None:
            break
        index, marker = nxt
        content_start = index + len(marker)
        if marker == user_marker:
            following = _next_marker(text, content_start, role_markers)
            content_end = following[0] if following is not None else len(text)
            pending_user = text[content_start:content_end]
            cursor = content_start
            continue

        next_assistant = text.find(assistant_marker, content_start)
        response_end = text.find(end_marker, content_start)
        if response_end < 0 or (
            next_assistant >= 0 and next_assistant < response_end
        ):
            break
        response = text[content_start:response_end]
        if not response.strip() or pending_user is None:
            break
        messages.append({"role": "user", "content": pending_user})
        messages.append({"role": "assistant", "content": response})
        pending_user = None
        cursor = response_end + len(end_marker)
    return messages


def iter_conversation_prefixes(
    messages: Sequence[dict[str, str]],
    *,
    explode_turns: bool,
) -> Iterator[list[dict[str, str]]]:
    if explode_turns:
        for index, message in enumerate(messages):
            if message["role"] == "assistant":
                yield [dict(item) for item in messages[: index + 1]]
        return
    if messages and messages[-1]["role"] == "assistant":
        yield [dict(item) for item in messages]


def conversation_hash(messages: Sequence[dict[str, str]]) -> bytes:
    payload = json.dumps(list(messages), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    input_format: str,
    prompt_field: str,
    bos_marker: str,
    user_marker: str,
    assistant_marker: str,
    end_marker: str,
    explode_turns: bool,
) -> Counter:
    stats: Counter = Counter()
    seen_prefix_hashes: set[bytes] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open(encoding="utf-8") as input_handle,
        output_path.open("x", encoding="utf-8") as output_handle,
    ):
        for line_number, line in enumerate(input_handle, 1):
            if not line.strip():
                continue
            stats["input_rows"] += 1

            try:
                row = parse_record(line, input_format)
            except (json.JSONDecodeError, SyntaxError, ValueError) as exc:
                stats["invalid_records"] += 1
                logger.warning(
                    "Skipping line %d: invalid %s record: %s",
                    line_number,
                    input_format,
                    exc,
                )
                continue

            if not isinstance(row, dict):
                stats["invalid_rows"] += 1
                logger.warning("Skipping line %d: expected a JSON object", line_number)
                continue

            prompt = row.get(prompt_field)
            if not isinstance(prompt, str) or not prompt:
                stats["missing_prompt"] += 1
                logger.warning(
                    "Skipping line %d: %r is not a non-empty string",
                    line_number,
                    prompt_field,
                )
                continue

            messages = parse_deepseek_messages(
                prompt,
                bos_marker=bos_marker,
                user_marker=user_marker,
                assistant_marker=assistant_marker,
                end_marker=end_marker,
            )
            prefixes = list(
                iter_conversation_prefixes(messages, explode_turns=explode_turns)
            )
            if not prefixes:
                stats["no_complete_response"] += 1
                logger.warning(
                    "Skipping line %d: no complete assistant response", line_number
                )
                continue

            for prefix in prefixes:
                stats["complete_responses"] += 1
                prefix_hash = conversation_hash(prefix)
                if prefix_hash in seen_prefix_hashes:
                    stats["duplicate_responses"] += 1
                    continue
                seen_prefix_hashes.add(prefix_hash)
                payload: dict[str, Any] = {
                    "conversations": prefix,
                    "source_line_number": line_number,
                }
                output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                stats["output_rows"] += 1

    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        validate_paths(args.input_path, args.output_path)
        validate_markers(
            args.bos_marker,
            args.user_marker,
            args.assistant_marker,
            args.end_marker,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    stats = convert_file(
        args.input_path,
        args.output_path,
        input_format=args.input_format,
        prompt_field=args.prompt_field,
        bos_marker=args.bos_marker,
        user_marker=args.user_marker,
        assistant_marker=args.assistant_marker,
        end_marker=args.end_marker,
        explode_turns=args.explode_turns,
    )
    logger.info("Input rows: %d", stats["input_rows"])
    logger.info("Invalid records: %d", stats["invalid_records"])
    logger.info("Invalid rows: %d", stats["invalid_rows"])
    logger.info("Missing prompt: %d", stats["missing_prompt"])
    logger.info("No complete response: %d", stats["no_complete_response"])
    logger.info("Complete responses: %d", stats["complete_responses"])
    logger.info("Duplicate responses: %d", stats["duplicate_responses"])
    logger.info("Output rows (unique conversations): %d", stats["output_rows"])
    logger.info("Saved converted data to %s", args.output_path)


if __name__ == "__main__":
    main()
