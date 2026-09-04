from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from utils.datasets.agent_vln.build import (
    PATTERNS,
    _write_gzip_json,
    _write_json,
    split_sizes,
)
from utils.datasets.agent_vln.render_paths import load_routes, safe_route_id


PROMPT_VERSION = "agent_vln_rewrite_v1"
STYLE_ORDER = ("concise", "natural", "landmark_rich")
STYLE_LIMITS = {
    "concise": (6, 22),
    "natural": (12, 36),
    "landmark_rich": (20, 52),
}
WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_REGEX = re.compile(r"([^\w-]+)")
FORBIDDEN_MEDIA_REFERENCE = re.compile(
    r"\b(?:dataset|annotation|waypoint|screenshot)\b|"
    r"\b(?:shown|visible|seen|depicted)\s+in\s+(?:the\s+)?"
    r"(?:(?:first|last|current|previous|next|route)\s+)?"
    r"(?:image|frame|photo|picture)\b|"
    r"\b(?:route|path|object|door|room)\s+(?:shown|visible|seen|depicted)\s+in\s+"
    r"(?:the\s+)?(?:(?:first|last|current|previous|next|route)\s+)?"
    r"(?:image|frame|photo|picture)\b|"
    r"\b(?:image|frame|photo|picture)\s+(?:shows|depicts|indicates)\b",
    re.I,
)

SYSTEM_INSTRUCTIONS = """You curate grounded English instructions for indoor
vision-and-language navigation.

The input contains several human instructions for one ground-truth route followed by
chronologically ordered RGB samples from that route. The first image uses the agent's
true starting heading. Intermediate images face the next route segment, and the final
image faces the arrival direction. Black regions can be missing Matterport scan data;
ignore them.

The human instructions collectively define the route, action order, turn direction,
and endpoint. Use images only to clarify visually supported rooms and landmarks. Do
not invent an object, doorway, turn, room transition, or destination. If annotations
differ in wording, preserve their shared route meaning. Write commands executable by
an agent starting at the first image. Never mention images, frames, coordinates,
waypoints, datasets, or annotations.

Return exactly three semantically equivalent instructions with distinct styles:
- concise: 6-22 words, direct imperative, only essential turns and endpoint;
- natural: 12-36 words, fluent directions a person would naturally give;
- landmark_rich: 20-52 words, ordered steps with only clearly supported landmarks.

Assess whether the sampled views and human annotations are sufficiently consistent.
Use partially_grounded when some textual landmark is outside the sampled view but the
route itself remains consistent. Use conflict only for a material route contradiction.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "route_check": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["grounded", "partially_grounded", "conflict"],
                },
                "notes": {"type": "string", "minLength": 1, "maxLength": 300},
                "verified_landmarks": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 80},
                    "maxItems": 8,
                },
            },
            "required": ["status", "notes", "verified_landmarks"],
            "additionalProperties": False,
        },
        "instructions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "style": {"type": "string", "enum": list(STYLE_ORDER)},
                    "text": {"type": "string", "minLength": 1, "maxLength": 500},
                },
                "required": ["style", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["route_check", "instructions"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class RewriteJob:
    route: Mapping[str, Any]
    episode: Mapping[str, Any]
    frame_manifest: Mapping[str, Any]
    fingerprint: str

    @property
    def route_id(self) -> str:
        return str(self.route["episode_id"])

    @property
    def split(self) -> str:
        return str(self.route["split"])


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def validate_generation(value: Mapping[str, Any]) -> dict[str, Any]:
    route_check = value.get("route_check")
    if not isinstance(route_check, Mapping):
        raise ValueError("route_check must be an object")
    if route_check.get("status") not in {"grounded", "partially_grounded", "conflict"}:
        raise ValueError("route_check.status is invalid")
    notes = route_check.get("notes")
    landmarks = route_check.get("verified_landmarks")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("route_check.notes must be non-empty")
    if not isinstance(landmarks, list) or not all(
        isinstance(item, str) and item.strip() for item in landmarks
    ):
        raise ValueError("verified_landmarks must be a string list")

    instructions = value.get("instructions")
    if not isinstance(instructions, list) or len(instructions) != 3:
        raise ValueError("exactly three instructions are required")
    by_style: dict[str, dict[str, Any]] = {}
    normalized: set[str] = set()
    for item in instructions:
        if not isinstance(item, Mapping):
            raise ValueError("instruction entries must be objects")
        style = item.get("style")
        text = item.get("text")
        if style not in STYLE_ORDER or style in by_style:
            raise ValueError("instruction styles must be unique and recognized")
        if not isinstance(text, str) or not (clean := _normalize_text(text)):
            raise ValueError(f"{style} instruction must be non-empty")
        if FORBIDDEN_MEDIA_REFERENCE.search(clean):
            raise ValueError(f"{style} instruction refers to dataset media")
        minimum, maximum = STYLE_LIMITS[str(style)]
        count = _word_count(clean)
        if not minimum <= count <= maximum:
            raise ValueError(
                f"{style} instruction has {count} words; expected {minimum}-{maximum}"
            )
        key = re.sub(r"\W+", " ", clean.casefold()).strip()
        if key in normalized:
            raise ValueError("generated instructions are not distinct")
        normalized.add(key)
        by_style[str(style)] = {
            "style": str(style),
            "text": clean,
            "word_count": count,
        }
    if set(by_style) != set(STYLE_ORDER):
        raise ValueError("one instruction per required style is required")
    return {
        "route_check": {
            "status": str(route_check["status"]),
            "notes": _normalize_text(str(notes)),
            "verified_landmarks": [_normalize_text(item) for item in landmarks],
        },
        "instructions": [by_style[style] for style in STYLE_ORDER],
    }


def _job_fingerprint(
    route: Mapping[str, Any], frame_manifest: Mapping[str, Any]
) -> str:
    source = {
        "route_id": route["episode_id"],
        "instructions": [item["text"] for item in route["instruction_variants"]],
        "frames": [item["sha256"] for item in frame_manifest["frames"]],
        "prompt_version": PROMPT_VERSION,
    }
    payload = json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_jobs(
    input_root: Path, output_root: Path, *, limit: int | None = None
) -> list[RewriteJob]:
    jobs: list[RewriteJob] = []
    routes = load_routes(input_root)
    if limit is not None:
        routes = routes[:limit]
    for route, episode in routes:
        frame_path = (
            output_root
            / "images"
            / str(route["split"])
            / safe_route_id(str(route["episode_id"]))
            / "frames.json"
        )
        if not frame_path.is_file():
            raise FileNotFoundError(
                f"missing route images for {route['episode_id']}: {frame_path}"
            )
        frame_manifest = _load_json(frame_path)
        frames = frame_manifest.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"route {route['episode_id']} has no rendered frames")
        missing = [
            item["path"]
            for item in frames
            if not (output_root / item["path"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"route {route['episode_id']} is missing {len(missing)} frame files"
            )
        jobs.append(
            RewriteJob(
                route,
                episode,
                frame_manifest,
                _job_fingerprint(route, frame_manifest),
            )
        )
    return jobs


def _generation_path(output_root: Path, job: RewriteJob) -> Path:
    return (
        output_root
        / "generations"
        / job.split
        / f"{safe_route_id(job.route_id)}.json"
    )


def _cached_generation(
    path: Path, *, fingerprint: str, model: str, reasoning_effort: str
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = _load_json(path)
        if (
            record.get("fingerprint") != fingerprint
            or record.get("model") != model
            or record.get("reasoning_effort") != reasoning_effort
            or record.get("prompt_version") != PROMPT_VERSION
        ):
            return None
        validate_generation(record["result"])
        return record
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _request_input(
    job: RewriteJob,
    output_root: Path,
    *,
    image_detail: str,
    feedback: str | None,
) -> list[dict[str, Any]]:
    route = job.route
    source = {
        "route_id": job.route_id,
        "approximate_distance_m": route["geometry"]["source_geodesic_distance_m"],
        "route_pattern": route["route_pattern"],
        "human_instructions": [
            item["text"] for item in route["instruction_variants"]
        ],
    }
    text = (
        "Rewrite this single route according to the developer instructions. "
        "The RGB samples follow in chronological order.\n"
        + json.dumps(source, ensure_ascii=False, indent=2)
    )
    if feedback:
        text += f"\nThe previous output failed validation: {feedback}. Correct it."
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    frames = job.frame_manifest["frames"]
    for index, frame in enumerate(frames):
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"Ordered route sample {index + 1}/{len(frames)} "
                    f"({frame['role']})."
                ),
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": _image_data_url(output_root / frame["path"]),
                "detail": image_detail,
            }
        )
    return [{"role": "user", "content": content}]


def _usage_dict(response: Any) -> Mapping[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "output_tokens_details", None)
    return {
        name: int(getattr(usage, name, 0) or 0)
        for name in ("input_tokens", "output_tokens", "total_tokens")
    } | {"reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0)}


async def _generate_one(
    client: Any,
    job: RewriteJob,
    output_root: Path,
    *,
    model: str,
    reasoning_effort: str,
    image_detail: str,
    retries: int,
    retry_backoff_s: float,
) -> Mapping[str, Any]:
    path = _generation_path(output_root, job)
    cached = _cached_generation(
        path,
        fingerprint=job.fingerprint,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    if cached is not None:
        return cached

    feedback: str | None = None
    for attempt in range(retries + 1):
        try:
            response = await client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=_request_input(
                    job,
                    output_root,
                    image_detail=image_detail,
                    feedback=feedback,
                ),
                reasoning={"effort": reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "agent_vln_instruction_variants",
                        "strict": True,
                        "schema": OUTPUT_SCHEMA,
                    }
                },
                max_output_tokens=4096,
                store=False,
            )
            if getattr(response, "status", None) != "completed":
                raise RuntimeError(
                    f"response status={getattr(response, 'status', None)!r}; "
                    f"details={getattr(response, 'incomplete_details', None)!r}"
                )
            output_text = getattr(response, "output_text", "")
            if not output_text:
                raise ValueError("response has no output_text")
            result = validate_generation(json.loads(output_text))
            record = {
                "schema_version": 1,
                "route_id": job.route_id,
                "fingerprint": job.fingerprint,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "response_id": getattr(response, "id", None),
                "usage": _usage_dict(response),
                "result": result,
            }
            _write_json(path, record)
            return record
        except Exception as error:
            feedback = f"{type(error).__name__}: {error}"
            print(
                f"ERROR rewrite {job.route_id} attempt {attempt + 1}/{retries + 1}: "
                f"{feedback}",
                file=sys.stderr,
                flush=True,
            )
            if attempt >= retries:
                raise
            await asyncio.sleep(retry_backoff_s * (2**attempt))
    raise AssertionError("unreachable")


async def generate_all(
    jobs: Sequence[RewriteJob], output_root: Path, args: argparse.Namespace
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, str]]]:
    from openai import AsyncOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in the process environment")
    client = AsyncOpenAI(timeout=args.api_timeout, max_retries=0)
    semaphore = asyncio.Semaphore(args.concurrency)
    results: dict[str, Mapping[str, Any]] = {}
    failures: list[dict[str, str]] = []

    async def run(job: RewriteJob) -> tuple[RewriteJob, Mapping[str, Any] | Exception]:
        async with semaphore:
            try:
                value = await _generate_one(
                    client,
                    job,
                    output_root,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    image_detail=args.image_detail,
                    retries=args.retries,
                    retry_backoff_s=args.retry_backoff,
                )
                return job, value
            except Exception as error:
                return job, error

    progress = tqdm(
        total=len(jobs), desc="rewrite routes", unit="route", dynamic_ncols=True
    )
    try:
        tasks = [asyncio.create_task(run(job)) for job in jobs]
        for task in asyncio.as_completed(tasks):
            job, value = await task
            if isinstance(value, Exception):
                failure = {
                    "route_id": job.route_id,
                    "type": type(value).__name__,
                    "message": str(value),
                }
                failures.append(failure)
                progress.write(
                    f"FAILED rewrite {job.route_id}: {failure['type']}: "
                    f"{failure['message']}",
                    file=sys.stderr,
                )
            else:
                results[job.route_id] = value
            progress.update()
            progress.set_postfix(ok=len(results), failed=len(failures))
    finally:
        progress.close()
        await client.close()
    return results, failures


def _tokenize(text: str, vocabulary: Mapping[str, Any]) -> list[int]:
    word_to_index = vocabulary.get("word2idx_dict", {})
    unknown = int(vocabulary.get("UNK_INDEX", 1))
    sentence = text.lower().replace("'s", " 's").replace(",", "").replace("?", "")
    tokens = [
        token.strip()
        for token in SENTENCE_SPLIT_REGEX.split(sentence)
        if token.strip()
    ]
    return [int(word_to_index.get(token, unknown)) for token in tokens]


def _source_documents(
    input_root: Path, splits: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    documents: dict[str, Mapping[str, Any]] = {}
    for split in splits:
        path = input_root / split / f"{split}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise ValueError(f"source split {split} must be an object")
        documents[split] = value
    return documents


def select_final_jobs(
    jobs: Sequence[RewriteJob],
    generations: Mapping[str, Mapping[str, Any]],
    *,
    count: int,
) -> tuple[list[RewriteJob], Mapping[str, Any]]:
    sizes = split_sizes(count)
    selected: list[RewriteJob] = []
    selected_ids: set[str] = set()
    required: dict[str, dict[str, int]] = {}
    for split, size in sizes.items():
        pattern_counts = {
            pattern: sum(
                PATTERNS[index % len(PATTERNS)] == pattern for index in range(size)
            )
            for pattern in PATTERNS
        }
        required[split] = pattern_counts
        for pattern, target in pattern_counts.items():
            eligible = [
                job
                for job in jobs
                if job.split == split
                and job.route["route_pattern"] == pattern
                and generations[job.route_id]["result"]["route_check"]["status"]
                != "conflict"
            ]
            if len(eligible) < target:
                raise ValueError(
                    f"only {len(eligible)} non-conflict routes for split={split}, "
                    f"pattern={pattern}; need {target}"
                )
            for job in eligible[:target]:
                selected.append(job)
                selected_ids.add(job.route_id)

    conflicts = [
        job.route_id
        for job in jobs
        if generations[job.route_id]["result"]["route_check"]["status"] == "conflict"
    ]
    curation = {
        "policy": (
            "exclude generator-reviewed conflicts and preserve split/pattern quotas"
        ),
        "candidate_routes": len(jobs),
        "selected_routes": len(selected),
        "required_patterns": required,
        "excluded_conflicts": conflicts,
        "unused_non_conflict": [
            job.route_id
            for job in jobs
            if job.route_id not in selected_ids and job.route_id not in conflicts
        ],
    }
    return selected, curation


def materialize(
    jobs: Sequence[RewriteJob],
    generations: Mapping[str, Mapping[str, Any]],
    *,
    input_root: Path,
    output_root: Path,
    model: str,
    reasoning_effort: str,
    curation: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if missing := [job.route_id for job in jobs if job.route_id not in generations]:
        raise ValueError(
            f"cannot materialize: {len(missing)} routes have no generation"
        )
    splits = sorted({job.split for job in jobs})
    source_documents = _source_documents(input_root, splits)
    routes: list[dict[str, Any]] = []
    expanded: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}

    for job in jobs:
        generation = generations[job.route_id]
        result = validate_generation(generation["result"])
        route_record = dict(job.route)
        route_record["route_images"] = job.frame_manifest
        route_record["visual_validation"] = {
            "method": "generator_review",
            **result["route_check"],
        }
        route_record["generated_instructions"] = result["instructions"]
        route_record["generation"] = {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "response_id": generation.get("response_id"),
            "usage": generation.get("usage", {}),
        }
        routes.append(route_record)

        vocabulary = source_documents[job.split]["instruction_vocab"]
        for instruction in result["instructions"]:
            episode = json.loads(json.dumps(job.episode))
            style = instruction["style"]
            episode["episode_id"] = f"{job.route_id}:{style}"
            episode["instruction"] = {
                "instruction_text": instruction["text"],
                "instruction_tokens": _tokenize(instruction["text"], vocabulary),
            }
            episode.setdefault("info", {}).update(
                {
                    "agent_vln_route_id": job.route_id,
                    "instruction_style": style,
                    "instruction_generator": model,
                }
            )
            expanded[job.split].append(episode)

    for split in splits:
        _write_gzip_json(
            output_root / split / f"{split}.json.gz",
            {
                "instruction_vocab": source_documents[split]["instruction_vocab"],
                "episodes": expanded[split],
            },
        )

    status_counts: dict[str, int] = {}
    for route in routes:
        status = route["visual_validation"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    source_manifest = input_root / "manifest.json"
    manifest = {
        "schema_version": 2,
        "name": "agent_vln_r2r_local_gpt56terra_v2",
        "source": {
            "path": str(input_root),
            "manifest_sha256": _file_sha256(source_manifest),
        },
        "generation": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "styles": list(STYLE_ORDER),
        },
        "curation": dict(curation or {}),
        "routes": routes,
    }
    summary = {
        "schema_version": 2,
        "name": manifest["name"],
        "route_count": len(routes),
        "candidate_route_count": int(
            (curation or {}).get("candidate_routes", len(routes))
        ),
        "episode_count": sum(len(values) for values in expanded.values()),
        "instruction_count": len(routes) * len(STYLE_ORDER),
        "splits": {
            split: {
                "routes": sum(job.split == split for job in jobs),
                "episodes": len(expanded[split]),
                "styles": {
                    style: sum(
                        episode["info"]["instruction_style"] == style
                        for episode in expanded[split]
                    )
                    for style in STYLE_ORDER
                },
            }
            for split in splits
        },
        "visual_validation": dict(sorted(status_counts.items())),
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "summary.json", summary)
    return summary


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite AgentVLN instructions from ordered route RGB samples"
    )
    parser.add_argument("--input", type=Path, required=True, help="v1 dataset root")
    parser.add_argument("--output", type=Path, required=True, help="v2 dataset root")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument(
        "--image-detail", choices=("low", "high", "auto"), default="high"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--api-timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--final-routes",
        type=int,
        help="materialize this many non-conflict routes with balanced splits/patterns",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.concurrency < 1 or args.retries < 0 or args.retry_backoff < 0:
        raise ValueError("concurrency must be positive and retry settings non-negative")
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    selected_jobs = load_jobs(input_root, output_root, limit=args.limit)
    generated, failures = asyncio.run(generate_all(selected_jobs, output_root, args))
    if failures:
        _write_json(output_root / "failures.json", {"failures": failures})
        print(
            f"ERROR {len(failures)} route rewrites failed; "
            "rerun the same command to resume",
            file=sys.stderr,
        )
        return 1
    failure_path = output_root / "failures.json"
    if failure_path.exists():
        failure_path.unlink()

    if args.limit is not None:
        print(
            json.dumps(
                {
                    "generated": len(generated),
                    "selected": len(selected_jobs),
                    "materialized": False,
                },
                indent=2,
            )
        )
        return 0

    all_jobs = selected_jobs
    curation: Mapping[str, Any] | None = None
    if args.final_routes is not None:
        all_jobs, curation = select_final_jobs(
            all_jobs, generated, count=args.final_routes
        )
        _write_json(output_root / "curation.json", curation)
    summary = materialize(
        all_jobs,
        generated,
        input_root=input_root,
        output_root=output_root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        curation=curation,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
