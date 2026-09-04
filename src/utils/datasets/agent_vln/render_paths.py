from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from PIL import Image, ImageDraw
from tqdm import tqdm


SENSOR_UUID = "route_rgb"


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def safe_route_id(route_id: str) -> str:
    return route_id.replace(":", "_").replace("/", "_")


def uniform_indices(size: int, limit: int) -> list[int]:
    if size < 1 or limit < 1:
        raise ValueError("size and limit must be positive")
    if size <= limit:
        return list(range(size))
    return [round(index * (size - 1) / (limit - 1)) for index in range(limit)]


def _direction_rotation(first: Sequence[float], last: Sequence[float]) -> Any:
    import numpy as np
    from habitat_sim.utils.common import quat_from_angle_axis

    dx = float(last[0]) - float(first[0])
    dz = float(last[2]) - float(first[2])
    if math.hypot(dx, dz) <= 0.05:
        raise ValueError("cannot orient a route frame along a zero-length segment")
    yaw = math.atan2(-dx, -dz)
    return quat_from_angle_axis(yaw, np.asarray([0.0, 1.0, 0.0]))


def frame_rotation(episode: Mapping[str, Any], path_index: int) -> Any:
    from habitat_sim.utils.common import quat_from_coeffs

    path = episode["reference_path"]
    if path_index == 0:
        return quat_from_coeffs(episode["start_rotation"])
    if path_index < len(path) - 1:
        return _direction_rotation(path[path_index], path[path_index + 1])
    return _direction_rotation(path[path_index - 1], path[path_index])


def quaternion_coefficients(rotation: Any) -> list[float]:
    return [
        float(rotation.imag[0]),
        float(rotation.imag[1]),
        float(rotation.imag[2]),
        float(rotation.real),
    ]


def create_simulator(
    scene_path: Path,
    *,
    gpu_device_id: int,
    width: int,
    height: int,
    hfov: float,
    sensor_height: float,
) -> Any:
    import habitat_sim

    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(scene_path)
    simulator_config.gpu_device_id = gpu_device_id
    simulator_config.enable_physics = False

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = SENSOR_UUID
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor.resolution = [height, width]
    sensor.position = [0.0, sensor_height, 0.0]
    sensor.hfov = hfov

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [sensor]
    return habitat_sim.Simulator(
        habitat_sim.Configuration(simulator_config, [agent_config])
    )


def _render_frame(
    simulator: Any, position: Sequence[float], rotation: Any
) -> Image.Image:
    import habitat_sim
    import numpy as np

    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = rotation
    simulator.get_agent(0).set_state(state, reset_sensors=True)
    pixels = np.asarray(simulator.get_sensor_observations()[SENSOR_UUID])
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError(f"unexpected RGB observation shape: {pixels.shape}")
    rgb = np.ascontiguousarray(pixels[:, :, :3].astype(np.uint8))
    if float(rgb.std()) < 1.0:
        raise ValueError("rendered route frame is blank or nearly constant")
    return Image.fromarray(rgb, mode="RGB")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contact_sheet(images: Sequence[Image.Image], labels: Sequence[str]) -> Image.Image:
    columns = 2
    label_height = 28
    rows = math.ceil(len(images) / columns)
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        draw.text((x + 8, y + 7), label, fill="black")
        sheet.paste(image, (x, y + label_height))
    return sheet


def render_route(
    simulator: Any,
    route: Mapping[str, Any],
    episode: Mapping[str, Any],
    output_root: Path,
    *,
    max_frames: int,
    width: int,
    height: int,
    hfov: float,
    sensor_height: float,
    overwrite: bool,
) -> Mapping[str, Any]:
    route_id = str(route["episode_id"])
    split = str(route["split"])
    directory = output_root / "images" / split / safe_route_id(route_id)
    metadata_path = directory / "frames.json"
    if metadata_path.is_file() and not overwrite:
        metadata = _load_json(metadata_path)
        frame_paths = [output_root / value["path"] for value in metadata["frames"]]
        if frame_paths and all(path.is_file() for path in frame_paths):
            return metadata

    path = episode.get("reference_path")
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError(f"route {route_id} has no usable reference_path")
    indices = uniform_indices(len(path), max_frames)
    directory.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    labels: list[str] = []
    for frame_index, path_index in enumerate(indices):
        rotation = frame_rotation(episode, path_index)
        image = _render_frame(simulator, path[path_index], rotation)
        filename = f"frame_{frame_index:02d}.jpg"
        image_path = directory / filename
        image.save(image_path, format="JPEG", quality=90, optimize=True)
        role = (
            "start"
            if path_index == 0
            else "goal"
            if path_index == len(path) - 1
            else "route"
        )
        relative_path = image_path.relative_to(output_root)
        frames.append(
            {
                "index": frame_index,
                "path_index": path_index,
                "role": role,
                "path": str(relative_path),
                "position": [float(value) for value in path[path_index]],
                "rotation": quaternion_coefficients(rotation),
                "sha256": _file_sha256(image_path),
            }
        )
        images.append(image)
        labels.append(f"{frame_index + 1}/{len(indices)}  {role}")

    sheet_path = directory / "contact_sheet.jpg"
    sheet = _contact_sheet(images, labels)
    sheet.save(sheet_path, format="JPEG", quality=90, optimize=True)
    metadata = {
        "schema_version": 1,
        "route_id": route_id,
        "scene_id": route["scene_id"],
        "render": {
            "width": width,
            "height": height,
            "hfov_degrees": hfov,
            "sensor_height_m": sensor_height,
            "orientation": (
                "original episode heading at start; outgoing route direction at "
                "intermediate points; arrival direction at goal"
            ),
        },
        "frames": frames,
        "contact_sheet": str(sheet_path.relative_to(output_root)),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def cached_route_frames(
    route: Mapping[str, Any], output_root: Path, args: argparse.Namespace
) -> Mapping[str, Any] | None:
    path = (
        output_root
        / "images"
        / str(route["split"])
        / safe_route_id(str(route["episode_id"]))
        / "frames.json"
    )
    if not path.is_file() or args.overwrite:
        return None
    try:
        metadata = _load_json(path)
        render = metadata["render"]
        if (
            metadata.get("route_id") != route["episode_id"]
            or metadata.get("scene_id") != route["scene_id"]
            or render.get("width") != args.width
            or render.get("height") != args.height
            or float(render.get("hfov_degrees")) != args.hfov
            or float(render.get("sensor_height_m")) != args.sensor_height
        ):
            return None
        frames = metadata.get("frames")
        if not isinstance(frames, list) or not frames:
            return None
        if any(not (output_root / frame["path"]).is_file() for frame in frames):
            return None
        return metadata
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def load_routes(
    dataset_root: Path,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    import gzip

    manifest = _load_json(dataset_root / "manifest.json")
    routes = manifest.get("episodes")
    if not isinstance(routes, list):
        raise ValueError("input manifest must contain an episodes list")
    episodes: dict[str, Mapping[str, Any]] = {}
    for split in sorted({str(route["split"]) for route in routes}):
        path = dataset_root / split / f"{split}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
        episodes.update(
            (str(episode["episode_id"]), episode) for episode in document["episodes"]
        )
    missing = [
        route["episode_id"]
        for route in routes
        if route["episode_id"] not in episodes
    ]
    if missing:
        raise ValueError(
            f"native split files are missing {len(missing)} manifest routes"
        )
    return [(route, episodes[str(route["episode_id"])]) for route in routes]


def render_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.input.resolve()
    output_root = args.output.resolve()
    routes = load_routes(dataset_root)
    if args.limit is not None:
        routes = routes[: args.limit]
    cached = 0
    pending = []
    for route, episode in routes:
        if cached_route_frames(route, output_root, args) is not None:
            cached += 1
        else:
            pending.append((route, episode))
    by_scene: dict[
        str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = defaultdict(list)
    for route, episode in pending:
        by_scene[str(route["scene_id"])].append((route, episode))
    scenes = [
        item
        for index, item in enumerate(sorted(by_scene.items()))
        if index % args.num_shards == args.shard_index
    ]

    rendered = 0
    shard_routes = sum(len(scene_routes) for _, scene_routes in scenes)
    progress = tqdm(
        total=shard_routes,
        desc=f"render shard {args.shard_index + 1}/{args.num_shards}",
        unit="route",
        dynamic_ncols=True,
    )
    try:
        for scene_id, scene_routes in scenes:
            scene_path = args.scenes_root.resolve() / scene_id
            if not scene_path.is_file():
                raise FileNotFoundError(f"scene does not exist: {scene_path}")
            with _quiet_native_output(args.quiet_native_logs):
                simulator = create_simulator(
                    scene_path,
                    gpu_device_id=args.gpu_device_id,
                    width=args.width,
                    height=args.height,
                    hfov=args.hfov,
                    sensor_height=args.sensor_height,
                )
            try:
                for route, episode in scene_routes:
                    try:
                        render_route(
                            simulator,
                            route,
                            episode,
                            output_root,
                            max_frames=args.max_frames,
                            width=args.width,
                            height=args.height,
                            hfov=args.hfov,
                            sensor_height=args.sensor_height,
                            overwrite=args.overwrite,
                        )
                    except Exception as error:
                        progress.write(
                            f"ERROR render {route['episode_id']}: "
                            f"{type(error).__name__}: {error}",
                            file=sys.stderr,
                        )
                        raise
                    rendered += 1
                    progress.update()
            finally:
                with _quiet_native_output(args.quiet_native_logs):
                    simulator.close()
    finally:
        progress.close()
    return {
        "routes": rendered,
        "cached": cached,
        "scenes": len(scenes),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "output": str(output_root),
    }


@contextlib.contextmanager
def _quiet_native_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    null_fd = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        os.close(null_fd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ordered RGB frames on AgentVLN routes"
    )
    parser.add_argument("--input", type=Path, required=True, help="v1 dataset root")
    parser.add_argument("--output", type=Path, required=True, help="v2 dataset root")
    parser.add_argument("--scenes-root", type=Path, default=Path("data/scene_datasets"))
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--sensor-height", type=float, default=1.25)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--show-native-logs", dest="quiet_native_logs", action="store_false"
    )
    parser.set_defaults(quiet_native_logs=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    summary = render_dataset(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
