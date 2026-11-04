"""SwanLab scalar/video logging without importing a simulator or torch.

SwanLab's documented Video API accepts GIF, not MP4. Keep the original MP4
and create a duration-preserving, bounded-size GIF preview for the dashboard.
https://docs.swanlab.cn/api/py-video.html (checked 2026-10-30)
"""
import importlib
import json
import math
import numbers
from pathlib import Path
import warnings


def is_video_record(value):
    return isinstance(value, dict) and value.get("type") == "video" and "path" in value


def scalar_metrics(values):
    return {key: float(value) for key, value in values.items()
            if isinstance(value, numbers.Real)}


def validate_rollout_videos(values, expected_count=None):
    """Fail at rollout time when configured recordings were not returned."""
    videos = {key: value for key, value in values.items() if is_video_record(value)}
    if expected_count is not None and len(videos) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} rollout videos, got {len(videos)}. "
            "Check n_train_vis/n_test_vis and that the loaded RobomimicImageRunner "
            "uses video_backend='path'. Videos are required even when reward is zero.")
    for key, value in videos.items():
        path = Path(value["path"]).expanduser()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty rollout video {key}: {path}")
    return len(videos)


def video_to_gif(source, destination, *, fps=10., max_fps=10., max_frames=200, max_side=256):
    """Preview the full clip, subsampling frames while retaining its duration."""
    from PIL import Image
    source, destination = Path(source), Path(destination)
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty rollout video: {source}")
    if source.suffix.lower() == ".gif":
        return source
    if (not math.isfinite(float(fps)) or fps <= 0 or not math.isfinite(float(max_fps))
            or max_fps <= 0 or max_frames < 2 or max_side < 1):
        raise ValueError("Video fps/size must be positive and max_frames must be >= 2")
    av = importlib.import_module("av")
    # Only timestamps are retained in this pass, not all uncompressed frames.
    times = []
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else float(fps)
        if not math.isfinite(rate) or rate <= 0:
            rate = float(fps)
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else index/rate
            times.append(timestamp)
    if not times:
        raise ValueError(f"Rollout video contains no decoded frames: {source}")
    duration = max(1/rate, times[-1]-times[0]+1/rate)
    count = min(len(times), int(max_frames), max(1, int(math.ceil(duration*max_fps))))
    indices = sorted({round(i*(len(times)-1)/max(1, count-1)) for i in range(count)})
    wanted = set(indices)
    images = []
    with av.open(str(source)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                image = frame.to_image().convert("RGB")
                image.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
                images.append(image)
    # Timing is carried by GIF durations; swanlab.Video has no fps argument.
    selected_times = [times[index]-times[0] for index in indices]
    ticks = [round(time*100) for time in selected_times] + [round(duration*100)]
    milliseconds = [max(1, b-a)*10 for a, b in zip(ticks, ticks[1:])]
    destination.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(destination, format="GIF", save_all=True, append_images=images[1:],
                   duration=milliseconds, loop=0, optimize=False, disposal=2)
    return destination


def prepare_swanlab_metrics(values, output_dir, *, swanlab_module=None, video_options=None,
                            strict_media=True):
    """Return SDK payload and JSON-safe media records; never log W&B objects."""
    sdk = swanlab_module or importlib.import_module("swanlab")
    payload, records = scalar_metrics(values), {}
    for key, value in values.items():
        if not is_video_record(value):
            continue
        source = Path(value["path"]).expanduser().resolve()
        record = {"path": str(source), "fps": float(value.get("fps", 10.))}
        try:
            preview = video_to_gif(source, Path(output_dir)/"media"/"swanlab"/(source.stem+".gif"),
                                   fps=record["fps"], **(video_options or {}))
            payload[key] = sdk.Video(str(preview), caption=key)
            record["preview"] = str(preview)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            if strict_media:
                raise RuntimeError(f"Cannot prepare SwanLab rollout video {key}: {source}") from error
            warnings.warn(f"SwanLab video {key}: {record['error']}", RuntimeWarning)
        records[key] = record
    return payload, records


class SwanLabLogger:
    """A run-local logger; disabled mode does not import or require SwanLab."""
    def __init__(self, cfg, output_dir, *, resume=False, swanlab_module=None):
        from omegaconf import OmegaConf
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        logging = OmegaConf.select(cfg, "logging")
        options = {} if logging is None else OmegaConf.to_container(logging, resolve=True)
        self.mode = options.get("mode", "disabled")
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("logging.mode must be online, offline, or disabled")
        backend = options.get("backend", "swanlab")
        if backend != "swanlab":
            raise ValueError("This workspace supports logging.backend=swanlab")
        self.options = options
        self.run = self.sdk = None
        if self.mode == "disabled":
            return
        try:
            self.sdk = swanlab_module or importlib.import_module("swanlab")
        except ImportError as error:
            raise ImportError("SwanLab logging requires `python -m pip install swanlab pillow av`; "
                              "or set logging.mode=disabled") from error
        identity_path = self.output/"swanlab_run.json"
        saved = json.loads(identity_path.read_text(encoding="utf-8")) if resume and identity_path.is_file() else {}
        project = options.get("project", "MP1_ablation")
        if saved and saved["project"] != project:
            raise ValueError("SwanLab resume project differs from swanlab_run.json")
        run_id = options.get("id") or saved.get("id")
        arguments = dict(project=project, experiment_name=options.get("name"),
                         mode=self.mode, logdir=str(self.output/"swanlog"),
                         config=OmegaConf.to_container(cfg, resolve=True),
                         tags=options.get("tags"), group=options.get("group"))
        if run_id:
            arguments.update(id=run_id, resume="allow")
        self.run = self.sdk.init(**arguments)
        actual_id = self.run.id
        identity = {"project": project, "id": actual_id,
                    "sdk_version": getattr(self.sdk, "__version__", None)}
        run_dir = getattr(self.run, "dir", None)
        if run_dir is not None:
            identity["run_dir"] = str(run_dir)
        identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")

    def log(self, report, *, step, rollout=None):
        rollout = rollout or {}
        records = {key: dict(value) for key, value in rollout.items() if is_video_record(value)}
        if self.run is not None:
            video_options = {key: self.options.get("video_"+key, default) for key, default in
                             (("max_fps", 10.), ("max_frames", 200), ("max_side", 256))}
            payload, records = prepare_swanlab_metrics(
                {**report, **rollout}, self.output, swanlab_module=self.sdk,
                video_options=video_options, strict_media=self.options.get("strict_media", True))
            self.run.log(payload, step=step)
        if records:
            with (self.output/"rollout_media.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "videos": records}, ensure_ascii=False)+"\n")

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None
