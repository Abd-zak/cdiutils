"""
Ordered, bounded, cancellable frame export for :class:`MultiVolumeViewer`.

Frames are *produced* sequentially in the main thread (scene JSON for frame k),
*rendered* to PNG either sequentially in a worker thread (``safe``) or in a
process pool (``fast`` / ``process``), and *written* strictly in order.
"""

import asyncio
import concurrent.futures
import io
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FORMATS = ("mp4", "gif", "png")


def output_path(raw: str, fmt: str, default_stem: str) -> Path:
    """Output path from a name or a path, with the right extension.

    Accepts ``name``, ``name.mp4``, ``folder/sub/name``, ``~/name``, an
    absolute path, or a folder ending with ``/`` (then ``default_stem`` is
    used). Missing folders are created when the file is written. Dots in
    the name are kept (``run_1.5bar`` -> ``run_1.5bar.mp4``). For
    ``fmt="png"`` the result is the folder that receives the frames.
    """
    raw = (raw or default_stem).strip()
    p = Path(raw).expanduser()
    if raw.endswith(("/", "\\")) or (
        p.exists() and p.is_dir() and fmt != "png"
    ):
        p = p / default_stem
    if p.suffix.lower() in (".mp4", ".gif", ".png"):
        p = p.with_suffix("")
    return p if fmt == "png" else p.with_name(f"{p.name}.{fmt}")


def _png_to_rgb_on_white(png_bytes):
    import imageio.v2 as iio

    img = np.asarray(iio.imread(io.BytesIO(png_bytes)))
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (
            (img[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a))
            .round()
            .astype(np.uint8)
        )
    return img


class FrameSink:
    """Writes PNG frames to MP4 (imageio-ffmpeg), GIF (ffmpeg palette) or a PNG folder.

    Transparent pixels are composited on white for MP4/GIF.
    """

    def __init__(self, out_path, fmt: str, fps: int):
        self.fmt = str(fmt).lower()
        if self.fmt not in FORMATS:
            raise ValueError(
                f"Unsupported format {fmt!r}; use one of {FORMATS}"
            )
        self.out_path = Path(out_path)
        self.fps = max(1, int(fps))
        self.n = 0
        self._writer = None
        self._tmp = None
        if self.fmt == "mp4":
            try:
                import imageio.v2 as iio
                import imageio_ffmpeg  # noqa: F401  (availability check)
            except Exception as e:
                raise RuntimeError(
                    "MP4 export requires imageio-ffmpeg: pip install imageio-ffmpeg"
                ) from e
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = iio.get_writer(
                str(self.out_path),
                fps=self.fps,
                codec="libx264",
                macro_block_size=2,
            )
        elif self.fmt == "gif":
            if shutil.which("ffmpeg") is None:
                raise RuntimeError(
                    "GIF export requires ffmpeg: conda install -c conda-forge ffmpeg"
                )
            self._tmp = tempfile.TemporaryDirectory()
            self._dir = Path(self._tmp.name)
        else:
            self._dir = self.out_path
            self._dir.mkdir(parents=True, exist_ok=True)

    def add(self, png_bytes: bytes):
        if self.fmt == "mp4":
            self._writer.append_data(_png_to_rgb_on_white(png_bytes))
        elif self.fmt == "gif":
            import imageio.v2 as iio

            iio.imwrite(
                self._dir / f"frame_{self.n:06d}.png",
                _png_to_rgb_on_white(png_bytes),
            )
        else:
            (self._dir / f"frame_{self.n:06d}.png").write_bytes(png_bytes)
        self.n += 1

    def close(self, encode: bool = True):
        try:
            if self.fmt == "mp4" and self._writer is not None:
                self._writer.close()
            elif self.fmt == "gif" and encode and self.n:
                self._encode_gif()
        finally:
            if self._tmp is not None:
                self._tmp.cleanup()
                self._tmp = None

    def _encode_gif(self):
        pattern, palette = (
            str(self._dir / "frame_%06d.png"),
            str(self._dir / "palette.png"),
        )
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        base = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(self.fps),
            "-i",
            pattern,
        ]
        for cmd in (
            base + ["-vf", "palettegen", palette],
            base + ["-i", palette, "-lavfi", "paletteuse", str(self.out_path)],
        ):
            p = subprocess.run(cmd, capture_output=True)
            if p.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed ({' '.join(cmd)}):\n{p.stderr.decode('utf-8', 'ignore')}"
                )


def make_executor(mode: str, workers: int):
    """``safe`` -> None (sequential in a thread); ``fast``/``process`` -> spawn process pool."""
    if mode == "safe":
        return None
    ctx = multiprocessing.get_context(os.environ.get("MV_MP_CONTEXT", "spawn"))
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, int(workers)), mp_context=ctx
    )


async def run_ordered_export(
    n_frames,
    produce,
    render_fn,
    render_args,
    sink,
    *,
    mode="safe",
    workers=1,
    in_flight=1,
    is_cancelled=lambda: False,
    on_progress=None,
):
    """Produce, render and write ``n_frames`` frames in order.

    ``produce(k)`` returns a figure JSON dict, or ``None`` to skip frame k.
    ``render_fn(fig_json, *render_args)`` returns PNG bytes; it must be a
    module-level function when a process pool is used.
    Returns the number of frames written.
    """
    loop = asyncio.get_running_loop()
    executor = make_executor(mode, workers)
    limit = 1 if executor is None else max(1, int(in_flight))
    tasks: dict = {}
    pending: dict = {}
    nxt = 0

    async def _render(k, fig_json):
        if executor is None:
            return k, await asyncio.to_thread(
                render_fn, fig_json, *render_args
            )
        return k, await loop.run_in_executor(
            executor, render_fn, fig_json, *render_args
        )

    def _flush():
        nonlocal nxt
        while nxt in pending:
            png = pending.pop(nxt)
            if png:
                sink.add(png)
            nxt += 1
            if on_progress is not None:
                on_progress(nxt, n_frames)

    async def _collect():
        done, _ = await asyncio.wait(
            tasks.keys(), return_when=asyncio.FIRST_COMPLETED
        )
        for t in done:
            tasks.pop(t)
            k, png = t.result()  # re-raises render errors
            pending[k] = png
        _flush()

    try:
        for k in range(int(n_frames)):
            if is_cancelled():
                break
            fig_json = produce(k)
            if fig_json is None:
                pending[k] = None
                _flush()
            else:
                tasks[asyncio.ensure_future(_render(k, fig_json))] = k
            while len(tasks) >= limit:
                await _collect()
            await asyncio.sleep(0)
        while tasks and not is_cancelled():
            await _collect()
    finally:
        for t in tasks:
            t.cancel()
        if executor is not None:
            await asyncio.to_thread(
                lambda: executor.shutdown(wait=True, cancel_futures=True)
            )
    return sink.n
