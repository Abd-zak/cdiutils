"""Animation export, still images and live rotation for :class:`MultiVolumeViewer` (mixin)."""

import asyncio
import math
import time
import warnings
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import ipywidgets as widgets
import numpy as np

from . import geometry as geo
from .export import FrameSink, output_path, run_ordered_export
from .rendering import render_png

ANIM_PARAMS = {
    "raw": ["opacity", "thr"],
    "diff": ["opacity", "thr"],
    "region": ["opacity"],
    "slice": ["pos", "opacity"],
    "plane": ["offset", "opacity"],
    "clip": ["offset", "opacity", "thr"],
    "lines": ["opacity"],
    "arrows": ["opacity"],
}
PARAM_LABELS = {
    "opacity": "opacity",
    "thr": "iso level",
    "pos": "slice position",
    "offset": "offset along normal",
}


def _fmt_seconds(sec):
    if sec < 60:
        return f"{sec:.1f} s"
    m, s = divmod(sec, 60.0)
    return (
        f"{int(m)} min {s:04.1f} s"
        if m < 60
        else f"{int(m // 60)} h {int(m % 60)} min"
    )


class AnimMixin:
    _ANIM_REGISTRY = ANIM_PARAMS

    # =========================================================================
    # UI
    # =========================================================================
    def _build_anim_ui(self):
        L = widgets.Layout
        D = {"description_width": "70px"}
        self._export_token = {"cancel": False}
        self._export_tokens = []
        self.anim_master_mode = widgets.Dropdown(
            options=[
                ("— Select —", None),
                ("Rotation", "rotation"),
                ("Layer", "layer"),
            ],
            value=None,
            description="Animate",
            style=D,
            layout=L(width="220px"),
        )
        self.anim_name = widgets.Text(
            value="animation",
            description="Save as",
            placeholder="name or path, e.g. figures/spin",
            tooltip="File name or path (relative to the notebook folder, "
            "absolute, or ~/...). Missing folders are created.",
            style=D,
            layout=L(width="300px"),
        )
        self.anim_format = widgets.Dropdown(
            options=[("MP4", "mp4"), ("GIF", "gif"), ("PNG frames", "png")],
            value="mp4",
            description="Type",
            style={"description_width": "40px"},
            layout=L(width="170px"),
        )
        self.anim_save_btn = widgets.Button(
            description="Export animation",
            button_style="primary",
            layout=L(width="160px"),
        )
        self.anim_stop_btn = widgets.Button(
            description="Stop", layout=L(width="90px")
        )
        self.anim_fps = widgets.IntSlider(
            value=30,
            min=1,
            max=60,
            description="FPS",
            continuous_update=False,
            style=D,
            layout=L(width="260px"),
        )
        self.anim_width = widgets.IntText(
            value=self.export_width,
            description="width",
            style={"description_width": "40px"},
            layout=L(width="130px"),
        )
        self.anim_height = widgets.IntText(
            value=self.export_height,
            description="height",
            style={"description_width": "45px"},
            layout=L(width="135px"),
        )
        self.anim_scale = widgets.IntSlider(
            value=2,
            min=1,
            max=4,
            description="scale",
            style={"description_width": "40px"},
            layout=L(width="190px"),
        )

        self.anim_rot_type = widgets.Dropdown(
            options=[
                ("— Select —", None),
                ("Orbit Z", "orbit_z"),
                ("Orbit X", "orbit_x"),
                ("Orbit Y", "orbit_y"),
                ("Custom axis", "axis"),
                ("Around a layer normal", "normal"),
            ],
            value=None,
            description="Mode",
            style=D,
            layout=L(width="280px"),
        )
        self.anim_axis_x, self.anim_axis_y, self.anim_axis_z = (
            widgets.FloatText(
                value=v,
                description=f"a{a}",
                style={"description_width": "25px"},
                layout=L(width="110px"),
            )
            for a, v in zip("xyz", (0.0, 0.0, 1.0))
        )
        self.anim_axis_box = widgets.HBox(
            [self.anim_axis_x, self.anim_axis_y, self.anim_axis_z]
        )
        self.anim_normal_src = widgets.Dropdown(
            options=[],
            description="Normal of",
            style=D,
            layout=L(width="300px"),
        )
        self.anim_frames = widgets.IntSlider(
            value=180,
            min=12,
            max=1440,
            step=6,
            description="Frames",
            continuous_update=False,
            style=D,
            layout=L(width="320px"),
        )
        self.anim_degrees = widgets.FloatText(
            value=360.0,
            description="Degrees",
            style=D,
            layout=L(width="160px"),
        )
        self.anim_use_current_camera = widgets.Checkbox(
            value=True, description="Start from current camera", indent=False
        )

        self.anim_layer_key = widgets.Dropdown(
            options=[("— Select —", None)],
            value=None,
            description="Layer",
            style=D,
            layout=L(width="300px"),
        )
        self.anim_layer_param = widgets.Dropdown(
            options=[("— Select —", None)],
            value=None,
            description="Param",
            style=D,
            layout=L(width="300px"),
        )
        self.anim_range = widgets.FloatRangeSlider(
            value=(0.0, 1.0),
            min=0.0,
            max=1.0,
            step=0.01,
            description="Range",
            continuous_update=False,
            readout_format=".4g",
            style=D,
            layout=L(width="520px"),
        )
        self.anim_points = widgets.IntSlider(
            value=120,
            min=2,
            max=2000,
            description="Frames",
            continuous_update=False,
            style=D,
            layout=L(width="320px"),
        )
        self.anim_fix_range = widgets.Checkbox(
            value=True,
            description="Fix colour range to current view",
            indent=False,
        )
        self.anim_auto_trim = widgets.Checkbox(
            value=False, description="Skip empty frames", indent=False
        )
        self.anim_min_valid = widgets.FloatSlider(
            value=0.01,
            min=0.0,
            max=0.5,
            step=0.01,
            description="Min valid",
            continuous_update=False,
            style=D,
            layout=L(width="260px"),
            tooltip="Slices/planes: minimum fraction of defined pixels",
        )

        self.anim_status = widgets.HTML("")
        self.anim_progress = widgets.IntProgress(
            value=0, min=0, max=1, layout=L(width="100%")
        )

        def row(*c):
            return widgets.HBox(
                list(c),
                layout=L(
                    flex_flow="row wrap", grid_gap="8px", align_items="center"
                ),
            )

        self.anim_rotation_box = widgets.VBox(
            [
                row(self.anim_rot_type, self.anim_normal_src),
                self.anim_axis_box,
                row(
                    self.anim_frames,
                    self.anim_degrees,
                    self.anim_use_current_camera,
                ),
            ]
        )
        self.anim_layer_box = widgets.VBox(
            [
                row(self.anim_layer_key, self.anim_layer_param),
                row(self.anim_range),
                row(self.anim_points, self.anim_fix_range),
                row(self.anim_auto_trim, self.anim_min_valid),
            ]
        )
        self.anim_panel = widgets.VBox(
            [
                widgets.HTML("<b>Export animation</b>"),
                row(
                    self.anim_master_mode,
                    self.anim_name,
                    self.anim_format,
                    self.anim_save_btn,
                    self.anim_stop_btn,
                ),
                row(
                    self.anim_fps,
                    self.anim_width,
                    self.anim_height,
                    self.anim_scale,
                ),
                self.anim_rotation_box,
                self.anim_layer_box,
                self.anim_status,
                self.anim_progress,
            ],
            layout=L(width="100%", padding="8px 10px", margin="10px 0 0 0"),
        )

        self.anim_save_btn.on_click(self._on_save_animation_clicked)
        self.anim_stop_btn.on_click(lambda _b: self._stop_export())
        for wd in (
            self.anim_master_mode,
            self.anim_rot_type,
            self.anim_layer_key,
            self.anim_layer_param,
        ):
            wd.observe(lambda ch: self._sync_anim_ui(ch), names="value")
        self._sync_anim_ui()

    def _anim_refresh_options(self):
        names = list(self._layers)
        self._set_options(
            self.anim_layer_key,
            [("— Select —", None)]
            + [(f"{k} [{self._layers[k]['type']}]", k) for k in names],
            fallback=None,
        )
        opts = []
        for k, s in self._layers.items():
            if s["type"] in ("plane", "clip"):
                opts.append((k, k))
            elif s["type"] == "slice":
                opts.append(
                    (
                        f"{k} (axis {self.frame.phys_axis_to_plot(s['axis_phys'])})",
                        k,
                    )
                )
        self._set_options(self.anim_normal_src, opts)

    def _sync_anim_ui(self, ch=None):
        mode = self.anim_master_mode.value
        owner = ch.get("owner") if isinstance(ch, dict) else None
        self.anim_rotation_box.layout.display = (
            "" if mode == "rotation" else "none"
        )
        self.anim_layer_box.layout.display = "" if mode == "layer" else "none"
        rot = self.anim_rot_type.value
        self.anim_axis_box.layout.display = "" if rot == "axis" else "none"
        self.anim_normal_src.layout.display = "" if rot == "normal" else "none"
        if owner is self.anim_layer_key:
            k = self.anim_layer_key.value
            params = (
                ANIM_PARAMS.get(self._layers[k]["type"], [])
                if k in self._layers
                else []
            )
            self.anim_layer_param.options = [("— Select —", None)] + [
                (PARAM_LABELS[p], p) for p in params
            ]
            self.anim_layer_param.value = None
        if owner is self.anim_layer_param and self.anim_layer_param.value:
            lo, hi = self._anim_param_bounds(
                self.anim_layer_key.value, self.anim_layer_param.value
            )
            if lo is not None:
                with self.anim_range.hold_trait_notifications():
                    if lo > self.anim_range.max:
                        self.anim_range.max, self.anim_range.min = hi, lo
                    else:
                        self.anim_range.min, self.anim_range.max = lo, hi
                    self.anim_range.step = (
                        1.0
                        if self.anim_layer_param.value == "pos"
                        else (hi - lo) / 500
                    )
                    self.anim_range.value = (lo, hi)
        ready = (mode == "rotation" and rot is not None) or (
            mode == "layer"
            and self.anim_layer_key.value is not None
            and self.anim_layer_param.value is not None
        )
        self.anim_save_btn.disabled = not ready

    def _anim_param_bounds(self, layer, param):
        """(min, max) of an animatable parameter, or (None, None)."""
        if layer not in self._layers:
            return None, None
        s = self._layers[layer]
        if param == "opacity":
            return 0.0, 1.0
        if param == "thr" and "thr" in self._layer_widgets.get(layer, {}):
            t = self._layer_widgets[layer]["thr"]
            return float(t.min), float(t.max)
        if param == "pos" and s["type"] == "slice":
            return 0.0, float(self.frame.n_along(s["axis_phys"]) - 1)
        if param == "offset" and s["type"] in ("plane", "clip"):
            return self._offset_range(s)
        return None, None

    # =========================================================================
    # Live rotation
    # =========================================================================
    def _rotation_axis_plot(self, mode, normal_layer=None):
        if mode in ("orbit_x", "orbit_y", "orbit_z"):
            return np.eye(3)["xyz".index(mode[-1])]
        if mode == "axis":
            return geo.unit(
                [
                    self.anim_axis_x.value,
                    self.anim_axis_y.value,
                    self.anim_axis_z.value,
                ]
            )
        if mode == "normal":
            s = self._layers.get(normal_layer or self.anim_normal_src.value)
            if s is not None:
                n = (
                    s.get("normal")
                    if s["type"] in ("plane", "clip")
                    else np.eye(3)["xyz".index(s["axis_phys"])]
                    if s["type"] == "slice"
                    else None
                )
                if n is not None:
                    return geo.unit(self.frame.phys_to_plot(n))
        return np.array([0.0, 0.0, 1.0])

    _get_rotation_axis_plot = _rotation_axis_plot

    def _current_camera(self):
        cam = self.fig.layout.scene.camera
        cam = cam.to_plotly_json() if cam is not None else {}
        eye = cam.get("eye") or {"x": 1.5, "y": 1.5, "z": 1.5}
        up = cam.get("up") or {"x": 0, "y": 0, "z": 1}
        return (
            np.array([eye.get(k, 1.5) for k in "xyz"], float),
            np.array([up.get(k, 0.0) for k in "xyz"], float),
            cam.get("center") or {"x": 0, "y": 0, "z": 0},
            cam.get("projection"),
        )

    @staticmethod
    def _camera_dict(eye, up, center, projection=None):
        d = dict(
            eye=dict(zip("xyz", map(float, eye))),
            up=dict(zip("xyz", map(float, up))),
            center=center,
        )
        if projection:
            d["projection"] = projection
        return d

    def _start_rotation(self):
        if self._rotation_task is not None and not self._rotation_task.done():
            return

        async def rotate():
            eye0, up0, center, proj = self._current_camera()
            ang = 0.0
            while self.rotate_toggle.value:
                ang += math.radians(2.0)
                R = geo.rodrigues(
                    self._rotation_axis_plot(
                        self.anim_rot_type.value or "orbit_z"
                    ),
                    ang,
                )
                self.fig.layout.scene.camera = self._camera_dict(
                    R @ eye0, R @ up0, center, proj
                )
                await asyncio.sleep(0.05)

        try:
            self._rotation_task = asyncio.get_running_loop().create_task(
                rotate()
            )
        except RuntimeError:
            self._rotation_task = None

    def _stop_rotation(self):
        if self._rotation_task is not None:
            self._rotation_task.cancel()
            self._rotation_task = None

    # =========================================================================
    # Still image
    # =========================================================================
    def _export_base_json(self, width, height):
        d = self.fig.to_plotly_json()
        d["data"] = [
            t for t in d["data"] if not str(t.get("name", "")).startswith("__")
        ]
        d["layout"] = dict(
            d.get("layout") or {},
            autosize=False,
            width=int(width),
            height=int(height),
        )
        return d

    def write_image(
        self,
        out_path,
        width=None,
        height=None,
        scale=1,
        rendering_backend=None,
    ):
        """Save the current view as PNG with the chosen (or the viewer's) backend."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = (
            int(width or self.export_width),
            int(height or self.export_height),
        )
        backend = str(rendering_backend or self.rendering_backend).lower()
        self._rebuild()
        png = render_png(
            self._export_base_json(width, height),
            width,
            height,
            int(scale),
            backend,
            self.fontsize,
        )
        if not png.startswith(b"\x89PNG"):
            raise RuntimeError(
                f"The {backend!r} renderer did not return PNG data."
            )
        out_path.write_bytes(png)
        return out_path

    # =========================================================================
    # Animation export
    # =========================================================================
    def _stop_export(self):
        """Stop the running export and drop the queued ones."""
        for tok in self._export_tokens:
            tok["cancel"] = True
        self._anim_cancel = True
        self.rotate_toggle.value = False
        self.anim_status.value = "Stopping…"
        self.anim_progress.bar_style = "warning"

    def _on_stop_rotation_export_clicked(self, _b=None):
        return self._stop_export()

    def _on_save_animation_clicked(self, _b=None):
        try:
            if self.anim_master_mode.value == "rotation":
                self.start_rotation_export()
            elif self.anim_master_mode.value == "layer":
                self.start_layer_export(
                    self.anim_layer_key.value,
                    self.anim_layer_param.value,
                    self.anim_range.value,
                )
        except Exception as e:
            self.anim_status.value = (
                f"<b style='color:#d62728'>{type(e).__name__}: {e}</b>"
            )

    def start_rotation_export(
        self,
        mode=None,
        axis=None,
        frames=None,
        fps=None,
        fmt=None,
        name=None,
        normal_layer=None,
        degrees=None,
        queue=False,
    ):
        """Export a camera rotation (MP4 / GIF / PNG frames).

        Returns the asyncio task (in a notebook) or the output path.
        ``queue=True`` waits for a running export instead of stopping it.
        """
        mode = mode or self.anim_rot_type.value or "orbit_z"
        if axis is not None:
            (
                self.anim_axis_x.value,
                self.anim_axis_y.value,
                self.anim_axis_z.value,
            ) = map(float, axis)
        n = int(frames or self.anim_frames.value)
        deg = float(self.anim_degrees.value if degrees is None else degrees)
        ax = self._rotation_axis_plot(mode, normal_layer)
        W, H = int(self.anim_width.value), int(self.anim_height.value)
        self.rotate_toggle.value = False
        base = self._export_base_json(W, H)
        if self.anim_use_current_camera.value:
            eye0, up0, center, proj = self._current_camera()
        else:
            eye0, up0, center, proj = (
                np.array([1.5, 1.5, 1.5]),
                np.array([0, 0, 1.0]),
                {"x": 0, "y": 0, "z": 0},
                None,
            )
        scene = dict(base["layout"].get("scene") or {})

        def produce(k):
            R = geo.rodrigues(ax, math.radians(deg) * k / max(1, n))
            lay = dict(
                base["layout"],
                scene=dict(
                    scene,
                    camera=self._camera_dict(R @ eye0, R @ up0, center, proj),
                ),
            )
            return {"data": base["data"], "layout": lay}

        return self._launch_export(
            n, produce, fmt, fps, name, "rotation", queue=queue
        )

    def start_layer_export(
        self,
        layer,
        param,
        value_range,
        points=None,
        fps=None,
        fmt=None,
        name=None,
        auto_trim=None,
        min_valid=None,
        queue=False,
    ):
        """Sweep a layer parameter (opacity, thr, pos, offset) and export it.

        Returns the asyncio task (in a notebook) or the output path.
        ``queue=True`` waits for a running export instead of stopping it.
        """
        layer = self._resolve_layer_name(layer)
        param = {"iso": "thr", "op": "opacity", "slice_pos": "pos"}.get(
            param, param
        )
        if param not in ANIM_PARAMS.get(self._layers[layer]["type"], []):
            raise ValueError(
                f"Parameter {param!r} cannot be animated for a {self._layers[layer]['type']} layer."
            )
        if layer not in self._trace_order:
            raise RuntimeError(
                f"Layer '{layer}' must be visible to be animated."
            )
        lo, hi = sorted(map(float, value_range))
        blo, bhi = self._anim_param_bounds(layer, param)
        if blo is not None:
            lo, hi = max(lo, blo), min(hi, bhi)
        if hi < lo:
            raise ValueError("Empty range after clamping to the valid bounds.")
        n = int(points or self.anim_points.value)
        trim = (
            self.anim_auto_trim.value if auto_trim is None else bool(auto_trim)
        )
        min_valid = float(
            self.anim_min_valid.value if min_valid is None else min_valid
        )
        values = np.linspace(lo, hi, max(2, n))
        W, H = int(self.anim_width.value), int(self.anim_height.value)
        base = self._export_base_json(W, H)
        idx = self._trace_order.index(layer)
        spec0, style0 = dict(self._layers[layer]), self._style(layer)
        live = base["data"][idx]
        if self.anim_fix_range.value and "cmin" in live and param != "opacity":
            style0.update(
                auto_range=False,
                range=(float(live["cmin"]), float(live["cmax"])),
            )
        ci = self._cbar_index_of(layer)

        def produce(k):
            spec, style, v = dict(spec0), dict(style0), float(values[k])
            if param == "opacity":
                style["op"] = v
            elif param == "thr":
                style["thr"] = v
            elif param == "pos":
                spec["pos"] = int(round(v))
            else:
                spec["offset"] = v
            tr, info = self._build_trace(
                layer, spec=spec, style=style, cbar_index=ci, sync=False
            )
            if trim and (
                info.get("empty") or info.get("valid", 1.0) < min_valid
            ):
                return None
            data = list(base["data"])
            data[idx] = tr.to_plotly_json()
            return {"data": data, "layout": base["layout"]}

        return self._launch_export(
            len(values), produce, fmt, fps, name, "layer_anim"
        )

    def _launch_export(self, n, produce, fmt, fps, name, stem, queue=False):
        fmt = (fmt or self.anim_format.value or "mp4").lower()
        fps = int(fps or self.anim_fps.value)
        out = output_path(name or self.anim_name.value, fmt, stem)
        prev = self._export_task
        prev = prev if prev is not None and not prev.done() else None
        if not queue:
            for tok in self._export_tokens:  # stop running/queued exports
                tok["cancel"] = True
            prev = None
        token = {"cancel": False}
        self._export_token = token
        self._export_tokens = [
            t for t in self._export_tokens if not t["cancel"]
        ] + [token]
        self._anim_cancel = False
        coro = self._export_coro(n, produce, out, fmt, fps, token)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        async def after_previous():
            if prev is not None:
                try:
                    await prev
                except BaseException:
                    pass
            if token["cancel"]:
                coro.close()
                return None
            return await coro

        task = loop.create_task(after_previous())
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        self._export_task = task
        return task

    async def _export_coro(self, n, produce, out, fmt, fps, token):
        t0 = time.perf_counter()
        W, H, S = (
            int(self.anim_width.value),
            int(self.anim_height.value),
            int(self.anim_scale.value),
        )
        (
            self.anim_progress.min,
            self.anim_progress.max,
            self.anim_progress.value,
        ) = 0, int(n), 0
        self.anim_progress.bar_style = "info"
        self.anim_status.value = "Checking the rendering backend…"
        sink = None
        try:
            backend = await asyncio.to_thread(lambda: self.rendering_backend)
            sink = FrameSink(out, fmt, fps)
            self.anim_status.value = (
                f"Rendering with {backend} ({self.rendering_mode})…"
            )

            def progress(i, total):
                self.anim_progress.value = i
                self.anim_status.value = f"Rendering {i}/{total}…"

            def cancelled():
                return token["cancel"] or (
                    self._anim_cancel and token is self._export_token
                )

            async def render(mode):
                return await run_ordered_export(
                    n,
                    produce,
                    render_png,
                    (W, H, S, backend, self.fontsize),
                    sink,
                    mode=mode,
                    workers=self.render_workers,
                    in_flight=self.render_in_flight,
                    is_cancelled=cancelled,
                    on_progress=progress,
                )

            try:
                written = await render(self.rendering_mode)
            except BrokenProcessPool:
                # Typical in a plain script without an
                # ``if __name__ == "__main__":`` guard (spawned workers
                # re-import it). Fall back to sequential rendering.
                warnings.warn(
                    "The render process pool failed; rendering "
                    "sequentially. In a script, put the code under "
                    "'if __name__ == \"__main__\":' or use "
                    "rendering_mode='safe'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                sink.close(encode=False)
                sink = FrameSink(out, fmt, fps)
                written = await render("safe")
            await asyncio.to_thread(sink.close, True)
            dt = time.perf_counter() - t0
            stopped = " (stopped)" if token["cancel"] else ""
            self.anim_status.value = (
                f"Saved{stopped}: <code>{out.resolve()}</code> | {written} frames in {_fmt_seconds(dt)} "
                f"({written / dt if dt > 0 else 0:.2f} frames/s)"
            )
            self.anim_progress.bar_style = "warning" if stopped else "success"
            return out
        except BaseException as e:
            if sink is not None:
                try:
                    sink.close(encode=False)
                except Exception:
                    pass
            self.anim_progress.bar_style = "danger"
            self.anim_status.value = f"<b style='color:#d62728'>Export failed: {type(e).__name__}: {e}</b>"
            raise

    async def wait_for_export(self):
        """Wait for the running (and queued) exports; returns the last output."""
        out = None
        while self._export_task is not None:
            task = self._export_task
            try:
                out = await task
            finally:
                if self._export_task is task and task.done():
                    self._export_task = None
        return out

    async def animate_layer_and_wait(self, **spec):
        await self.run_actions_async([{"action": "animate_layer", **spec}])
        return await self.wait_for_export()

    async def animate_rotation_and_wait(self, **spec):
        await self.run_actions_async([{"action": "animate_rotation", **spec}])
        return await self.wait_for_export()
