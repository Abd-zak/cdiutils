"""
Interactive 3D multi-volume viewer for Jupyter notebooks.

:class:`MultiVolumeViewer` is an ipywidgets/Plotly interface for exploring
several scalar volumes defined on a shared voxel grid (amplitude, phase,
displacement, strain, ...), typically BCDI reconstructions.

Layers
------
``raw``     the input volumes (isosurface, optionally as a binary mask)
``diff``    ``A - B`` between two volumes, or ``A(current) - A(reference condition)``
``region``  a volume restricted to the surface shell or the core of a support
``slice``   axis-aligned slice (mean over a thickness)
``plane``   arbitrary plane (mean over a thickness)
``clip``    isosurface cut by a half-space, with a smooth planar cap
``lines``   polylines, e.g. dislocation lines
``arrows``  arrows, e.g. Burgers vectors, q vector, crystal-axis triad, scale bar

Every layer can be coloured by itself, by any volume, by its displayed
coordinates, by a constant, or (isosurfaces) by facet family.

Tools
-----
Several conditions (operando states, pressure steps, ...) with a slider;
click-to-probe with point values, distance and line profiles; facet
classification and per-facet statistics; surface-shell / core statistics;
linked orthoslices; linked cameras between viewers; mesh export
(STL/PLY/OBJ/VTK); JSON save/load of the whole view; scripted actions.

Rendering and export
--------------------
The interactive view is Plotly. Static frames (``write_image`` and MP4 / GIF /
PNG-sequence animations) use ``rendering_backend``: ``"kaleido"``, ``"vtk"``
or ``"auto"`` (Kaleido, then VTK). The backend is tested on first use only.
``rendering_mode`` sets the concurrency: ``"safe"`` (sequential, default),
``"fast"`` / ``"process"`` (process pool of ``render_workers``).

Coordinates
-----------
Arrays are indexed ``(z, y, x)`` and ``voxel_size`` is ``(vz, vy, vx)``.
Every coordinate typed in the UI or passed to ``add_*`` / ``run_actions``
(normals, origins, axes) is in *plot* order, i.e. it follows ``PLOT_ORDER``.
Physical coordinates start at the centre of voxel ``(0, 0, 0)``.
"""

import asyncio
import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from scipy.interpolate import RegularGridInterpolator

from .multiviewer import analysis as mva
from .multiviewer import geometry as geo
from .multiviewer.actions import (  # noqa: F401
    ActionError,
    ActionScript,
    ActionsMixin,
)
from .multiviewer.anim import AnimMixin
from .multiviewer.help import HelpMixin
from .multiviewer.rendering import resolve_backend
from .multiviewer.tools import ToolsMixin
from .multiviewer.ui import UIMixin

try:
    from .volume import colorcet_to_plotly
except Exception:  # standalone use

    def colorcet_to_plotly(cmap_name: str, n_colors: int = 256):
        if cmap_name not in plt.colormaps():
            raise ValueError(f"Colormap '{cmap_name}' not found.")
        cmap = plt.get_cmap(cmap_name)
        cols = [cmap(i) for i in np.linspace(0, 1, n_colors)]
        return [
            [
                i / (n_colors - 1),
                f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})",
            ]
            for i, c in enumerate(cols)
        ]


warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

SOLID_COLORS = (
    "red",
    "black",
    "white",
    "gray",
    "blue",
    "green",
    "orange",
    "purple",
    "yellow",
    "cyan",
    "magenta",
)
VOLUME_TYPES = ("raw", "diff", "region")
MESH_TYPES = VOLUME_TYPES + ("clip",)
SURFACE_TYPES = ("slice", "plane")
OVERLAY_TYPES = ("lines", "arrows")
REF_KEYS = ("source", "a", "b", "field", "support")
GEOMETRY_BY_TYPE = {
    "slice": {"pos", "axis", "thickness", "source"},
    "plane": {
        "normal",
        "origin",
        "offset",
        "thickness",
        "extent",
        "resolution",
        "source",
    },
    "clip": {"normal", "origin", "offset", "side", "smooth", "source"},
    "diff": {"a", "b", "ref"},
    "region": {"field", "support", "threshold", "mode", "depth"},
}
COORD_SEL = {"__x__": 0, "__y__": 1, "__z__": 2}


class _LRU(OrderedDict):
    def __init__(self, maxsize=24):
        super().__init__()
        self.maxsize = maxsize

    def get_or(self, key, builder):
        if key in self:
            self.move_to_end(key)
            return self[key]
        val = builder()
        self[key] = val
        while len(self) > self.maxsize:
            self.popitem(last=False)
        return val


class MultiVolumeViewer(
    UIMixin, ToolsMixin, AnimMixin, ActionsMixin, HelpMixin, widgets.Box
):
    """Interactive 3D viewer for comparing volumetric scalar fields.

    Parameters
    ----------
    dict_data : dict[str, ndarray] or list of such dicts
        Real 3D volumes on the same grid. A list defines several conditions
        (same keys and shapes) browsed with a slider.
    voxel_size : (vz, vy, vx)
    unit : axis unit label (e.g. ``"nm"``).
    PLOT_ORDER : ``"zyx"`` (default) or ``"xyz"``.
    condition_labels : labels of the conditions (default ``"0", "1", ...``).
    orientation_matrix : 3x3 ``U`` with ``n_crystal = U @ n_lab`` (lab = physical
        x, y, z of the volume). Identity when the volume axes are the crystal axes.
    facet_families : families used for facet colouring, e.g. ``"100, 110, 111"``.
    facet_tolerance : maximum angle (deg) to a family direction.
    iso_step : marching-cubes step (2 or 3 speeds up very large volumes).
    default_cmaps : optional ``{layer: cmap}``; layers whose name contains
        ``"phase"`` default to the cyclic ``"twilight"`` map.
    """

    def __init__(
        self,
        dict_data=None,
        voxel_size=(1, 1, 1),
        unit=None,
        figsize=(6, 6),
        fontsize=14,
        PLOT_ORDER: Literal["xyz", "zyx"] = "zyx",
        CBAR_LEN=0.7,
        render_workers: int | None = None,
        render_in_flight: int | None = None,
        rendering_mode: Literal["safe", "fast", "process"] = "safe",
        rendering_backend: Literal["auto", "kaleido", "vtk"] = "auto",
        export_width: int | None = None,
        export_height: int | None = None,
        condition_labels=None,
        orientation_matrix=None,
        facet_families="100, 110, 111",
        facet_tolerance: float = 10.0,
        iso_step: int = 1,
        default_cmaps: dict | None = None,
    ):
        super().__init__()
        if PLOT_ORDER not in ("xyz", "zyx"):
            raise ValueError("PLOT_ORDER must be 'xyz' or 'zyx'")
        mode = str(rendering_mode or "safe").lower().strip()
        if mode in ("vtk", "auto"):  # former API
            warnings.warn(
                "Use rendering_backend='vtk'/'auto'; rendering_mode controls concurrency only.",
                DeprecationWarning,
                stacklevel=2,
            )
            rendering_backend, mode = mode, "safe"
        if mode not in ("safe", "fast", "process"):
            raise ValueError(
                f"rendering_mode must be 'safe', 'fast' or 'process'; got {rendering_mode!r}"
            )

        self.voxel_size = np.asarray(voxel_size, dtype=float).reshape(3)
        self.unit = unit
        self.figsize = figsize
        self.fontsize = int(fontsize)
        self.PLOT_ORDER = PLOT_ORDER
        self.CBAR_LEN = float(CBAR_LEN)
        self.rendering_mode = mode
        self._requested_backend = rendering_backend
        self._backend = None
        self.rendering_backend_diagnostics = {}
        self.export_width = int(export_width or figsize[0] * 200)
        self.export_height = int(export_height or figsize[1] * 130)
        cpu = __import__("os").cpu_count() or 1
        self.render_workers = (
            int(render_workers) if render_workers is not None else min(4, cpu)
        )
        self.render_in_flight = min(
            int(render_in_flight or self.render_workers), self.render_workers
        )
        if self.render_workers < 1 or self.render_in_flight < 1:
            raise ValueError(
                "render_workers and render_in_flight must be >= 1"
            )

        self.orientation_matrix = mva.parse_matrix(orientation_matrix)
        self.facet_families = mva.parse_families(facet_families)
        self.facet_tolerance = float(facet_tolerance)
        self.iso_step = max(1, int(iso_step))
        self.default_cmaps = dict(default_cmaps or {})

        self._conditions, self._cond_labels, self._cond_index = [], [], 0
        self.dict_data = {}
        self.frame = None
        self._shape0 = None
        self._layers: dict = {}
        self._layer_widgets: dict = {}
        self._visible_cb: dict = {}
        self._layer_transform: dict = {}
        self._data_version = 0
        self._vol_cache, self._interp_cache = {}, {}
        self._geom_cache, self._surf_cache = _LRU(24), _LRU(48)
        self._trace_order, self._trace_cbar = [], {}
        self._batch, self._dirty = False, False
        self._export_task, self._anim_cancel = None, False
        self._rotation_task = None

        self._build_ui()
        if dict_data is not None:
            self.set_data(dict_data, labels=condition_labels)

    # =========================================================================
    # Rendering backend (resolved lazily)
    # =========================================================================
    @property
    def rendering_backend(self) -> str:
        if self._backend is None:
            self._backend, self.rendering_backend_diagnostics = (
                resolve_backend(self._requested_backend)
            )
        return self._backend

    @rendering_backend.setter
    def rendering_backend(self, value):
        self._requested_backend, self._backend = value, None

    # =========================================================================
    # Data and conditions
    # =========================================================================
    @staticmethod
    def get_colorscale(name):
        if name in SOLID_COLORS:
            return [[0.0, name], [1.0, name]]
        return colorcet_to_plotly(name)

    _colorscale = get_colorscale

    @staticmethod
    def _validate_dict(d, ref_keys=None, ref_shape=None):
        if not isinstance(d, dict):
            raise TypeError("dict_data must be a dict {name: np.ndarray}.")
        if not d:
            raise ValueError("dict_data is empty.")
        shape = ref_shape
        for k, v in d.items():
            if not isinstance(k, str):
                raise TypeError(f"Layer key {k!r} is not a str.")
            if not isinstance(v, np.ndarray):
                raise TypeError(f"Layer '{k}' is not a numpy array.")
            if v.ndim != 3:
                raise ValueError(
                    f"Layer '{k}' must be a 3D array, got ndim={v.ndim}."
                )
            if np.iscomplexobj(v) or v.dtype.kind not in "biuf":
                raise TypeError(
                    f"Layer '{k}' must be real numeric (pass amplitude/phase separately); got {v.dtype}."
                )
            shape = shape or v.shape
            if v.shape != shape:
                raise ValueError(
                    f"Shape mismatch: '{k}' has {v.shape}, expected {shape}."
                )
        if ref_keys is not None and list(d.keys()) != list(ref_keys):
            raise ValueError(
                "All conditions must have the same keys in the same order."
            )
        return shape

    def _validate_dict_data(self, dict_data):
        """Validate one ``{name: 3D array}`` dict (kept for backward compatibility)."""
        self._shape0 = self._validate_dict(dict_data)
        return self._shape0

    _apply_nan_policy = staticmethod(geo.apply_nan_policy)

    def set_data(self, dict_data, labels=None):
        """Register volumes (a dict, or a list of dicts for several conditions).

        Derived layers are kept when the new data has the same keys and shape.
        """
        conds = [dict_data] if isinstance(dict_data, dict) else list(dict_data)
        if not conds:
            raise ValueError("dict_data is empty.")
        shape = self._validate_dict(conds[0])
        for c in conds[1:]:
            self._validate_dict(c, ref_keys=conds[0].keys(), ref_shape=shape)
        raw_now = [k for k, s in self._layers.items() if s["type"] == "raw"]
        keep = raw_now == list(conds[0].keys()) and shape == self._shape0
        self._conditions = conds
        self._cond_labels = [str(x) for x in (labels or range(len(conds)))]
        if len(self._cond_labels) != len(conds):
            raise ValueError(
                "condition_labels must have one label per condition."
            )
        self._cond_index = 0
        self.dict_data = conds[0]
        self._shape0 = shape
        self.frame = geo.PlotFrame(self.PLOT_ORDER, self.voxel_size, shape)
        self._bump_data()
        if not keep:
            for name in list(self._layers):
                self._remove_layer_widgets(name)
            self._layers, self._layer_transform = {}, {}
            self._trace_order = []
            for i, k in enumerate(conds[0]):
                self._layers[k] = {"type": "raw"}
                self._ensure_transform(k)
                self._build_layer_widgets(k)
                self._add_visible_checkbox(k, visible=(i == 0))
        self._ui_after_data()
        self._rebuild()

    def set_condition(self, index: int):
        """Show condition ``index`` (all layers and settings are kept)."""
        index = int(index)
        if not 0 <= index < len(self._conditions):
            raise IndexError(
                f"Condition {index} out of range (0..{len(self._conditions) - 1})."
            )
        self._cond_index = index
        self.dict_data = self._conditions[index]
        self._bump_data()
        self._update_condition_ui()
        self._tools_after_data()
        self._rebuild()

    def _bump_data(self):
        self._data_version += 1
        self._vol_cache.clear()
        self._interp_cache.clear()
        self._geom_cache.clear()
        self._surf_cache.clear()

    # =========================================================================
    # Volume registry (raw + derived), signatures and caches
    # =========================================================================
    def volume_keys(self):
        return [
            k for k, s in self._layers.items() if s["type"] in VOLUME_TYPES
        ]

    def raw_keys(self):
        return [k for k, s in self._layers.items() if s["type"] == "raw"]

    def _sig(self, key):
        s = self._layers[key]
        t = s["type"]
        if t == "raw":
            return ("raw", key, self._data_version)
        if t == "diff":
            b = s.get("b")
            return (
                "diff",
                self._sig(s["a"]),
                self._sig(b) if b else ("ref", int(s.get("ref") or 0)),
                self._data_version,
            )
        if t == "region":
            return (
                "region",
                self._sig(s["field"]),
                self._sig(s["support"]),
                float(s["threshold"]),
                s["mode"],
                float(s["depth"]),
            )
        raise KeyError(f"'{key}' is not a volume layer.")

    def get_volume(self, key) -> np.ndarray:
        """Array of a raw or derived volume layer (current condition)."""
        sig = self._sig(key)
        hit = self._vol_cache.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        s = self._layers[key]
        if s["type"] == "raw":
            arr = self.dict_data[key]
        elif s["type"] == "diff":
            a = np.asarray(self.get_volume(s["a"]), dtype=float)
            if s.get("b"):
                b = np.asarray(self.get_volume(s["b"]), dtype=float)
            else:
                ref = int(
                    np.clip(
                        int(s.get("ref") or 0), 0, len(self._conditions) - 1
                    )
                )
                b = self._volume_in_condition(s["a"], ref)
            arr = a - b
        else:  # region
            mask = self._region_mask(s)
            arr = np.where(
                mask,
                np.asarray(self.get_volume(s["field"]), dtype=float),
                np.nan,
            )
        self._vol_cache[key] = (sig, arr)
        return arr

    def _volume_in_condition(self, key, index):
        if index == self._cond_index:
            return np.asarray(self.get_volume(key), dtype=float)
        cur = self._cond_index
        saved = (self.dict_data, self._data_version, dict(self._vol_cache))
        try:
            self._cond_index, self.dict_data = index, self._conditions[index]
            self._data_version = -index - 1  # temporary signature space
            self._vol_cache = {}
            return np.asarray(self.get_volume(key), dtype=float)
        finally:
            self._cond_index = cur
            self.dict_data, self._data_version, self._vol_cache = saved

    def _region_mask(self, s):
        sup = np.asarray(self.get_volume(s["support"]), dtype=float)
        support = np.nan_to_num(sup, nan=-np.inf) >= float(s["threshold"])
        shell, core, _ = mva.shell_core_masks(
            support, self.voxel_size, float(s["depth"])
        )
        return shell if s["mode"] == "shell" else core

    def _interp(self, key):
        sig = self._sig(key)
        hit = self._interp_cache.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        vol = np.asarray(self.get_volume(key))
        if vol.dtype.kind != "f":
            vol = vol.astype(np.float32)
        nz, ny, nx = vol.shape
        fn = RegularGridInterpolator(
            (
                np.arange(nz, dtype=float),
                np.arange(ny, dtype=float),
                np.arange(nx, dtype=float),
            ),
            vol,
            bounds_error=False,
            fill_value=np.nan,
        )
        self._interp_cache[key] = (sig, fn)
        return fn

    def sample(self, key, points, coords="physical"):
        """Values of volume ``key`` at points (``physical`` xyz, ``plot`` or ``index`` zyx)."""
        p = np.asarray(points, dtype=float)
        if coords == "plot":
            p = self.frame.plot_to_phys(p)
        idx = p if coords == "index" else self.frame.phys_to_idx(p)
        return np.asarray(self._interp(key)(idx), dtype=float)

    # =========================================================================
    # Coordinates and per-layer transforms
    # =========================================================================
    def _axis_label(self, name):
        return name if self.unit is None else f"{name} ({self.unit})"

    def _ensure_transform(self, name):
        self._layer_transform.setdefault(
            name, {"rot_deg": [0.0, 0.0, 0.0], "trans": [0.0, 0.0, 0.0]}
        )

    def _transform_parts(self, name):
        tr = self._layer_transform.get(name) or {
            "rot_deg": [0, 0, 0],
            "trans": [0, 0, 0],
        }
        R = geo.rotation_matrix_xyz(*tr["rot_deg"])
        pivot = self.frame.phys_to_plot(self.frame.center_phys())
        return R, pivot, np.asarray(tr["trans"], dtype=float)

    def _to_display(self, pts_plot, name):
        """Apply the layer's rigid transform (rotation about the volume centre)."""
        R, pivot, T = self._transform_parts(name)
        if np.allclose(R, np.eye(3)) and not T.any():
            return pts_plot
        return (np.asarray(pts_plot, dtype=float) - pivot) @ R.T + pivot + T

    def _from_display(self, pts_plot, name):
        R, pivot, T = self._transform_parts(name)
        return (np.asarray(pts_plot, dtype=float) - pivot - T) @ R + pivot

    def _phys_to_display(self, pts_phys, name):
        return self._to_display(self.frame.phys_to_plot(pts_phys), name)

    def _effective_origin(self, spec):
        return geo.effective_origin(
            spec["origin0"], spec["normal"], spec.get("offset", 0.0)
        )

    def _offset_range(self, spec):
        lo, hi = self.frame.bounds_phys()
        return geo.offset_bounds(spec["origin0"], spec["normal"], lo, hi)

    # =========================================================================
    # Geometry (cached)
    # =========================================================================
    def _mesh_geometry(self, name, spec, style):
        t, step, vox = spec["type"], self.iso_step, self.voxel_size
        if t == "region":
            key = ("region", self._sig(name), step)

            def build():
                return geo.isosurface(
                    np.isfinite(self.get_volume(name)).astype(np.float32),
                    0.5,
                    vox,
                    step,
                )
        elif t in ("raw", "diff"):
            iso, as_mask, pol = (
                float(style["thr"]),
                bool(style["as_mask"]),
                style["nan_policy"],
            )
            use_abs = bool(style.get("abs_iso", False))
            key = ("vol", self._sig(name), iso, as_mask, pol, use_abs, step)

            def build():
                vol = self.get_volume(name)
                if as_mask:
                    m = (
                        np.nan_to_num(
                            geo.apply_nan_policy(vol, pol), nan=-np.inf
                        )
                        >= iso
                    ).astype(np.float32)
                    return geo.isosurface(m, 0.5, vox, step)
                g = (
                    geo.apply_nan_policy(vol, pol)
                    if pol != "none"
                    else np.asarray(vol, dtype=float)
                )
                if use_abs:
                    g = np.abs(g)
                return geo.isosurface(
                    geo.fill_outside(g, geo.low_value(g, iso)), iso, vox, step
                )
        elif t == "clip":
            src, iso = spec["source"], float(style["thr"])
            origin = self._effective_origin(spec)
            n = geo.unit(spec["normal"])
            smooth = bool(spec.get("smooth", True))
            key = (
                "clip",
                self._sig(src),
                iso,
                tuple(np.round(n, 9)),
                tuple(np.round(origin, 9)),
                spec.get("side", "up"),
                smooth,
                step,
            )

            def build():
                f = geo.clip_implicit(
                    self.get_volume(src),
                    iso,
                    n,
                    origin,
                    spec.get("side", "up"),
                    vox,
                    smooth,
                )
                return None if f is None else geo.isosurface(f, 0.0, vox, step)
        else:
            raise ValueError(t)
        return self._geom_cache.get_or(key, build)

    def _surface_data(self, spec, source=None):
        """(data2d, X, Y, Z) of a slice/plane layer sampled from ``source`` (default: its own)."""
        src = source or spec["source"]
        if spec["type"] == "slice":
            key = (
                "slice",
                self._sig(src),
                spec["axis_phys"],
                int(spec["pos"]),
                int(spec["thickness"]),
            )

            def build():
                return geo.axis_slice(
                    self.get_volume(src),
                    self.frame,
                    spec["axis_phys"],
                    int(spec["pos"]),
                    int(spec["thickness"]),
                )
        else:
            origin = self._effective_origin(spec)
            n = geo.unit(spec["normal"])
            key = (
                "plane",
                self._sig(src),
                tuple(np.round(n, 9)),
                tuple(np.round(origin, 9)),
                float(spec["thickness"]),
                float(spec["extent"]),
                int(spec.get("resolution", 200)),
            )

            def build():
                return geo.sample_plane(
                    self._interp(src),
                    self.frame,
                    n,
                    origin,
                    float(spec["thickness"]),
                    float(spec["extent"]),
                    int(spec.get("resolution", 200)),
                )

        return self._surf_cache.get_or(key, build)

    # =========================================================================
    # Trace building (single code path for live view, exports and mesh files)
    # =========================================================================
    def _build_trace(
        self, name, spec=None, style=None, cbar_index=0, sync=False
    ):
        """Return ``(trace, info)``; ``spec``/``style`` overrides leave the live state untouched."""
        spec = self._layers[name] if spec is None else spec
        style = self._style(name) if style is None else style
        t = spec["type"]
        if t in MESH_TYPES:
            return self._mesh_trace(name, spec, style, cbar_index, sync)
        if t in SURFACE_TYPES:
            return self._surface_trace(name, spec, style, cbar_index, sync)
        if t == "lines":
            return self._lines_trace(name, spec, style)
        if t == "arrows":
            return self._arrows_trace(name, spec, style)
        return go.Scatter3d(name=name), {"cbar": False, "empty": True}

    def _empty_mesh(self, name, style):
        return go.Mesh3d(
            name=name,
            x=[],
            y=[],
            z=[],
            i=[],
            j=[],
            k=[],
            opacity=float(style["op"]),
            showscale=False,
        ), {"cbar": False, "empty": True}

    def _lighting(self, style):
        return {
            k: float(style[f"light_{k}"])
            for k in ("ambient", "diffuse", "specular", "roughness", "fresnel")
        }

    def _color_source(self, name, spec):
        t = spec["type"]
        return (
            name
            if t in ("raw", "diff")
            else spec["field"]
            if t == "region"
            else spec.get("source")
        )

    def _mesh_trace(self, name, spec, style, cbar_index, sync):
        geom = self._mesh_geometry(name, spec, style)
        if geom is None:
            return self._empty_mesh(name, style)
        verts_idx, faces, normals = geom
        policy, sel = style["nan_policy"], style["color_by"]
        src = self._color_source(name, spec)
        if policy == "none":  # keep true holes where the source is undefined
            self_vals = np.asarray(self._interp(src)(verts_idx), dtype=float)
            bad = ~np.isfinite(self_vals)
            if bad.any():
                faces = faces[~bad[faces].any(axis=1)]
                if not len(faces):
                    return self._empty_mesh(name, style)
        verts_phys = self.frame.idx_to_phys(verts_idx)
        verts_plot = self._phys_to_display(verts_phys, name)
        colorscale = self._colorscale(style["cmap"])
        extra = {"intensitymode": "vertex"}
        title = (
            name
            if sel == "__self__"
            else sel.strip("_")
            if sel in COORD_SEL
            else sel
        )
        if sel == "__facet__":
            fam = self._face_families(verts_phys, faces, normals)
            nf = len(self.facet_families)
            vals = np.where(fam < 0, nf, fam).astype(float) + 0.5
            colorscale, top = mva.facet_colorscale(nf)
            cmin, cmax = 0.0, float(top)
            labels = [lab for lab, _ in self.facet_families] + ["other"]
            extra = {"intensitymode": "cell"}
            cbar = (
                self._colorbar_dict("facet", cbar_index)
                if style["show_colorbar"]
                else None
            )
            if cbar:
                cbar.update(
                    tickvals=[i + 0.5 for i in range(nf + 1)], ticktext=labels
                )
        else:
            if sel == "__constant__":
                vals = np.zeros(len(verts_plot))
            elif sel in COORD_SEL:
                vals = verts_plot[:, COORD_SEL[sel]]
            elif (
                sel in self._layers
                and self._layers[sel]["type"] in VOLUME_TYPES
            ):
                vals = self._interp(sel)(verts_idx)
            else:
                vals = self._interp(src)(verts_idx)
            vals = geo.apply_nan_policy(vals, policy)
            if policy == "none":
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            if sel == "__constant__":
                cmin, cmax, cbar = 0.0, 1.0, None
            else:
                cmin, cmax = self._color_range(
                    name, style, *geo.finite_range(vals), sync
                )
                cbar = (
                    self._colorbar_dict(title, cbar_index)
                    if self._wants_cbar(style)
                    else None
                )
        tr = go.Mesh3d(
            name=name,
            x=verts_plot[:, 0],
            y=verts_plot[:, 1],
            z=verts_plot[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=vals,
            colorscale=colorscale,
            opacity=float(style["op"]),
            colorbar=cbar,
            showscale=cbar is not None,
            cmin=cmin,
            cmax=cmax,
            flatshading=False,
            lighting=self._lighting(style),
            hoverinfo="name",
            **extra,
        )
        return tr, {
            "cbar": cbar is not None,
            "empty": False,
            "n": int(len(faces)),
        }

    def _face_families(self, verts_phys, faces, vertex_normals):
        fn, _ = geo.face_normals(verts_phys, faces)
        fn = geo.orient_faces_like(fn, vertex_normals, faces)
        fam, _, _ = mva.classify_normals(
            fn,
            self.facet_families,
            self.orientation_matrix,
            self.facet_tolerance,
        )
        return fam

    def _surface_trace(self, name, spec, style, cbar_index, sync):
        base, X, Y, Z = self._surface_data(spec)
        policy, sel = style["nan_policy"], style["color_by"]
        data = geo.apply_nan_policy(base, policy)
        P = self._phys_to_display(np.stack([X, Y, Z], axis=-1), name)
        if sel == "__constant__":
            C = np.zeros_like(data)
        elif sel in COORD_SEL:
            C = P[..., COORD_SEL[sel]]
        elif (
            sel in self._layers
            and self._layers[sel]["type"] in VOLUME_TYPES
            and sel != spec["source"]
        ):
            C = geo.apply_nan_policy(
                self._surface_data(spec, source=sel)[0], policy
            )
        else:
            C = data
        C = np.array(C, dtype=float, copy=True)
        if policy == "none":
            hole = ~np.isfinite(base)
            P = P.copy()
            P[hole] = np.nan
            C[hole] = np.nan
        if sel == "__constant__":
            cmin, cmax, cbar = 0.0, 1.0, None
        else:
            cmin, cmax = self._color_range(
                name, style, *geo.finite_range(C), sync
            )
            title = (
                name
                if sel in ("__self__", "__facet__")
                else sel.strip("_")
                if sel in COORD_SEL
                else sel
            )
            cbar = (
                self._colorbar_dict(title, cbar_index)
                if self._wants_cbar(style)
                else None
            )
        empty = not np.isfinite(C).any() if policy == "none" else False
        tr = go.Surface(
            name=name,
            x=P[..., 0],
            y=P[..., 1],
            z=P[..., 2],
            surfacecolor=C,
            colorscale=self._colorscale(style["cmap"]),
            cmin=cmin,
            cmax=cmax,
            opacity=float(style["op"]),
            showscale=cbar is not None,
            colorbar=cbar,
            connectgaps=False,
            hoverinfo="name",
        )  # no custom lighting: it blanks surfaces in Kaleido
        valid = float(np.isfinite(C).mean()) if C.size else 0.0
        return tr, {"cbar": cbar is not None, "empty": empty, "valid": valid}

    def _lines_trace(self, name, spec, style):
        segs = [
            np.asarray(s, dtype=float)
            for s in spec.get("lines", [])
            if len(s) > 1
        ]
        if segs:
            pts = np.concatenate(
                [np.vstack([s, np.full((1, 3), np.nan)]) for s in segs]
            )
            P = self._phys_to_display(pts, name)
        else:
            P = np.zeros((0, 3))
        tr = go.Scatter3d(
            name=name,
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            mode="lines",
            opacity=float(style["op"]),
            line=dict(color=style["color"], width=float(style["width"])),
            showlegend=False,
            hoverinfo="name",
        )
        return tr, {"cbar": False, "empty": not segs}

    def _arrows_trace(self, name, spec, style):
        o = np.asarray(spec["origins"], dtype=float).reshape(-1, 3)
        v = np.asarray(spec["vectors"], dtype=float).reshape(-1, 3)
        radius = (
            float(spec.get("radius") or 0.01 * self.frame.diagonal())
            * float(style.get("width", 4))
            / 4.0
        )
        V, F = geo.arrow_mesh(o, v, radius)
        P = self._phys_to_display(V, name) if len(V) else V
        tr = go.Mesh3d(
            name=name,
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            i=F[:, 0],
            j=F[:, 1],
            k=F[:, 2],
            color=style["color"],
            opacity=float(style["op"]),
            flatshading=False,
            lighting=self._lighting(style),
            hoverinfo="name",
        )
        return tr, {"cbar": False, "empty": not len(F)}

    def _scene_annotations(self, names):
        anns = []
        for n in names:
            s = self._layers[n]
            for pos, text in s.get("labels") or []:
                p = self._phys_to_display(np.asarray(pos, dtype=float), n)
                anns.append(
                    dict(
                        x=float(p[0]),
                        y=float(p[1]),
                        z=float(p[2]),
                        text=f"<b>{text}</b>",
                        showarrow=False,
                        font=dict(
                            size=int(self.fontsize * 1.3),
                            color=self._style(n).get("color", "black"),
                        ),
                    )
                )
        return anns

    # --- colour ranges and colorbars ----------------------------------------
    def _color_range(self, name, style, dmin, dmax, sync):
        w = self._layer_widgets.get(name)
        if sync and w is not None and "range_slider" in w:
            self._sync_range_widget(w, dmin, dmax, bool(style["auto_range"]))
            if not style["auto_range"]:
                lo, hi = w["range_slider"].value
                return float(lo), float(hi) if hi > lo else float(lo) + 1e-12
        if style["auto_range"] or style.get("range") is None:
            return dmin, dmax
        lo, hi = map(float, style["range"])
        return lo, hi if hi > lo else lo + 1e-12

    @staticmethod
    def _wants_cbar(style):
        return (
            bool(style["show_colorbar"]) and style["cmap"] not in SOLID_COLORS
        )  # a solid colour needs no bar

    def _colorbar_dict(self, title, cbar_index=0):
        dark = self._dark()
        txt = "#e6eefc" if dark else "#000000"
        return dict(
            title=dict(
                text=title,
                font=self._bold_font(self.fontsize * 1.5),
                side="top",
            ),
            tickfont=self._bold_font(self.fontsize * 1.5) | dict(color=txt),
            thickness=28,
            outlinewidth=2,
            outlinecolor=txt,
            len=self.CBAR_LEN,
            lenmode="fraction",
            x=1.02 + 0.2 * cbar_index,
            xanchor="left",
            y=0.5,
            yanchor="middle",
        )

    def _bold_font(self, size=None):
        return dict(family="DejaVu Sans Bold", size=int(size or self.fontsize))

    # =========================================================================
    # Refresh
    # =========================================================================
    def _visible_names(self):
        return [
            k
            for k in self._layers
            if k in self._visible_cb and self._visible_cb[k].value
        ]

    def _cbar_index_of(self, name):
        i = 0
        for n in self._trace_order:
            if n == name:
                return i
            i += int(self._trace_cbar.get(n, False))
        return i

    def _refresh(self, names=None):
        """Update the given layers in place (or rebuild everything when the layout changes)."""
        if self._batch:
            self._dirty = True
            return
        vis = self._visible_names()
        if (
            names is None
            or vis != self._trace_order
            or len(self.fig.data) < len(vis)
        ):
            return self._rebuild()
        for n in names:
            if n not in vis:
                continue
            tr, info = self._build_trace(
                n, cbar_index=self._cbar_index_of(n), sync=True
            )
            if info["cbar"] != self._trace_cbar.get(n) or self._layers[n].get(
                "labels"
            ):
                return self._rebuild()
            d = tr.to_plotly_json()
            d.pop("type", None)
            with self.fig.batch_update():
                self.fig.data[vis.index(n)].update(d)

    def _rebuild(self):
        if self._batch:
            self._dirty = True
            return
        if self.frame is None:
            return
        vis = self._visible_names()
        traces, cbar, ci = [], {}, 0
        for n in vis:
            tr, info = self._build_trace(n, cbar_index=ci, sync=True)
            traces.append(tr)
            cbar[n] = info["cbar"]
            ci += int(info["cbar"])
        traces += self._extra_traces()
        with self.fig.batch_update():
            self.fig.data = ()
            if traces:
                self.fig.add_traces(traces)
            self.fig.layout.scene.annotations = self._scene_annotations(vis)
        self._trace_order, self._trace_cbar = vis, cbar
        for t in self.fig.data:
            t.on_click(self._on_trace_click)

    # backward-compatible names
    def _update_all_traces(self):
        self._rebuild()

    def _make_trace_for_layer(self, layer_name, cbar_index=0):
        tr, info = self._build_trace(
            layer_name, cbar_index=cbar_index, sync=True
        )
        return tr, info["cbar"]

    _make_mesh_trace_for_key = _make_trace_for_layer

    # =========================================================================
    # Public layer API
    # =========================================================================
    def _unique_name(self, name):
        name = str(name).strip() or "layer"
        out, k = name, 2
        while out in self._layers:
            out, k = f"{name}#{k}", k + 1
        return out

    def _check_volume(self, key):
        if (
            key not in self._layers
            or self._layers[key]["type"] not in VOLUME_TYPES
        ):
            raise KeyError(
                f"'{key}' is not a volume layer. Available: {', '.join(self.volume_keys())}"
            )
        return key

    def _add_layer(self, name, spec, visible=True, select=True, style=None):
        name = self._unique_name(name)
        self._layers[name] = spec
        self._ensure_transform(name)
        self._build_layer_widgets(name)
        if style:
            self._set_style(name, **style)
        self._add_visible_checkbox(name, visible)
        self._after_layers_changed(select=name if select else None)
        self._rebuild()
        return name

    def _plot_vec_to_phys(self, v):
        return self.frame.plot_to_phys(np.asarray(v, dtype=float).reshape(3))

    def add_slice(
        self, name, source=None, axis="z", pos=None, thickness=0, visible=True
    ):
        """Axis-aligned slice; ``axis`` is a plot axis, ``pos`` a voxel index (default: centre)."""
        src = self._check_volume(source or self.raw_keys()[0])
        ax = self.frame.plot_axis_to_phys(axis)
        n = self.frame.n_along(ax)
        pos = (n - 1) // 2 if pos is None else int(np.clip(pos, 0, n - 1))
        spec = {
            "type": "slice",
            "source": src,
            "axis_phys": ax,
            "pos": pos,
            "thickness": int(thickness),
        }
        return self._add_layer(name, spec, visible)

    def add_plane(
        self,
        name,
        source=None,
        normal=(0, 0, 1),
        origin=None,
        offset=0.0,
        thickness=0.0,
        extent=None,
        resolution=200,
        visible=True,
    ):
        """Arbitrary plane; ``normal`` and ``origin`` in plot coordinates (origin default: centre)."""
        src = self._check_volume(source or self.raw_keys()[0])
        o0 = (
            self.frame.center_phys()
            if origin is None
            else self._plot_vec_to_phys(origin)
        )
        spec = {
            "type": "plane",
            "source": src,
            "normal": self._plot_vec_to_phys(normal),
            "origin0": o0,
            "offset": 0.0,
            "thickness": float(thickness),
            "extent": float(extent or 0.5 * self.frame.diagonal()),
            "resolution": int(resolution),
        }
        spec["offset"] = float(np.clip(offset, *self._offset_range(spec)))
        return self._add_layer(name, spec, visible)

    def add_clip(
        self,
        name,
        source=None,
        normal=(0, 0, 1),
        origin=None,
        offset=0.0,
        side="up",
        smooth=True,
        visible=True,
    ):
        """Isosurface of ``source`` cut by a half-space (``side`` 'up' keeps +n)."""
        src = self._check_volume(source or self.raw_keys()[0])
        o0 = (
            self.frame.center_phys()
            if origin is None
            else self._plot_vec_to_phys(origin)
        )
        spec = {
            "type": "clip",
            "source": src,
            "normal": self._plot_vec_to_phys(normal),
            "origin0": o0,
            "offset": 0.0,
            "side": side,
            "smooth": bool(smooth),
        }
        spec["offset"] = float(np.clip(offset, *self._offset_range(spec)))
        style = (
            {"thr": self._style(src)["thr"]}
            if src in self._layer_widgets and "thr" in self._layer_widgets[src]
            else None
        )
        return self._add_layer(name, spec, visible, style=style)

    def add_difference(self, name, a, b=None, ref=0, visible=True):
        """``a - b``, or ``a(current condition) - a(condition ref)`` when ``b`` is None."""
        spec = {
            "type": "diff",
            "a": self._check_volume(a),
            "b": self._check_volume(b) if b else None,
            "ref": int(ref),
        }
        return self._add_layer(name, spec, visible, style={"cmap": "RdBu_r"})

    def add_region(
        self,
        name,
        field,
        support,
        threshold=None,
        mode="shell",
        depth=None,
        visible=True,
    ):
        """``field`` restricted to the surface shell (distance to the support boundary
        <= ``depth``) or to the core of ``support >= threshold``."""
        self._check_volume(field)
        self._check_volume(support)
        if threshold is None:
            threshold = 0.3 * float(np.nanmax(self.get_volume(support)))
        spec = {
            "type": "region",
            "field": field,
            "support": support,
            "threshold": float(threshold),
            "mode": mode,
            "depth": float(
                depth
                if depth is not None
                else 2 * float(np.max(self.voxel_size))
            ),
        }
        return self._add_layer(name, spec, visible)

    def _points_to_phys(self, pts, coords):
        p = np.asarray(pts, dtype=float).reshape(-1, 3)
        if coords == "index":
            return self.frame.idx_to_phys(p)
        if coords == "plot":
            return self.frame.plot_to_phys(p)
        return p

    def _vectors_to_phys(self, v, coords):
        v = np.asarray(v, dtype=float).reshape(-1, 3)
        if coords == "index":
            return v[:, ::-1] * self.frame.voxel_xyz
        if coords == "plot":
            return self.frame.plot_to_phys(v)
        return v

    def add_lines(
        self,
        name,
        lines,
        coords="index",
        color="black",
        width=6,
        labels=None,
        visible=True,
    ):
        """Polylines (e.g. dislocation lines from cdi_dislo).

        ``lines``: an (N, 3) array or a list of them. ``coords``: ``"index"``
        (z, y, x voxel indices, default), ``"physical"`` (x, y, z) or ``"plot"``.
        ``labels``: optional list of ``(point, text)`` in the same coordinates.
        """
        if isinstance(lines, np.ndarray) and lines.ndim == 2:
            lines = [lines]
        segs = [self._points_to_phys(s, coords) for s in lines]
        labs = [
            (self._points_to_phys(p, coords)[0], str(t))
            for p, t in (labels or [])
        ]
        spec = {"type": "lines", "lines": segs, "labels": labs}
        return self._add_layer(
            name, spec, visible, style={"color": color, "width": width}
        )

    def add_fitted_line(
        self,
        name,
        centroid,
        direction,
        length=None,
        coords="index",
        color="black",
        width=6,
        visible=True,
    ):
        """Draw a straight line given by a centroid and a direction, e.g. the output of
        ``cdiutils.analysis.dislocation.fit_line_3d`` on ``np.argwhere`` points
        (``coords="index"``, i.e. (z, y, x) voxel indices).

        Without ``length`` the line spans the whole volume box.
        """
        c = self._points_to_phys(centroid, coords)[0]
        d = geo.unit(self._vectors_to_phys(direction, coords)[0])
        if length is None:
            a, b = geo.offset_bounds(c, d, *self.frame.bounds_phys())
        else:
            a, b = -0.5 * float(length), 0.5 * float(length)
        return self.add_lines(
            name,
            [np.stack([c + a * d, c + b * d])],
            coords="physical",
            color=color,
            width=width,
            visible=visible,
        )

    def add_arrows(
        self,
        name,
        origins,
        vectors,
        coords="physical",
        color="crimson",
        radius=None,
        labels=None,
        visible=True,
    ):
        """Arrows (Burgers vectors, q vector, ...). ``labels``: one text per arrow (at its tip)."""
        o = self._points_to_phys(origins, coords)
        v = self._vectors_to_phys(vectors, coords)
        labs = (
            [(oo + 1.08 * vv, str(t)) for oo, vv, t in zip(o, v, labels)]
            if labels
            else []
        )
        spec = {
            "type": "arrows",
            "origins": o,
            "vectors": v,
            "radius": radius,
            "labels": labs,
            "texts": [str(t) for t in labels] if labels else [],
        }
        return self._add_layer(name, spec, visible, style={"color": color})

    def add_axes_triad(
        self,
        name="crystal axes",
        matrix=None,
        labels=("[100]", "[010]", "[001]"),
        origin=None,
        length=None,
        color="black",
        visible=True,
    ):
        """Crystal axes drawn in the lab frame from ``n_crystal = U @ n_lab``.

        ``matrix`` defaults to the viewer's orientation matrix; ``origin``
        (plot coordinates) defaults to a corner of the volume.
        """
        U = (
            self.orientation_matrix
            if matrix is None
            else mva.parse_matrix(matrix)
        )
        dirs = np.array([geo.unit(U[i]) for i in range(3)])
        L = float(length or 0.18 * self.frame.diagonal())
        lo, hi = self.frame.bounds_phys()
        o = (
            lo + 0.12 * (hi - lo)
            if origin is None
            else self._plot_vec_to_phys(origin)
        )
        return self.add_arrows(
            name,
            np.repeat(o[None], 3, axis=0),
            dirs * L,
            coords="physical",
            color=color,
            radius=0.012 * L,
            labels=list(labels),
            visible=visible,
        )

    def add_scale_bar(
        self,
        length,
        name="scale bar",
        axis="x",
        origin=None,
        color="black",
        visible=True,
    ):
        """Scale bar of ``length`` (physical units) along a plot axis, labelled with the unit."""
        lo, hi = self.frame.bounds_phys()
        d = self._plot_vec_to_phys(
            {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
        )
        o = (
            lo + np.array([0.1, 0.1, 0.0]) * (hi - lo)
            if origin is None
            else self._plot_vec_to_phys(origin)
        )
        p1 = o + d * float(length)
        lab = f"{length:g} {self.unit or ''}".strip()
        spec = {
            "type": "lines",
            "lines": [np.stack([o, p1])],
            "labels": [
                (
                    0.5 * (o + p1)
                    + 0.04
                    * self.frame.diagonal()
                    * self._plot_vec_to_phys((0, 0, 1)),
                    lab,
                )
            ],
        }
        return self._add_layer(
            name, spec, visible, style={"color": color, "width": 10}
        )

    # =========================================================================
    # Rename / delete
    # =========================================================================
    def _dependents(self, name):
        return [
            k
            for k, s in self._layers.items()
            if k != name and any(s.get(r) == name for r in REF_KEYS)
        ]

    def rename_layer(self, old, new):
        new = str(new).strip()
        if (
            old not in self._layers
            or not new
            or new == old
            or new in self._layers
        ):
            return
        self._layers = {
            (new if k == old else k): v for k, v in self._layers.items()
        }
        for s in self._layers.values():
            for r in REF_KEYS:
                if s.get(r) == old:
                    s[r] = new
        if self._layers[new]["type"] == "raw":
            for c in self._conditions:
                items = list(c.items())
                c.clear()
                c.update((new if k == old else k, v) for k, v in items)
        for d in (self._layer_transform, self._vol_cache, self._interp_cache):
            if old in d:
                d[new] = d.pop(old)
        self._vol_cache.clear()
        self._interp_cache.clear()
        self._trace_order = [new if k == old else k for k in self._trace_order]
        self._trace_cbar = {
            (new if k == old else k): v for k, v in self._trace_cbar.items()
        }
        self._rename_layer_widgets(old, new)
        self._after_layers_changed(select=new)
        self._rebuild()

    def delete_layer(self, name, cascade=True):
        if name not in self._layers:
            return
        if cascade:
            for dk in self._dependents(name):
                self.delete_layer(dk, cascade=True)
        spec = self._layers.pop(name)
        self._layer_transform.pop(name, None)
        self._vol_cache.pop(name, None)
        self._interp_cache.pop(name, None)
        if spec["type"] == "raw":
            for c in self._conditions:
                c.pop(name, None)
        self._remove_layer_widgets(name)
        self._after_layers_changed(select=None)
        self._rebuild()

    _rename_layer = rename_layer
    _delete_layer = delete_layer

    # =========================================================================
    # Mesh export
    # =========================================================================
    def layer_mesh(self, name, frame="plot"):
        """(verts, faces, scalars) of a visible-style rendering of ``name``.

        ``frame="plot"``: as displayed (plot order, layer transform applied);
        ``"physical"``: physical (x, y, z) without the layer transform.
        """
        style = self._style(name)
        if frame == "physical":
            saved = self._layer_transform.get(name)
            self._layer_transform[name] = {
                "rot_deg": [0, 0, 0],
                "trans": [0, 0, 0],
            }
        try:
            tr, _ = self._build_trace(name, style=style)
        finally:
            if frame == "physical":
                self._layer_transform[name] = saved
        d = tr.to_plotly_json()
        if d["type"] == "mesh3d":
            V = np.column_stack(
                [np.asarray(d.get(k, []), dtype=float) for k in "xyz"]
            )
            F = np.column_stack(
                [np.asarray(d.get(k, []), dtype=np.int64) for k in "ijk"]
            )
            S = d.get("intensity")
        elif d["type"] == "surface":
            V, F, S = geo.grid_to_mesh(
                d["x"], d["y"], d["z"], d.get("surfacecolor")
            )
        else:
            raise ValueError(
                f"Layer '{name}' ({self._layers[name]['type']}) has no surface to export."
            )
        if frame == "physical" and len(V):
            V = self.frame.plot_to_phys(V)
        return V, F, None if S is None else np.asarray(S, dtype=float)

    def export_mesh(self, name, path, frame="plot"):
        """Write layer ``name`` as .stl / .ply / .obj / .vtk (scalars in PLY and VTK)."""
        V, F, S = self.layer_mesh(name, frame=frame)
        if not len(F):
            raise ValueError(f"Layer '{name}' is empty; nothing to export.")
        out = mva.write_mesh(
            path,
            V,
            F,
            S,
            scalar_name=str(self._style(name).get("color_by", "scalar")).strip(
                "_"
            )
            or "scalar",
        )
        return out

    # =========================================================================
    # Save / load the whole view
    # =========================================================================
    @staticmethod
    def _jsonable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [MultiVolumeViewer._jsonable(x) for x in v]
        if isinstance(v, dict):
            return {k: MultiVolumeViewer._jsonable(x) for k, x in v.items()}
        return v

    def get_view_state(self) -> dict:
        """JSON-serialisable description of layers, styles, transforms, camera and tools."""
        layers = []
        for name, spec in self._layers.items():
            layers.append(
                {
                    "name": name,
                    "spec": self._jsonable(spec),
                    "style": self._jsonable(self._style(name)),
                    "transform": self._jsonable(
                        self._layer_transform.get(name)
                    ),
                    "visible": bool(self._visible_cb[name].value),
                }
            )
        cam = self.fig.layout.scene.camera
        return {
            "version": 2,
            "plot_order": self.PLOT_ORDER,
            "voxel_size": self.voxel_size.tolist(),
            "unit": self.unit,
            "condition": self._cond_index,
            "layers": layers,
            "camera": cam.to_plotly_json() if cam is not None else None,
            "dark": self._dark(),
            "grid": bool(self.grid_toggle.value),
            "orientation_matrix": self.orientation_matrix.tolist(),
            "facet_families": ", ".join(
                lab.strip("{}") for lab, _ in self.facet_families
            ),
            "facet_tolerance": self.facet_tolerance,
        }

    def set_view_state(self, state: dict):
        """Restore a state from :meth:`get_view_state` (raw volumes must be loaded)."""
        missing = [
            L["name"]
            for L in state["layers"]
            if L["spec"]["type"] == "raw" and L["name"] not in self._layers
        ]
        if missing:
            raise KeyError(f"Raw volumes missing for this state: {missing}")
        self._batch = True
        try:
            for name in [
                k for k, s in self._layers.items() if s["type"] != "raw"
            ]:
                if name in self._layers:
                    self.delete_layer(name, cascade=False)
            self.orientation_matrix = mva.parse_matrix(
                state.get("orientation_matrix")
            )
            self.facet_families = mva.parse_families(
                state.get("facet_families", "100, 110, 111")
            )
            self.facet_tolerance = float(
                state.get("facet_tolerance", self.facet_tolerance)
            )
            for L in state["layers"]:
                name, spec = L["name"], dict(L["spec"])
                for k in ("normal", "origin0"):
                    if k in spec:
                        spec[k] = np.asarray(spec[k], dtype=float)
                if spec["type"] == "lines":
                    spec["lines"] = [
                        np.asarray(s, dtype=float)
                        for s in spec.get("lines", [])
                    ]
                if spec["type"] == "arrows":
                    spec["origins"], spec["vectors"] = (
                        np.asarray(spec[k], dtype=float)
                        for k in ("origins", "vectors")
                    )
                if "labels" in spec:
                    spec["labels"] = [
                        (np.asarray(p, dtype=float), t)
                        for p, t in spec["labels"]
                    ]
                if spec["type"] != "raw":
                    self._layers[name] = spec
                    self._ensure_transform(name)
                    self._build_layer_widgets(name)
                    self._add_visible_checkbox(name, False)
                self._set_style(name, **L.get("style", {}))
                if L.get("transform"):
                    self._layer_transform[name] = L["transform"]
                    self._load_transform_widgets(name)
                self._visible_cb[name].value = bool(L.get("visible", False))
            order = [L["name"] for L in state["layers"]]
            self._layers = {
                k: self._layers[k]
                for k in order + [k for k in self._layers if k not in order]
            }
            self.theme_toggle.value = bool(state.get("dark", False))
            self.grid_toggle.value = bool(state.get("grid", True))
            if len(self._conditions) > 1:
                self.set_condition(int(state.get("condition", 0)))
            self._after_layers_changed(select=None)
        finally:
            self._batch = False
        if state.get("camera"):
            self.fig.layout.scene.camera = state["camera"]
        self._rebuild()

    def save_state(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.get_view_state(), indent=1))
        return path

    def load_state(self, path):
        self.set_view_state(json.loads(Path(path).read_text()))

    # =========================================================================
    # Scripted actions
    # =========================================================================
    _STYLE_ALIASES = {
        "opacity": "op",
        "range_slider": "range",
        "range_min_override": "range_min",
        "range_max_override": "range_max",
        "iso": "thr",
    }
    _PARAM_KEYS = {
        "pos",
        "slice_pos",
        "axis",
        "thickness",
        "normal",
        "origin",
        "offset",
        "side",
        "smooth",
        "extent",
        "resolution",
        "source",
        "a",
        "b",
        "ref",
        "field",
        "support",
        "threshold",
        "mode",
        "depth",
    }
    _TRANSFORM_KEYS = {
        "rot_x",
        "rot_y",
        "rot_z",
        "trans_x",
        "trans_y",
        "trans_z",
    }

    def _resolve_layer_name(self, layer):
        if layer in self._layers:
            return layer
        alias = {
            "amp": "amplitude",
            "disp": "displacement",
            "strain": "het_strain",
        }.get(layer, layer)
        if alias in self._layers:
            return alias
        raise KeyError(
            f"Unknown layer '{layer}'. Available layers: {', '.join(self._layers)}"
        )

    def set_layer_params(self, layer, **params):
        """Change style, geometry or transform parameters of a layer.

        Geometry keys use plot coordinates (``normal``, ``origin``) and plot
        axis names (``axis``), like the UI.
        """
        layer = self._resolve_layer_name(layer)
        spec = self._layers[layer]
        style, geom, trans = {}, {}, {}
        for k, v in params.items():
            k = self._STYLE_ALIASES.get(k, k)
            if k in self._TRANSFORM_KEYS:
                trans[k] = v
            elif k in self._PARAM_KEYS:
                geom["pos" if k == "slice_pos" else k] = v
            else:
                style[k] = v
        if geom:
            allowed = GEOMETRY_BY_TYPE.get(spec["type"], set())
            bad = sorted(set(geom) - allowed)
            if bad:
                raise ValueError(
                    f"{', '.join(bad)} do(es) not apply to '{layer}' "
                    f"({spec['type']} layer); allowed: "
                    f"{', '.join(sorted(allowed)) or 'none'}"
                )
            for k, v in geom.items():
                if k == "axis":
                    spec["axis_phys"] = self.frame.plot_axis_to_phys(v)
                elif k == "normal":
                    spec["normal"] = self._plot_vec_to_phys(v)
                elif k == "origin":
                    spec["origin0"] = self._plot_vec_to_phys(v)
                elif k in REF_KEYS:
                    spec[k] = self._check_volume(v) if v else None
                else:
                    spec[k] = v
            if "offset" in spec and "origin0" in spec:
                spec["offset"] = float(
                    np.clip(spec["offset"], *self._offset_range(spec))
                )
            if spec["type"] == "slice":
                spec["pos"] = int(
                    np.clip(
                        int(spec["pos"]),
                        0,
                        self.frame.n_along(spec["axis_phys"]) - 1,
                    )
                )
            self._params_to_widgets(layer)
            if spec["type"] in VOLUME_TYPES:
                self._vol_cache.pop(layer, None)
        if trans:
            self._set_transform_widgets(layer, **trans)
        if style:
            self._set_style(layer, **style)
        self._refresh(None if spec["type"] in VOLUME_TYPES else [layer])

    _apply_layer_params = set_layer_params

    # =========================================================================
    # Display
    # =========================================================================
    def show(self):
        display(self)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.call_later(0.0125, self._fit_plot_to_container)
        loop.call_later(0.025, self._fit_plot_to_container)

    def close(self):
        if not hasattr(
            self, "_rotation_task"
        ):  # closed before __init__ finished
            return super().close()
        self._stop_rotation()
        t = self._export_task
        if t is not None and not t.done():
            self._stop_export()
        super().close()
