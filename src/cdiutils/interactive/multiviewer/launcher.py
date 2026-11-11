"""Notebook form to open cdiutils results in :class:`MultiVolumeViewer`."""

import os
from pathlib import Path

import ipywidgets as widgets
import numpy as np

from . import loader

_L = widgets.Layout
_D = {"description_width": "110px"}


def parse_units(text: str) -> dict:
    """``"het_strain=%, displacement=nm*0.1"`` -> ``{field: (unit, scale)}``."""
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Units: expected field=unit, got {part!r}")
        k, v = (x.strip() for x in part.split("=", 1))
        unit, _, scale = v.partition("*")
        out[k] = (unit.strip(), float(scale) if scale.strip() else 1.0)
    return out


def parse_floats(text: str, n=None):
    vals = [float(x) for x in text.replace(";", ",").split(",") if x.strip()]
    if n is not None and vals and len(vals) not in (1, n):
        raise ValueError(f"expected 1 or {n} numbers, got {len(vals)}")
    return vals


class Launcher(widgets.VBox):
    """Pick results, fields and options; *Open viewer* replaces the form
    with the viewer (``self.viewer``); *Back* returns to the form."""

    def __init__(self, viewer_cls, root=".", **viewer_kwargs):
        super().__init__(layout=_L(width="100%"))
        self._cls, self._kwargs = viewer_cls, viewer_kwargs
        self.viewer = None
        self._cache = None  # (config key, results, labels, notes)
        self.summary_text = ""

        self.folder = widgets.Text(
            value=str(root),
            description="Folder",
            placeholder="results folder, scan folder or file",
            style=_D,
            layout=_L(width="70%"),
        )
        self.find_btn = widgets.Button(
            description="Find results", layout=_L(width="130px")
        )
        self.scans = widgets.SelectMultiple(
            options=[],
            description="Results",
            rows=8,
            style=_D,
            layout=_L(width="95%"),
        )
        self.fields_box = widgets.HBox(
            layout=_L(flex_flow="row wrap", grid_gap="4px 14px")
        )
        self._field_cbs = {}
        self.labels = widgets.Text(
            description="Labels",
            placeholder="comma-separated (default: scan numbers)",
            style=_D,
            layout=_L(width="95%"),
        )
        self.voxel = widgets.Text(
            description="Voxel size (nm)",
            placeholder="from the file; or 5 or 5, 5, 5 (z, y, x)",
            style=_D,
            layout=_L(width="60%"),
        )
        self.units = widgets.Text(
            description="Units",
            placeholder="override, e.g. het_strain=%, displacement=nm*0.1",
            style=_D,
            layout=_L(width="95%"),
        )
        self.mask = widgets.Checkbox(
            value=True, description="NaN outside the support", indent=False
        )
        self.norm = widgets.Checkbox(
            value=True, description="Normalise amplitude", indent=False
        )
        self.script = widgets.Text(
            description="Start script",
            placeholder="optional .yaml / .json action script",
            style=_D,
            layout=_L(width="95%"),
        )
        self.show_btn = widgets.Button(
            description="Show configuration", layout=_L(width="170px")
        )
        self.open_btn = widgets.Button(
            description="Open viewer",
            button_style="primary",
            layout=_L(width="140px"),
        )
        self.status = widgets.HTML("")
        self.out = widgets.Output(layout=_L(width="100%"))
        self.back_btn = widgets.Button(
            description="◀ Back to the launcher", layout=_L(width="200px")
        )

        self.find_btn.on_click(lambda _b: self.find())
        self.folder.continuous_update = False  # Enter / leaving the field
        self.folder.observe(lambda _c: self.find(), names="value")
        self.scans.observe(lambda _c: self._refresh_fields(), names="value")
        self.show_btn.on_click(lambda _b: self.show_config())
        self.open_btn.on_click(lambda _b: self.open_viewer())
        self.back_btn.on_click(lambda _b: self.back())

        self.form = widgets.VBox(
            [
                widgets.HTML("<b>Open cdiutils results</b>"),
                widgets.HBox([self.folder, self.find_btn]),
                self.scans,
                widgets.HTML("Fields"),
                self.fields_box,
                self.labels,
                self.voxel,
                self.units,
                widgets.HBox([self.mask, self.norm]),
                self.script,
                widgets.HBox([self.show_btn, self.open_btn]),
                self.status,
                self.out,
            ],
            layout=_L(width="100%", padding="6px"),
        )
        self.children = [self.form]
        if root is not None and Path(root).expanduser().exists():
            self.find()

    # --- form -----------------------------------------------------------------
    def _error(self, e):
        self.status.value = (
            f"<b style='color:#d62728'>{type(e).__name__}: {e}</b>"
        )

    def find(self):
        try:
            root = Path(self.folder.value).expanduser()
            files = loader.find_results(root)
            base = root if root.is_dir() else root.parent
            self.scans.options = [
                (os.path.relpath(f, base), str(f)) for f in files
            ]
            self.scans.value = tuple(str(f) for f in files)
            self.status.value = (
                f"{len(files)} result(s) found"
                if files
                else "No cdiutils result found here."
            )
        except Exception as e:
            self._error(e)

    def _refresh_fields(self):
        sel = list(self.scans.value)
        prev = {k for k, cb in self._field_cbs.items() if cb.value}
        try:
            names = loader.peek_fields(sel[0]) if sel else []
        except Exception as e:
            self._error(e)
            names = []
        names = [n for n in names if n not in ("support", "surface")]
        defaults = set(loader.DEFAULT_FIELDS)
        self._field_cbs = {
            n: widgets.Checkbox(
                value=(n in prev) if prev else (n in defaults),
                description=n,
                indent=False,
                layout=_L(width="auto"),
            )
            for n in names
        }
        self.fields_box.children = list(self._field_cbs.values())

    def config(self) -> dict:
        """The options of the form, as ``from_cdiutils`` arguments."""
        if not self.scans.value:
            raise ValueError("Select at least one result.")
        fields = [k for k, cb in self._field_cbs.items() if cb.value] or None
        labels = [x.strip() for x in self.labels.value.split(",") if x.strip()]
        cfg = {
            "paths": list(self.scans.value),
            "fields": fields,
            "labels": labels or None,
            "mask_outside": self.mask.value,
            "normalise_amplitude": self.norm.value,
        }
        vs = parse_floats(self.voxel.value, 3)
        if vs:
            cfg["voxel_size"] = vs * 3 if len(vs) == 1 else vs
        units = parse_units(self.units.value)
        if units:
            cfg["field_units"] = units
        if self.script.value.strip():
            cfg["script"] = self.script.value.strip()
        return cfg

    def _load(self, cfg):
        key = repr(
            (
                cfg["paths"],
                cfg["fields"],
                cfg["labels"],
                cfg["mask_outside"],
                cfg["normalise_amplitude"],
            )
        )
        if self._cache is None or self._cache[0] != key:
            self.status.value = "Reading…"
            res = loader.load(
                cfg["paths"],
                cfg["fields"],
                cfg["labels"],
                mask_outside=cfg["mask_outside"],
                normalise_amplitude=cfg["normalise_amplitude"],
            )
            self._cache = (key, *res)
        return self._cache[1:]

    def show_config(self):
        """Read the selection and print the configuration (no viewer)."""
        try:
            cfg = self.config()
            results, labels, notes = self._load(cfg)
            vs = cfg.get("voxel_size") or next(
                (r.voxel_size for r in results if r.voxel_size), (1, 1, 1)
            )
            units = {
                k: v
                for k, v in loader.CDIUTILS_UNITS.items()
                if k in results[0].fields
            }
            units.update(cfg.get("field_units", {}))
            thr = next(
                (r.isosurface for r in results if r.isosurface is not None),
                None,
            )
            self.summary_text = loader.summary(
                results, labels, vs, "nm", units, thr, notes
            )
            self.out.clear_output()
            with self.out:
                print(self.summary_text)
            self.status.value = ""
        except Exception as e:
            self._error(e)

    def open_viewer(self):
        """Open the viewer with the current options (replaces the form)."""
        try:
            cfg = self.config()
            results, labels, notes = self._load(cfg)
            for r in results:  # the viewer may keep these arrays
                r.fields = {k: np.asarray(a) for k, a in r.fields.items()}
            kw = dict(self._kwargs)
            for k in ("voxel_size", "field_units"):
                if k in cfg:
                    kw[k] = cfg[k]
            self.out.clear_output()
            with self.out:
                self.viewer = self._cls._from_results(
                    results,
                    labels,
                    notes,
                    script=cfg.get("script"),
                    verbose=True,
                    **kw,
                )
            self.status.value = ""
            self.children = [
                widgets.HBox([self.back_btn]),
                self.out,
                self.viewer,
            ]
        except Exception as e:
            self._error(e)

    def back(self):
        """Close the viewer and show the form again."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.children = [self.form]
