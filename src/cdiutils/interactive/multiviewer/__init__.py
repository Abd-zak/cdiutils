"""
Building blocks of :class:`cdiutils.interactive.MultiVolumeViewer`.

The viewer class lives in :mod:`cdiutils.interactive.multiviewer_3d`; this
subpackage holds its parts:

- ``geometry``: coordinates, isosurfaces, slices, planes, clips (numpy only)
- ``analysis``: facets, line profiles, shell/core masks, mesh writers
- ``rendering``: still-image backends (Kaleido, VTK)
- ``export``: ordered frame pipeline and MP4 / GIF / PNG writers
- ``ui``: widgets and layer panel (mixin)
- ``tools``: probe, facets, statistics, orthoslices, linked cameras (mixin)
- ``anim``: animation export and live rotation (mixin)
- ``actions``: scripted actions (mixin)
- ``help``: built-in help and guided tutorial (mixin)
"""
