"""CVAT integration for MaskReview (P8).

`export` is the pure-offline layer (no cvat-sdk dependency): it converts SAM2 masks
and the review queue into CVAT-importable artifacts. The live cvat-sdk client lives in
a separate module so this layer stays unit-testable without a running CVAT.
See docs/cvat_plugin.md.
"""
