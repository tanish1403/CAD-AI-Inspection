# Input data format

A case is a directory with one real image, an optional CAD render, one instance/stencil mask, and one JSON mapping.

Expected files:
- `BG_00.png` – real photograph.
- `FG_00.png` – CAD render.
- `Mask_00_00.bmp` – mask; this supplied dataset encodes integer stencil values in the red channel.
- `0000.json` – a mapping object whose first value is a list of objects with `Name` and `StencilValue`.

The loader derives a bounding box for each stencil value. If a value has zero pixels in the mask, its CAD reference crop is unavailable; it can still be sent to text detection, but the system should generally be conservative and return `UNCERTAIN` unless a separate registered expected ROI is available.

The sample contains 32 JSON part entries in TestCase2. Some stencil values have no pixels in the supplied mask, which is retained as missing reference evidence rather than silently interpreted as proof that the part is absent from the real photo.
