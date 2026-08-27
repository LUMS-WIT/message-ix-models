"""Minimal, dependency-free readers for the ESRI Shapefile format (``.dbf``
attribute tables and ``.shp`` polygon geometry) -- shared by every
``build_*_IRB.py`` conversion script that needs one, rather than each
re-implementing its own copy (as ``build_basin_links_IRB.py`` originally
did) or reaching for a GIS library (none is available in this environment).

Both formats are simple, well-documented binary layouts -- see the `dBASE
III specification
<https://en.wikipedia.org/wiki/.dbf>`_ and `ESRI's Shapefile technical
description
<https://www.esri.com/content/dam/esrisites/sitecore-archive/Files/Pdfs/library/whitepapers/pdfs/shapefile.pdf>`_.
Only what's actually needed is implemented: attribute records (no memo
fields), and Polygon-type geometry (shape type 5) reduced to a plain area
per record via the shoelace formula -- not full ring/hole topology, since
nothing here needs the actual boundary, only relative areas for weighting.
"""

import struct
from pathlib import Path


def read_dbf(path: Path) -> list[dict]:
    """Read a ``.dbf`` attribute table's records as a list of dicts."""
    data = path.read_bytes()
    n_records = struct.unpack("<I", data[4:8])[0]
    header_len = struct.unpack("<H", data[8:10])[0]
    record_len = struct.unpack("<H", data[10:12])[0]

    fields = []
    pos = 32
    while data[pos] != 0x0D:
        name = data[pos : pos + 11].split(b"\x00")[0].decode("latin1")
        flen = data[pos + 16]
        fields.append((name, flen))
        pos += 32

    records = []
    rec_start = header_len
    for _ in range(n_records):
        rec = data[rec_start : rec_start + record_len]
        rec_start += record_len
        if not rec or rec[0:1] == b"*":
            continue
        offset = 1
        row = {}
        for name, flen in fields:
            row[name] = rec[offset : offset + flen].decode("latin1", errors="replace").strip()
            offset += flen
        records.append(row)
    return records


def _ring_area(points: list[tuple[float, float]]) -> float:
    """Signed polygon area via the shoelace formula (positive if the ring
    winds counter-clockwise, negative if clockwise)."""
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def read_shp_polygon_areas(path: Path) -> list[float]:
    """Read a Polygon-type ``.shp`` file's records, in file order (aligned
    positionally with the matching ``.dbf``'s records -- standard
    shapefile convention), returning each record's total area (sum of its
    rings' signed areas, since a real ring winding clockwise -- a
    "hole" -- correctly subtracts; absolute value taken at the end since
    only relative magnitude is needed here, never sign).

    Only shape type 5 (Polygon) is supported -- this project's shapefiles
    (basin/province outlines) are always this type.
    """
    data = path.read_bytes()
    shape_type = struct.unpack("<I", data[32:36])[0]
    if shape_type != 5:
        raise ValueError(f"Expected Polygon shape type (5), got {shape_type}")

    areas = []
    pos = 100  # end of the 100-byte file header
    while pos < len(data):
        record_number = struct.unpack(">I", data[pos : pos + 4])[0]  # noqa: F841
        content_len_words = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        content_start = pos + 8
        content_len_bytes = content_len_words * 2

        rec_shape_type = struct.unpack("<I", data[content_start : content_start + 4])[0]
        if rec_shape_type == 0:  # null shape
            areas.append(0.0)
        else:
            num_parts = struct.unpack("<I", data[content_start + 36 : content_start + 40])[0]
            num_points = struct.unpack("<I", data[content_start + 40 : content_start + 44])[0]
            parts_start = content_start + 44
            parts = [
                struct.unpack("<I", data[parts_start + 4 * i : parts_start + 4 * i + 4])[0]
                for i in range(num_parts)
            ]
            points_start = parts_start + 4 * num_parts
            points = [
                struct.unpack(
                    "<dd", data[points_start + 16 * i : points_start + 16 * i + 16]
                )
                for i in range(num_points)
            ]
            part_bounds = [*parts, num_points]
            total = sum(
                _ring_area(points[part_bounds[i] : part_bounds[i + 1]])
                for i in range(num_parts)
            )
            areas.append(abs(total))

        pos = content_start + content_len_bytes
    return areas
