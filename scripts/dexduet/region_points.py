#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""DexDuet contact-region surface points of the G1 Dex3 hands, in their link frames -> ``dex3_region_points.npz``.

Port of DexDuet ``dexmachina/envs/ho_contact.py`` ``robot_anchor_points`` (with ``collision_elements``,
``_LinkSurface``, ``_split``, the primitive samplers and ``rpy_matrix``; github.com/HaoranZhangumich/DexDuet),
applied to the collision geometry FlashCHORD simulates (``assets/urdfs/g1/main_with_hand.urdf``). Regions and links
(DexDuet finger_spec CONTACT_DEPTH dex3_hand (2, 1)): palm = palm_link; thumb = thumb_1 + thumb_2; index = index_1;
middle = middle_1. Points per region (ANCHOR_POINTS): 256 / 64 / 48 / 48, split over a region's links by area (the
remainder to the largest), each link sampled with ``trimesh.sample.sample_surface(mesh, n, seed=0)``.
Run in the FlashCHORD environment (trimesh). Read-only on the assets.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REGIONS = ("palm", "thumb", "index", "middle")
REGION_LINKS = {
    "palm": ("hand_palm_link",),
    "thumb": ("hand_thumb_1_link", "hand_thumb_2_link"),
    "index": ("hand_index_1_link",),
    "middle": ("hand_middle_1_link",),
}
ANCHOR_POINTS = {"palm": 256, "thumb": 64, "index": 48, "middle": 48}
SIDES = ("left", "right")


def _floats(text, default):
    return tuple(float(v) for v in text.split()) if text is not None else default


def rpy_matrix(rpy):
    """URDF origin rpy: fixed-axis roll (x), pitch (y), yaw (z), i.e. R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def collision_elements(urdf_path):
    """Every <collision> element of every link: {link: [dict(kind, ..., xyz, rpy)]}."""
    root, base = ET.parse(urdf_path).getroot(), os.path.dirname(urdf_path)
    out = {}
    for link in root.findall("link"):
        items = []
        for col in link.findall("collision"):
            org = col.find("origin")
            xyz = _floats(org.get("xyz") if org is not None else None, (0.0, 0.0, 0.0))
            rpy = _floats(org.get("rpy") if org is not None else None, (0.0, 0.0, 0.0))
            geom = col.find("geometry")
            shapes = [] if geom is None else list(geom)
            if len(shapes) != 1:
                raise ValueError(f'{link.get("name")}: a collision geometry needs exactly one shape, found {len(shapes)}')
            g = shapes[0]
            item = dict(kind=g.tag, xyz=xyz, rpy=rpy)
            if g.tag == "mesh":
                item.update(
                    path=os.path.join(base, g.get("filename").replace("package://", "")),
                    scale=_floats(g.get("scale"), (1.0, 1.0, 1.0)),
                )
            elif g.tag == "box":
                item.update(size=_floats(g.get("size"), None))
            elif g.tag == "cylinder":
                item.update(radius=float(g.get("radius")), length=float(g.get("length")))
            elif g.tag == "sphere":
                item.update(radius=float(g.get("radius")))
            else:
                raise ValueError(f'{link.get("name")}: unsupported collision geometry <{g.tag}>')
            items.append(item)
        if items:
            out[link.get("name")] = items
    return out


def is_plain_mesh(items):
    return (
        len(items) == 1
        and items[0]["kind"] == "mesh"
        and all(v == 0.0 for v in items[0]["xyz"] + items[0]["rpy"])
        and items[0]["scale"] == (1.0, 1.0, 1.0)
    )


def sample_box(size, count, rng):
    a, b, c = (float(v) for v in size)
    areas = np.array([b * c, b * c, a * c, a * c, a * b, a * b])
    face = rng.choice(6, size=count, p=areas / areas.sum())
    pts = (rng.random((count, 3)) - 0.5) * np.array([a, b, c])
    axis, sign = face // 2, np.where(face % 2 == 0, 1.0, -1.0)
    pts[np.arange(count), axis] = sign * np.array([a, b, c])[axis] / 2.0
    return pts


def sample_cylinder(radius, length, count, rng):
    side, cap = 2.0 * np.pi * radius * length, np.pi * radius**2
    part = rng.choice(3, size=count, p=np.array([side, cap, cap]) / (side + 2.0 * cap))
    theta = rng.random(count) * 2.0 * np.pi
    r = np.where(part == 0, radius, radius * np.sqrt(rng.random(count)))
    z = np.where(part == 0, (rng.random(count) - 0.5) * length, np.where(part == 1, 0.5, -0.5) * length)
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=-1)


def sample_sphere(radius, count, rng):
    v = rng.normal(size=(count, 3))
    return radius * v / np.linalg.norm(v, axis=-1, keepdims=True)


def _element_area(item, mesh):
    if item["kind"] == "mesh":
        return float(mesh.area)
    if item["kind"] == "box":
        a, b, c = item["size"]
        return 2.0 * (a * b + b * c + a * c)
    if item["kind"] == "cylinder":
        return 2.0 * np.pi * item["radius"] * item["length"] + 2.0 * np.pi * item["radius"] ** 2
    return 4.0 * np.pi * item["radius"] ** 2


def _split(total, areas):
    """Points per entry, proportional to area; the remainder goes to the largest entry."""
    areas = np.asarray(areas, dtype=float)
    counts = np.floor(total * areas / areas.sum()).astype(int)
    counts[np.argmax(areas)] += total - counts.sum()
    return counts


class _LinkSurface:
    def __init__(self, items):
        import trimesh

        self.items = items
        self.plain = is_plain_mesh(items)
        self.meshes = []
        for item in items:
            mesh = None
            if item["kind"] == "mesh":
                mesh = trimesh.load(item["path"], force="mesh")
                if item["scale"] != (1.0, 1.0, 1.0):
                    mesh = trimesh.Trimesh(
                        vertices=np.asarray(mesh.vertices) * np.asarray(item["scale"]),
                        faces=np.asarray(mesh.faces),
                        process=False,
                    )
            self.meshes.append(mesh)
        self.areas = [_element_area(item, mesh) for item, mesh in zip(self.items, self.meshes)]
        self.area = float(sum(self.areas))

    def sample(self, count, seed):
        import trimesh

        if self.plain:
            return trimesh.sample.sample_surface(self.meshes[0], int(count), seed=seed)[0]
        out = []
        for i, (item, mesh, n) in enumerate(zip(self.items, self.meshes, _split(count, self.areas))):
            n = int(n)
            if n == 0:
                continue
            element_seed = [int(seed), i]
            if item["kind"] == "mesh":
                p = trimesh.sample.sample_surface(mesh, n, seed=element_seed)[0]
            else:
                rng = np.random.default_rng(element_seed)
                if item["kind"] == "box":
                    p = sample_box(item["size"], n, rng)
                elif item["kind"] == "cylinder":
                    p = sample_cylinder(item["radius"], item["length"], n, rng)
                else:
                    p = sample_sphere(item["radius"], n, rng)
            out.append(np.asarray(p, dtype=np.float64) @ rpy_matrix(item["rpy"]).T + np.asarray(item["xyz"]))
        return np.concatenate(out, 0) if out else np.zeros((0, 3))


def robot_anchor_points(urdf_path, side, seed=0):
    elements = collision_elements(urdf_path)
    links, pts, slices, sources = [], [], [], {}
    point_link = []
    for region in REGIONS:
        names = [f"{side}_{suffix}" for suffix in REGION_LINKS[region]]
        missing = [n for n in names if n not in elements]
        if missing:
            raise ValueError(f"{side}: {region} region links {missing} have no collision geometry in {urdf_path}")
        surfaces = [_LinkSurface(elements[n]) for n in names]
        counts = _split(ANCHOR_POINTS[region], [surface.area for surface in surfaces])
        start = sum(len(p) for p in pts)
        for name, surface, count in zip(names, surfaces, counts):
            sample = surface.sample(int(count), seed).astype(np.float32)
            links.append(name)
            pts.append(sample)
            point_link.extend([len(links) - 1] * len(sample))
            sources[name] = [
                {**{k: v for k, v in item.items() if k != "path"}, "path": item.get("path"), "sha256": _sha(item.get("path"))}
                for item in elements[name]
            ]
        slices.append((start, sum(len(p) for p in pts)))
    return links, np.concatenate(pts, 0), np.asarray(point_link, dtype=np.int64), np.asarray(slices), sources


def _sha(path):
    if path is None:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    import json

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--urdf", default=str(Path(__file__).resolve().parents[2] / "src/flash_chord/assets/urdfs/g1/main_with_hand.urdf"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    payload = {"regions": np.asarray(REGIONS), "seed": np.asarray(args.seed), "urdf_sha256": np.asarray(_sha(args.urdf))}
    report = {}
    for side in SIDES:
        links, points, point_link, slices, sources = robot_anchor_points(args.urdf, side, args.seed)
        payload[f"points_{side}"] = points
        payload[f"links_{side}"] = np.asarray(links)
        payload[f"point_link_{side}"] = point_link
        payload[f"region_slices_{side}"] = slices
        report[side] = {
            "links": links,
            "slices": slices.tolist(),
            "points_per_link": np.bincount(point_link).tolist(),
            "sources": sources,
            "extent_m": (points.max(0) - points.min(0)).round(4).tolist(),
        }
    tmp = Path(args.out).with_suffix(".tmp.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, args.out)
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
