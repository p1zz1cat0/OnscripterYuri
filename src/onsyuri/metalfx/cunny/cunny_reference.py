#!/usr/bin/env python3
"""CPU reference implementation of the CuNNy veryfast-NVL network.

Independently reimplements the forward pass of the upstream HLSL
(funnyplanter/CuNNy commit 906031b, LGPL-3.0) so the MSL port can be
validated against a non-uniform golden frame. Small integer differences are
expected because Metal executes the network with half-precision arithmetic.

Weights are parsed from the same HLSL source as the MSL generator, but the
computation is a separate implementation (numpy, explicit loops), so a close
agreement between reference and port validates the weight expansion,
neighbourhood coordinates, normalized sampling, pixel shuffle and thread
mapping.

Usage:
  cunny_reference.py <input.bgra> <width> <height> <output.bgra>
"""

import sys
import struct
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cunny_hlsl2msl import parse_pass, Pass  # noqa: E402
import cunny_hlsl2msl as _conv


def load_bgra(path: Path, w: int, h: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8).astype(np.float32) / 255.0
    img = data.reshape(h, w, 4)  # B, G, R, A
    rgb = img[:, :, [2, 1, 0]]  # -> R, G, B
    return rgb


def save_bgra(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    out[:, :, [2, 1, 0]] = rgb
    out[:, :, 3] = 1.0
    quant = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    quant.tofile(path)


def quantize_unorm(v: np.ndarray) -> np.ndarray:
    """Simulate a write to an R8G8B8A8_UNORM texture."""
    return np.clip(np.rint(v * 255.0), 0, 255) / 255.0


def clamp_idx(i, n):
    return min(max(i, 0), n - 1)


def gather3(img: np.ndarray, y: int, x: int) -> list:
    """3x3 neighbourhood of a HxWxC image, clamped at the edges."""
    h, w, c = img.shape
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out.append(img[clamp_idx(y + dy, h), clamp_idx(x + dx, w)])
    return out


def vec4_mul(v, w):
    """out[j] = sum_i v[i] * w[i*4+j]  (HLSL mul(v, M4) row-major)."""
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(4, 4)
    return w.T @ v


def run_pass1(img: np.ndarray, ops: list) -> tuple:
    h, w, _ = img.shape
    t0 = np.zeros((h, w, 4), dtype=np.float32)
    t1 = np.zeros((h, w, 4), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            # L0 over 3x3 of luma values
            neigh = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = clamp_idx(y + dy, h), clamp_idx(x + dx, w)
                    neigh.append(0.299 * img[yy, xx, 0] + 0.587 * img[yy, xx, 1] + 0.114 * img[yy, xx, 2])
            r0 = np.zeros(4, dtype=np.float64)
            r1 = np.zeros(4, dtype=np.float64)
            for target, kind, wts, src in ops:
                if kind == "vec4":
                    v = neigh[int(src.split("_")[1]) * 3 + int(src.split("_")[2])]
                    if target == "r0":
                        r0 += np.asarray(wts, dtype=np.float64) * v
                    else:
                        r1 += np.asarray(wts, dtype=np.float64) * v
                elif kind == "bias":
                    if target == "r0":
                        r0 += wts
                    else:
                        r1 += wts
                else:
                    raise AssertionError(f"unexpected op kind {kind} in pass1")
            r0 = np.clip(r0, 0.0, None)
            r1 = np.clip(r1, 0.0, None)
            t0[y, x] = quantize_unorm(r0.astype(np.float32))
            t1[y, x] = quantize_unorm(r1.astype(np.float32))
    return t0, t1


def run_conv(img_a: np.ndarray, img_b: np.ndarray | None, ops: list, out_channels: int) -> list:
    h, w, _ = img_a.shape
    outs = [np.zeros((h, w, 4), dtype=np.float32) for _ in range(out_channels)]
    for y in range(h):
        for x in range(w):
            na = gather3(img_a, y, x)          # 9 x 4
            nb = gather3(img_b, y, x) if img_b is not None else []
            samples = {}
            for i in range(9):
                samples[f"s0_{i // 3}_{i % 3}"] = np.asarray(na[i], dtype=np.float64)
            for i in range(9):
                samples[f"s1_{i // 3}_{i % 3}"] = np.asarray(nb[i], dtype=np.float64)
            rs = [np.zeros(4, dtype=np.float64) for _ in range(out_channels)]
            for target, kind, wts, src in ops:
                idx = int(target[1])
                if kind == "mat4":
                    rs[idx] += vec4_mul(samples[src], wts)
                elif kind == "bias":
                    rs[idx] += wts
                else:
                    raise AssertionError(f"unexpected op kind {kind}")
            for i in range(out_channels):
                rs[i] = np.clip(rs[i], 0.0, None)
                outs[i][y, x] = quantize_unorm(rs[i].astype(np.float32))
    return outs


def bilinear(img: np.ndarray, u: float, v: float) -> np.ndarray:
    """Simulate HLSL SampleLevel with a clamped normalized linear sampler."""
    h, w, c = img.shape
    u = min(max(u, 0.0), 1.0)
    v = min(max(v, 0.0), 1.0)
    fx = u * w - 0.5
    fy = v * h - 0.5
    raw_x0 = int(np.floor(fx))
    raw_y0 = int(np.floor(fy))
    x0 = clamp_idx(raw_x0, w)
    y0 = clamp_idx(raw_y0, h)
    x1 = clamp_idx(raw_x0 + 1, w)
    y1 = clamp_idx(raw_y0 + 1, h)
    tx = fx - raw_x0
    ty = fy - raw_y0
    a = img[y0, x0]
    b = img[y0, x1]
    cc = img[y1, x0]
    d = img[y1, x1]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (cc * (1 - tx) + d * tx) * ty


def run_pass4(t0: np.ndarray, input_rgb: np.ndarray, ops: list) -> np.ndarray:
    h, w, _ = t0.shape
    oh, ow = h * 2, w * 2
    out = np.zeros((oh, ow, 3), dtype=np.float32)
    for by in range(h):
        for bx in range(w):
            na = gather3(t0, by, bx)
            samples = {f"s0_{i // 3}_{i % 3}": np.asarray(na[i], dtype=np.float64) for i in range(9)}
            r0 = np.zeros(4, dtype=np.float64)
            for target, kind, wts, src in ops:
                if kind == "mat4":
                    r0 += vec4_mul(samples[src], wts)
                elif kind == "bias":
                    r0 += wts
                else:
                    raise AssertionError(f"unexpected op kind {kind} in pass4")
            # YUV luma correction
            for dy in range(2):
                for dx in range(2):
                    opx = (bx * 2 + dx, by * 2 + dy)
                    u = (opx[0] + 0.5) / ow
                    v = (opx[1] + 0.5) / oh
                    rgb = bilinear(input_rgb, u, v)
                    yuv = np.array([
                        0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2],
                        -0.169 * rgb[0] - 0.331 * rgb[1] + 0.5 * rgb[2],
                        0.5 * rgb[0] - 0.419 * rgb[1] - 0.081 * rgb[2],
                    ])
                    ch = dx + dy * 2
                    luma = min(max(yuv[0] + r0[ch], 0.0), 1.0)
                    yuv[0] = luma
                    rgb_out = np.array([
                        yuv[0] + -0.00093 * yuv[1] + 1.401687 * yuv[2],
                        yuv[0] + -0.3437 * yuv[1] + -0.71417 * yuv[2],
                        yuv[0] + 1.77216 * yuv[1] + 0.00099 * yuv[2],
                    ])
                    out[opx[1], opx[0]] = rgb_out
    return quantize_unorm(out)


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: cunny_reference.py <input.bgra> <width> <height> <output.bgra>", file=sys.stderr)
        return 1
    in_path = Path(sys.argv[1])
    w, h = int(sys.argv[2]), int(sys.argv[3])
    out_path = Path(sys.argv[4])

    hlsl = Path(__file__).parent / "CuNNy-veryfast-NVL.hlsl"
    src = hlsl.read_text(encoding="utf-8")

    import re
    blocks = re.split(r"//!PASS\s+(\d)", src)
    passes = []
    for i in range(1, len(blocks), 2):
        passes.append(parse_pass(blocks[i + 1], int(blocks[i])))
    passes.sort(key=lambda p: p.number)
    p1, p2, p3, p4 = passes

    rgb = load_bgra(in_path, w, h)
    t0, t1 = run_pass1(rgb, p1.ops)
    t2, t3 = run_conv(t0, t1, p2.ops, 2)
    t0 = run_conv(t2, t3, p3.ops, 1)[0]
    out_rgb = run_pass4(t0, rgb, p4.ops)
    save_bgra(out_path, out_rgb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
