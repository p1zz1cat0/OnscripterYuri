#!/usr/bin/env python3
"""Convert CuNNy Magpie-format HLSL to Metal Shading Language compute kernels.

Parses the 4-pass CuNNy network (upstream funnyplanter/CuNNy, LGPL-3.0,
e.g. magpie/normal/CuNNy-veryfast-NVL.hlsl at commit 906031b) and emits
Metal compute kernels. All weights are expanded into explicit per-channel
multiply-adds so the port never depends on HLSL/MSL matrix layout
equivalence.

Generated MSL semantics:
  - textures are read via integer coordinates with clamp-to-edge (matching
    the Magpie POINT sampler default)
  - intermediate passes quantize through R8G8B8A8_UNORM writes like the
    HLSL textures
  - PASS 4 reproduces the YUV luma-correction out-shuffle (RY/YR matrices
    expanded row-wise), including HLSL normalized linear-sampler texel-center
    coordinates

Usage:
  cunny-hlsl2msl.py CuNNy-veryfast-NVL.hlsl > cunny.metal
"""

import re
import sys
from typing import List, Tuple


def parse_floats(text: str) -> List[float]:
    return [float(x) for x in text.split(",")]


def fmt_float(x: float) -> str:
    # Keep enough digits for exact reconstruction; strip trailing zeros.
    return format(x, ".9g")


class Pass:
    def __init__(self, number: int, desc: str, inputs: List[str], outputs: List[str]):
        self.number = number
        self.desc = desc
        self.inputs = inputs
        self.outputs = outputs
        # (target, vec4_or_matrix, weights, source)
        self.ops: List[Tuple[str, str, List[float], str]] = []


def parse_pass(block: str, number: int) -> Pass:
    desc = re.search(r"//!DESC\s+(\S+)", block)
    inputs = [t.strip() for t in re.search(r"//!IN\s+([^\n]+)", block).group(1).split(",")]
    outputs = [t.strip() for t in re.search(r"//!OUT\s+([^\n]+)", block).group(1).split(",")]

    p = Pass(number, desc.group(1) if desc else "", inputs, outputs)

    # PASS 1 uses per-element V4 multiplications:
    #   r0 += V4(w0, w1, w2, w3) * s0_1_1;
    for m in re.finditer(r"r(\d)\s*\+=\s*V4\(([^)]*)\)\s*\*\s*(s\d+_\d+_\d+);", block):
        p.ops.append(("r" + m.group(1), "vec4", parse_floats(m.group(2)), m.group(3)))

    # Bias rows:
    #   r0 += V4(w0, w1, w2, w3);
    for m in re.finditer(r"r(\d)\s*\+=\s*V4\(([^)]*)\);", block):
        p.ops.append(("r" + m.group(1), "bias", parse_floats(m.group(2)), ""))

    # Convolution rows:
    #   r0 += mul(s0_0_0, M4(w0, ..., w15));
    for m in re.finditer(r"r(\d)\s*\+=\s*mul\(\s*(s\d+_\d+_\d+)\s*,\s*M4\(([^)]*)\)\s*\);", block):
        p.ops.append(("r" + m.group(1), "mat4", parse_floats(m.group(3)), m.group(2)))

    return p


def msl_for_op(op) -> str:
    target, kind, w, src = op
    if kind == "vec4":
        return f"{target} += half4({', '.join(fmt_float(x) for x in w)}) * {src};"
    if kind == "bias":
        return f"{target} += half4({', '.join(fmt_float(x) for x in w)});"
    # mat4: HLSL row-major mul(vector, matrix):
    #   out[j] = sum_i v[i] * M[i][j]
    # M rows: row0 = w[0..3], row1 = w[4..7], row2 = w[8..11], row3 = w[12..15]
    lines = []
    for j in range(4):
        terms = [f"{src}.{'xyzw'[i]} * {fmt_float(w[i * 4 + j])}" for i in range(4)]
        lines.append(f"{target}.{'xyzw'[j]} += {' + '.join(terms)};")
    return "\n".join(lines)


def emit_sampler_helpers() -> str:
    return """    // Clamp-to-edge integer sampling (matches the Magpie POINTER sampler default).
    template <typename T>
    static inline T cunny_clamp(T v, T lo, T hi) { return metal::clamp(v, lo, hi); }

    static inline float4 cunny_sample_point(texture2d<float, access::read> tex,
                                            int2 p, int2 size) {
        int2 c = cunny_clamp(p, int2(0, 0), size - int2(1, 1));
        return tex.read(uint2(c));
    }

    static inline float4 cunny_sample_linear(texture2d<float, access::read> tex,
                                             float2 uv, int2 size) {
        float2 c = cunny_clamp(uv, float2(0.0, 0.0), float2(1.0, 1.0));
        // HLSL SampleLevel with a normalized linear sampler maps texel center
        // i to (i + 0.5) / size. Preserve that half-texel convention rather
        // than shifting the base color 0.5 input pixels down and right.
        float2 f = c * float2(size) - 0.5;
        int2 rawP0 = int2(metal::floor(f));
        int2 p0 = cunny_clamp(rawP0, int2(0, 0), size - int2(1, 1));
        int2 p1 = cunny_clamp(rawP0 + int2(1, 1), int2(0, 0), size - int2(1, 1));
        float2 t = f - metal::floor(f);
        float4 a = tex.read(uint2(p0));
        float4 b = tex.read(uint2(int2(p1.x, p0.y)));
        float4 c0 = tex.read(uint2(int2(p0.x, p1.y)));
        float4 d = tex.read(uint2(p1));
        return metal::mix(metal::mix(a, b, t.x), metal::mix(c0, d, t.x), t.y);
    }
"""


def emit_kernel(p: Pass, tex_names: dict) -> str:
    if p.number == 1:
        return emit_pass1(p, tex_names)
    if p.number == 4:
        return emit_pass4(p, tex_names)
    return emit_conv(p, tex_names)


def emit_pass1(p: Pass, tex_names: dict) -> str:
    in_tex = p.inputs[0]
    out_tex = p.outputs  # T0, T1
    body = []
    for op in p.ops:
        body.append(msl_for_op(op))
    return f"""kernel void cunny_p1(texture2d<float, access::read> {in_tex} [[texture(0)]],
                      texture2d<float, access::write> {out_tex[0]} [[texture(1)]],
                      texture2d<float, access::write> {out_tex[1]} [[texture(2)]],
                      uint2 gid [[thread_position_in_grid]]) {{
    const int2 size = int2({in_tex}.get_width(), {in_tex}.get_height());
    if (gid.x >= uint(size.x) || gid.y >= uint(size.y)) return;
    const int2 c = int2(gid);
    // L0(x, y) = min16float(dot(float3(0.299, 0.587, 0.114), INPUT(x, y).rgb))
    auto L0 = [&](int2 off) -> half {{
        float4 v = cunny_sample_point({in_tex}, c + off, size);
        return half(0.299f * v.x + 0.587f * v.y + 0.114f * v.z);
    }};
    half s0_0_0 = L0(int2(-1, -1)); half s0_0_1 = L0(int2(0, -1)); half s0_0_2 = L0(int2(1, -1));
    half s0_1_0 = L0(int2(-1, 0)); half s0_1_1 = L0(int2(0, 0)); half s0_1_2 = L0(int2(1, 0));
    half s0_2_0 = L0(int2(-1, 1)); half s0_2_1 = L0(int2(0, 1)); half s0_2_2 = L0(int2(1, 1));
    half4 r0 = 0.0h, r1 = 0.0h;
{chr(10).join('    ' + l for l in body)}
    r0 = metal::max(r0, 0.0h);
    {out_tex[0]}.write(float4(r0), gid);
    r1 = metal::max(r1, 0.0h);
    {out_tex[1]}.write(float4(r1), gid);
}}
"""


def emit_conv(p: Pass, tex_names: dict) -> str:
    kernel_name = f"cunny_p{p.number}"
    in_texs = p.inputs
    out_texs = p.outputs
    body = []
    for op in p.ops:
        body.append(msl_for_op(op))

    # Textures: inputs first (read), then outputs (write).
    params = []
    bind = 0
    for t in in_texs:
        params.append(f"texture2d<float, access::read> {t} [[texture({bind})]]")
        bind += 1
    for t in out_texs:
        params.append(f"texture2d<float, access::write> {t} [[texture({bind})]]")
        bind += 1
    params.append(f"uint2 gid [[thread_position_in_grid]]")

    # Sample helpers per input texture.
    sample_code = []
    for idx, t in enumerate(in_texs):
        sample_code.append(
            f"    auto L{idx} = [&](int2 off) -> half4 {{ return half4(cunny_sample_point({t}, c + off, size)); }};"
        )
    # Gather 3x3 neighbourhood for each input texture.
    decls = []
    for idx, t in enumerate(in_texs):
        base = f"s{idx}"
        for dy in range(3):
            for dx in range(3):
                decls.append(f"    half4 {base}_{dy}_{dx} = L{idx}(int2({dx - 1}, {dy - 1}));")
    writes = []
    for idx, t in enumerate(out_texs):
        writes.append(f"    {t}.write(float4(r{idx}), gid);")

    return f"""kernel void {kernel_name}({', '.join(params)}) {{
    const int2 size = int2({in_texs[0]}.get_width(), {in_texs[0]}.get_height());
    if (gid.x >= uint(size.x) || gid.y >= uint(size.y)) return;
    const int2 c = int2(gid);
{chr(10).join(sample_code)}
{chr(10).join(decls)}
    half4 r0 = 0.0h{', r1 = 0.0h' if len(out_texs) > 1 else ''};
{chr(10).join('    ' + l for l in body)}
    r0 = metal::max(r0, 0.0h);
{chr(10).join(writes)}
    {'r1 = metal::max(r1, 0.0h);' if len(out_texs) > 1 else ''}
}}
"""


def emit_pass4(p: Pass, tex_names: dict) -> str:
    body = []
    for op in p.ops:
        body.append(msl_for_op(op))
    return f"""kernel void cunny_p4(texture2d<float, access::read> INPUT [[texture(0)]],
                      texture2d<float, access::read> T0 [[texture(1)]],
                      texture2d<float, access::write> OUTPUT [[texture(2)]],
                      uint2 gid [[thread_position_in_grid]]) {{
    const int2 isz = int2(T0.get_width(), T0.get_height());
    const int2 osz = int2(OUTPUT.get_width(), OUTPUT.get_height());
    // Each thread produces a 2x2 block of output pixels.
    const int2 bc = int2(gid) * 2;
    if (bc.x >= osz.x || bc.y >= osz.y) return;
    const int2 c = int2(bc.x >> 1, bc.y >> 1);
    auto L0 = [&](int2 off) -> half4 {{ return half4(cunny_sample_point(T0, c + off, isz)); }};
    half4 s0_0_0 = L0(int2(-1, -1)); half4 s0_0_1 = L0(int2(0, -1)); half4 s0_0_2 = L0(int2(1, -1));
    half4 s0_1_0 = L0(int2(-1, 0)); half4 s0_1_1 = L0(int2(0, 0)); half4 s0_1_2 = L0(int2(1, 0));
    half4 s0_2_0 = L0(int2(-1, 1)); half4 s0_2_1 = L0(int2(0, 1)); half4 s0_2_2 = L0(int2(1, 1));
    half4 r0 = 0.0h;
{chr(10).join('    ' + l for l in body)}
    // YUV luma correction (RY / YR row-major matrices expanded).
    auto to_yuv = [&](float3 rgb) -> float3 {{
        return float3(0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z,
                      -0.169f * rgb.x - 0.331f * rgb.y + 0.5f * rgb.z,
                      0.5f * rgb.x - 0.419f * rgb.y - 0.081f * rgb.z);
    }};
    auto from_yuv = [&](float3 yuv) -> float3 {{
        return float3(yuv.x + -0.00093f * yuv.y + 1.401687f * yuv.z,
                      yuv.x + -0.3437f * yuv.y + -0.71417f * yuv.z,
                      yuv.x + 1.77216f * yuv.y + 0.00099f * yuv.z);
    }};
    const float2 opt = float2(1.0f / float(osz.x), 1.0f / float(osz.y));
    for (int dy = 0; dy < 2; ++dy) {{
        for (int dx = 0; dx < 2; ++dx) {{
            int2 opx = bc + int2(dx, dy);
            if (opx.x >= osz.x || opx.y >= osz.y) continue;
            float2 fpos = (float2(opx) + 0.5f) * opt;
            float3 yuv = to_yuv(cunny_sample_linear(INPUT, fpos, isz).xyz);
            int ch = dx + dy * 2;
            float luma = metal::saturate(yuv.x + float(r0[ch]));
            float3 rgb = from_yuv(float3(luma, yuv.y, yuv.z));
            OUTPUT.write(float4(rgb, 1.0f), uint2(opx));
        }}
    }}
}}
"""


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("usage: cunny_hlsl2msl.py <model.hlsl> [--header]", file=sys.stderr)
        return 1
    src = open(sys.argv[1], encoding="utf-8").read()
    header_mode = len(sys.argv) == 3 and sys.argv[2] == "--header"

    blocks = re.split(r"//!PASS\s+(\d)", src)
    # blocks: ["preamble", "1", "pass1...", "2", "pass2...", ...]
    passes = []
    for i in range(1, len(blocks), 2):
        number = int(blocks[i])
        passes.append(parse_pass(blocks[i + 1], number))
    passes.sort(key=lambda p: p.number)

    if [p.number for p in passes] != [1, 2, 3, 4]:
        print("error: expected passes 1..4", file=sys.stderr)
        return 1

    out = []
    out.append("// Generated by cunny_hlsl2msl.py from upstream CuNNy (LGPL-3.0).")
    out.append("// Source: funnyplanter/CuNNy commit 906031bb00c15dd6a6bbbaa21c0eb0b724ca8437")
    out.append("#include <metal_stdlib>\nusing namespace metal;\n")
    out.append(emit_sampler_helpers())
    for p in passes:
        out.append(emit_kernel(p, {}))
    msl = "\n".join(out)

    if header_mode:
        import textwrap
        escaped = msl.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
        header = textwrap.dedent(f"""\
            // Generated by cunny_hlsl2msl.py --header (do not edit by hand).
            // Source: funnyplanter/CuNNy commit 906031bb00c15dd6a6bbbaa21c0eb0b724ca8437
            #ifndef CUNNY_METAL_H
            #define CUNNY_METAL_H

            static const char *kCuNNyMSL = "{escaped}";

            #endif // CUNNY_METAL_H
            """)
        sys.stdout.write(header)
        return 0

    sys.stdout.write(msl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
