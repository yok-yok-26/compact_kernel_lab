#!/usr/bin/env python3
import csv
import math
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
bench_dir = ROOT / "reports" / "benchmark"
out_dir = ROOT / "reports" / "roofline"
out_dir.mkdir(parents=True, exist_ok=True)

PEAK_BW_GBPS = 672.0  # placeholder until replaced with measured/sustained bandwidth
PEAK_COMPUTE_GFLOPS = 30000.0  # placeholder until replaced with measured peak

points = []
for path in sorted(bench_dir.glob("*.csv")):
    with path.open() as f:
        row = next(csv.DictReader(f), None)
        if not row:
            continue
    mode = row.get("mode", "unknown")
    batch_size = row.get("batch_size", "1")
    n = row.get("n", "0")
    ms = float(row["avg_ms"])
    logical_gb = float(row["logical_gb"])
    gbps = float(row["logical_gbps"])
    total_elements = float(row.get("total_elements", float(batch_size) * float(n)))
    est_ops = max(total_elements, 1.0)
    gflops = est_ops / (ms / 1000.0) / 1e9
    bytes_moved = max(logical_gb * 1e9, 1.0)
    ai = est_ops / bytes_moved
    points.append((mode, int(float(batch_size)), int(float(n)), ai, gflops, ms, gbps, path.name))

if not points:
    raise SystemExit("No benchmark CSV files found. Run scripts/run_benchmark.sh first.")

W, H = 1100, 760
M = 90
WHITE = (255, 255, 255)
BLACK = (25, 25, 25)
GRID = (220, 220, 220)
BLUE = (32, 92, 180)
RED = (210, 60, 40)
GREEN = (30, 140, 80)
ORANGE = (230, 145, 40)

def png_write(path, pixels):
    raw = b"".join(b"\x00" + bytes(px for rgb in row for px in rgb) for row in pixels)
    def chunk(tag, data):
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xffffffff)
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack("!IIBBBBB", W, H, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw, 9))
    data += chunk(b"IEND", b"")
    path.write_bytes(data)

def draw_line(pixels, x0, y0, x1, y1, color):
    x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        if 0 <= x0 < W and 0 <= y0 < H:
            pixels[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy

def draw_rect(pixels, x, y, r, color):
    for yy in range(max(0, y-r), min(H, y+r+1)):
        for xx in range(max(0, x-r), min(W, x+r+1)):
            pixels[yy][xx] = color

def tiny_text(pixels, x, y, text, color=BLACK):
    for i, ch in enumerate(text[:48]):
        v = ord(ch)
        xx = x + i * 4
        for bit in range(7):
            if v & (1 << bit):
                for yy in range(y + bit * 2, y + bit * 2 + 2):
                    if 0 <= xx < W and 0 <= yy < H:
                        pixels[yy][xx] = color

def render(path, global_view):
    if global_view:
        xmin, xmax = 1e-3, 1e3
        ymin, ymax = 1e-3, max(PEAK_COMPUTE_GFLOPS * 2, max(p[4] for p in points) * 4)
    else:
        xmin = max(min(p[3] for p in points) / 4, 1e-4)
        xmax = max(max(p[3] for p in points) * 4, xmin * 10)
        ymin = max(min(p[4] for p in points) / 4, 1e-3)
        ymax = max(max(p[4] for p in points) * 4, ymin * 10)
    lx0, lx1 = math.log10(xmin), math.log10(xmax)
    ly0, ly1 = math.log10(ymin), math.log10(ymax)
    def sx(x): return int(M + (math.log10(max(x, xmin)) - lx0) / (lx1 - lx0) * (W - 2*M))
    def sy(y): return int(H - M - (math.log10(max(y, ymin)) - ly0) / (ly1 - ly0) * (H - 2*M))

    pixels = [[WHITE for _ in range(W)] for _ in range(H)]
    draw_line(pixels, M, H-M, W-M, H-M, BLACK)
    draw_line(pixels, M, M, M, H-M, BLACK)
    for exp in range(math.floor(lx0), math.ceil(lx1) + 1):
        draw_line(pixels, sx(10 ** exp), M, sx(10 ** exp), H-M, GRID)
    for exp in range(math.floor(ly0), math.ceil(ly1) + 1):
        draw_line(pixels, M, sy(10 ** exp), W-M, sy(10 ** exp), GRID)

    xs = [10 ** (lx0 + i * (lx1 - lx0) / 300) for i in range(301)]
    prev = None
    for x in xs:
        y = min(PEAK_COMPUTE_GFLOPS, x * PEAK_BW_GBPS)
        cur = (sx(x), sy(y))
        if prev: draw_line(pixels, prev[0], prev[1], cur[0], cur[1], BLUE)
        prev = cur

    colors = [RED, GREEN, ORANGE, BLACK]
    for idx, (mode, batch_size, n, ai, gflops, ms, gbps, source) in enumerate(points):
        x, y = sx(ai), sy(gflops)
        color = colors[idx % len(colors)]
        draw_rect(pixels, x, y, 5, color)
        tiny_text(pixels, min(x + 8, W - 260), max(y - 8, M), f"{mode} B{batch_size} N{n} {ms:.3g}ms {gbps:.1f}GB/s", color)

    tiny_text(pixels, M, 25, "batched compact roofline estimated; see compact_latest.md")
    tiny_text(pixels, M, H-45, "x: arithmetic intensity estimated ops/logical byte")
    tiny_text(pixels, 12, M, "y: estimated GFLOP/s")
    png_write(path, pixels)

render(out_dir / "compact_latest.png", global_view=False)
render(out_dir / "compact_global_latest.png", global_view=True)
with (out_dir / "compact_latest.md").open("w") as f:
    f.write("# Batched Compact Roofline Notes\n\n")
    f.write("The PNGs are generated with a no-dependency Python renderer so the lab works without matplotlib.\n\n")
    f.write("Data source: benchmark CSV logical bytes. After NCU profiling, replace/augment this with measured DRAM bytes and throughput from reports/ncu.\n\n")
    f.write(f"Peak assumptions are placeholders: bandwidth={PEAK_BW_GBPS} GB/s, compute={PEAK_COMPUTE_GFLOPS} GFLOP/s. Treat conclusions from the line as estimated.\n\n")
    f.write("`compact_latest.png` is a local view around measured points. `compact_global_latest.png` includes the roofline elbow.\n\n")
    for mode, batch_size, n, ai, gflops, ms, gbps, source in points:
        f.write(f"- mode={mode} batch_size={batch_size} n={n} ai={ai:.6g} est_gflops={gflops:.6g} avg_ms={ms:.6g} logical_gbps={gbps:.6g} source={source}\n")
print(out_dir / "compact_latest.png")
print(out_dir / "compact_global_latest.png")
print(out_dir / "compact_latest.md")
