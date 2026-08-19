"""
Generate the app icon: the CODE Lab logo's 'O' -- a ring of colored
segments (chromosome-ideogram style) on a transparent background.

Writes assets/codelab_o.png (1024x1024), assets/codelab_o.ico (Windows
shortcuts/taskbar), and, on macOS with iconutil available,
assets/codelab_o.icns.

Deterministic (seeded), so re-running reproduces the same mark.
"""
import os
import random
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw

SIZE = 1024
PAD = 90
WIDTH = 150
# Palette read off the logo: bright chromosome-paint colors on dark.
PALETTE = ['#e63946', '#f4a300', '#ffd23f', '#2a9d34', '#1f7a8c',
           '#3557d4', '#7b2fbf', '#e04f9e', '#ff6b35', '#12b5a5',
           '#c1121f', '#5aa9e6', '#8ac926', '#f9c74f']


def draw_ring(size=SIZE, pad=PAD, width=WIDTH, seed=19):
    rng = random.Random(seed)
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bbox = [pad, pad, size - pad, size - pad]
    start = rng.uniform(0, 360)
    covered = 0.0
    i = 0
    while covered < 360:
        span = rng.uniform(14, 34)
        gap = rng.uniform(3.5, 6.5)
        span = min(span, 360 - covered - gap) if 360 - covered > gap else max(0.0, 360 - covered)
        if span <= 0:
            break
        a0 = start + covered
        d.arc(bbox, a0, a0 + span, fill=PALETTE[i % len(PALETTE)], width=width)
        covered += span + gap
        i += 1
    return img, i


def build_icns(png_path, icns_path):
    """macOS .icns via sips + iconutil (both ship with macOS)."""
    iconset = icns_path.replace('.icns', '.iconset')
    if os.path.isdir(iconset):
        shutil.rmtree(iconset)
    os.makedirs(iconset)
    for edge in (16, 32, 64, 128, 256, 512):
        for scale, suffix in ((1, ''), (2, '@2x')):
            px = edge * scale
            out = os.path.join(iconset, f'icon_{edge}x{edge}{suffix}.png')
            subprocess.run(['sips', '-z', str(px), str(px), png_path, '--out', out],
                           check=True, capture_output=True)
    subprocess.run(['iconutil', '-c', 'icns', iconset, '-o', icns_path], check=True)
    shutil.rmtree(iconset)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets = os.path.join(repo, 'assets')
    os.makedirs(assets, exist_ok=True)
    png = os.path.join(assets, 'codelab_o.png')
    img, n = draw_ring()
    img.save(png)
    print(f'{png}: {n} segments')
    ico = os.path.join(assets, 'codelab_o.ico')
    img.save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(ico)
    if sys.platform == 'darwin' and shutil.which('iconutil'):
        icns = os.path.join(assets, 'codelab_o.icns')
        build_icns(png, icns)
        print(icns)


if __name__ == '__main__':
    main()
