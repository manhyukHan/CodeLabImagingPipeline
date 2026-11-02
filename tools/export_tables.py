# -*- coding: utf-8 -*-
"""Export a project's cells, spots and alleles without starting the app.

The export has lived behind a button since it was written, which meant
the only way to get the tables off a store was to open the GUI, point
it at the project and wait. Everything it does is already Qt-free
(codelab_pipeline.analysis.export reads the store and nothing else), so
this is the same call from a terminal -- and the only one that can run
on a server, in a cron job, or over ssh.

    python tools/export_tables.py <project-root> [-o OUT] [--fovs 1-9,12]
    python tools/export_tables.py <project-root> --format csv
    python tools/export_tables.py <project-root> --combined

A project root is the directory HOLDING the modality stores, the one
with manifest.json in it -- `.../2026-10-11_MAZ_IF+15ug`, not
`.../2026-10-11_MAZ_IF+15ug/DNA`. Every modality in the manifest is
exported in one pass; passing a modality store directly works too and
exports just that one.

WHAT COMES OUT

    <out>/cells.csv          one row per segmented cell
    <out>/spots.csv          one row per spot
    <out>/alleles.csv        one row per (allele, bin, candidate)
    <out>/cells_arrays.npz   the masks and matrices the table summarises
    <out>/combined.csv       --combined: spots with their cell joined
                             on, compact -- the keys, the position and
                             every field the pipeline's own schema does
                             not declare, which is to say the
                             measurements somebody attached.
                             --combined-full keeps every column.

or one .xlsx with a sheet each, when --format xlsx (the default when
every table fits -- Excel stops at ~1.05M rows and the writer refuses
rather than truncating, so a large project falls back to CSV and says
so).

NO FIELD LIST LIVES HERE. The columns are discovered from the data, so
a field a project writes onto its cells or spots comes out without this
tool or export.py naming it.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from codelab_pipeline.analysis import export            # noqa: E402
from codelab_pipeline.io import paths as store_paths    # noqa: E402


def parse_fovs(text):
    """'1-9,12,15' -> [1..9, 12, 15]. None when empty."""
    if not text:
        return None
    out = []
    for part in str(text).replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part[1:]:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def modalities_of(root):
    """{modality: storage_path} for a project root, or for one store.

    A path that IS a modality store (no manifest beside it, but a
    manifest one level up naming it) is accepted and exported alone --
    that is what someone types when they have a store path to hand,
    and refusing it would be pedantry.
    """
    root = os.path.abspath(os.path.normpath(root))
    man = store_paths.read_manifest(root) or {}
    mods = dict(man.get('modalities') or {})
    if mods:
        return root, {m: os.path.join(root, m) for m in mods}
    parent = os.path.dirname(root)
    man = store_paths.read_manifest(parent) or {}
    mods = dict(man.get('modalities') or {})
    name = os.path.basename(root)
    if name in mods:
        return parent, {name: root}
    raise SystemExit(
        'no manifest.json naming any modality at %s (nor one naming %r '
        'at %s). Point this at the project root -- the directory that '
        'HOLDS the DNA/RNA stores.' % (root, name, parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('project', help='the project root (holds manifest.json)')
    ap.add_argument('-o', '--out', default=None,
                    help='output directory (default: <project>/export)')
    ap.add_argument('--fovs', default=None,
                    help="e.g. '1-9,12'; default every FOV in the store")
    ap.add_argument('--format', choices=('xlsx', 'csv'), default='xlsx')
    ap.add_argument('--combined', action='store_true',
                    help='also write the compact spot x cell join')
    ap.add_argument('--combined-full', action='store_true',
                    help='... with every column instead of the '
                         'analysis-relevant ones')
    ap.add_argument('--no-arrays', action='store_true',
                    help='skip the masks and matrices (.npz)')
    a = ap.parse_args(argv)

    root, by_modality = modalities_of(a.project)
    out = os.path.abspath(a.out or os.path.join(root, 'export'))
    os.makedirs(out, exist_ok=True)
    fovs = parse_fovs(a.fovs)
    print('project   %s' % root)
    print('modality  %s' % ', '.join('%s -> %s' % (m, p)
                                     for m, p in sorted(by_modality.items())))
    print('fovs      %s' % ('every one in the store' if fovs is None
                            else '%d listed' % len(fovs)))
    print('out       %s' % out)
    print()

    def progress(i, total, msg):
        sys.stdout.write('\r  %d/%d  %s   ' % (i, total, msg))
        sys.stdout.flush()

    t = export.collect(by_modality, fovs=fovs, progress=progress,
                       want_arrays=not a.no_arrays)
    print()
    s = t.summary
    print('read %d FOV(s): %d cells, %d spots, %d allele rows'
          % (len(s['fovs']), s['n_cells'], s['n_spots'], s['n_allele_rows']))

    sheets = t.sheets()
    if a.combined or a.combined_full:
        compact = not a.combined_full
        rows = export.combined_rows(t, compact=compact)
        sheets.append(('combined', rows,
                       export.combined_columns(rows, t,
                                               compact=compact)))
        print('combined: %d rows x %d columns (%s)'
              % (len(rows), len(sheets[-1][2]),
                 'compact' if compact else 'every column'))

    wrote = []
    if a.format == 'xlsx':
        try:
            wrote.append(export.write_excel(
                sheets, os.path.join(out, 'tables.xlsx')))
        except ValueError as e:
            print('  %s' % e)
            print('  writing CSV instead')
            a.format = 'csv'
    if a.format == 'csv':
        for name, rows, cols in sheets:
            wrote.append(export.write_csv(
                rows, cols, os.path.join(out, name + '.csv')))
    if t.arrays:
        p = export.write_npz(t.arrays,
                             os.path.join(out, 'cells_arrays.npz'))
        if p:
            wrote.append(p)

    print()
    for p in wrote:
        print('  wrote %s (%.2f MB)' % (p, os.path.getsize(p) / 1048576.0))
    if s['errors']:
        print()
        print('NOT everything succeeded:')
        for e in s['errors'][:12]:
            print('  %s' % e)
        if len(s['errors']) > 12:
            print('  ... and %d more' % (len(s['errors']) - 12))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
