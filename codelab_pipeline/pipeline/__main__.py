"""The pipeline CLI (Qt-free half).

    python -m codelab_pipeline.pipeline status <config.xml> [--fovs 1-5,7] [--shallow] [--detail]
    python -m codelab_pipeline.pipeline plan   <config.xml> [--fovs ...] [--policy step=existing,step=redo,...]

`status` reads the store and prints the FOV x step grid; `plan` prints
which FOVs each step would run under the policies (default: every step
'existing' = only what is not done; segmentation is never started by
the runner on its own). Running is the app's job (windows/
pipeline_wiring.py), headless through an offscreen MainWindow.
"""
import argparse
import sys
import time

from codelab_pipeline.pipeline import status as S


def _project(args):
    project = S.load_project(args.config)
    if not project['project_root']:
        sys.exit(f'{args.config}: no project_root in the config')
    if args.fovs:
        project['fovs'] = S.parse_fovs(args.fovs)
    if not project['fovs']:
        sys.exit('no FOVs: give --fovs or put fov_list in the config')
    for name, m in project['modalities'].items():
        if not m['records']:
            print(f'warning: {name}: no layout records ({m["layout_path"] or "no layout path"}'
                  + (f'; {m["fields"]["layout_error"]}' if 'layout_error' in m['fields'] else '') + ')', file=sys.stderr)
    return project


def _rows(project, args):
    t0 = time.time()
    rows = S.status_table(project, deep=not args.shallow,
                          on_fov=(lambda i, n, f: print(f'  [{i}/{n}] fov {f}', file=sys.stderr, flush=True)) if args.progress else None)
    print(f'# {project["config"]}: {len(rows)} FOVs, {"shallow" if args.shallow else "deep"} status in {time.time() - t0:.1f}s',
          file=sys.stderr)
    return rows


def cmd_status(args):
    project = _project(args)
    rows = _rows(project, args)
    print(S.format_grid(rows))
    print()
    print(S.format_summary(rows))
    lvl = S.project_level(project)
    print(f'\nproject: cell-cycle model {"present" if lvl["cellcycle_model"] else "absent"}; '
          f'celltype config {"present" if lvl["celltype_config"] else "absent"}')
    if args.detail:
        print()
        for r in rows:
            print(f'fov {r["fov"]}: ' + ' | '.join(f'{s}: {r[s]["detail"]}' for s in S.STEPS if r[s]['detail']))


def cmd_plan(args):
    project = _project(args)
    rows = _rows(project, args)
    pol = S.parse_policy(args.policy)
    p = S.plan(rows, pol)
    eff = dict(S.DEFAULT_POLICY)
    eff.update(pol)
    for s in S.STEPS:
        fovs = p[s]
        note = ''
        if s in S.NEVER_AUTO and eff[s] != 'skip' and fovs:
            note = '   (segmentation is never started by the runner on its own: choose it explicitly in the app)'
        print(f'{s:15s} {eff[s]:8s} -> {len(fovs):3d} FOVs' + (f': {fovs}' if fovs and len(fovs) <= 40 else '') + note)


def cmd_run(args):
    """The plan, then the run through an offscreen MainWindow (the app's
    own batch actions), unless --dry-run."""
    project = _project(args)
    pol = S.parse_policy(args.policy)
    rows = _rows(project, args)
    p = S.plan(rows, pol)
    eff = dict(S.DEFAULT_POLICY)
    eff.update(pol)
    todo = [(s, p[s]) for s in S.STEPS if p[s] and eff[s] != 'skip' and (s not in S.NEVER_AUTO or s in pol)]
    for s in S.STEPS:
        print(f'{s:15s} {eff[s]:8s} -> {len(p[s]):3d} FOVs' + ('' if (s not in S.NEVER_AUTO or s in pol or not p[s]) else '   (not run: choose segmentation explicitly)'))
    if not todo:
        print('nothing to run')
        return
    if args.dry_run:
        print('dry run: would run ' + ', '.join(f'{s} ({len(f)} FOVs, {"overwrite" if eff[s] == "redo" else "append"})' for s, f in todo))
        return
    from windows.pipeline_wiring import run_headless
    report = run_headless(args.config, fovs=project['fovs'], policies=pol, explicit=tuple(pol), log=lambda m: print(m, flush=True),
                          timeout_s=args.timeout)
    print()
    for e in report['steps']:
        print(f'{e["step"]:15s} {e.get("seconds", 0):7.1f}s  {e.get("summary") or e.get("error", "")}' + (f'  [{e["after"]}]' if e.get('after') else ''))
    if report['failed']:
        print('failed:', report['failed'])
        sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m codelab_pipeline.pipeline')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('status', cmd_status), ('plan', cmd_plan), ('run', cmd_run)):
        sp = sub.add_parser(name)
        sp.add_argument('config')
        sp.add_argument('--fovs', default='')
        sp.add_argument('--shallow', action='store_true', help='directory listings and manifests only (no HDF5 opened)')
        sp.add_argument('--progress', action='store_true')
        if name == 'status':
            sp.add_argument('--detail', action='store_true')
        else:
            sp.add_argument('--policy', default='', help='step=existing|redo|skip, comma separated')
        if name == 'run':
            sp.add_argument('--dry-run', action='store_true', help='print what would run and exit')
            sp.add_argument('--timeout', type=float, default=None, help='seconds before the run stops after its current step')
        sp.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
