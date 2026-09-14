"""What the store already holds, per FOV and per step -- Qt-free.

A step's state for one FOV is read from the analysis store the way the
app's own append modes decide what is left to do (mips_present,
aligned_hybes, spot_slices, the cells / alleles / cellcycle capsules),
never from a flag somebody remembered to set:

    ingestion       every hybe of every modality has its MIP file
    segmentation    the FOV's cells capsule holds cells
    fov_alignment   every non-reference hybe has a persisted matrix
    cross_modal     a cross-modal matrix exists (n/a with one modality)
    cell_alignment  every cell carries per-hybe matrices      (deep)
    localization    every expected (hybe, channel) has a spot slice: the
                    config's localization round, the cell-cycle genes, and
                    every countable readout round
    celltype        every cell has a celltype                 (deep)
    cellcycle       the FOV's cellcycle.json places every cell (deep)
    tracing         every allele carries a traced polymer     (deep)

'deep' states open the FOV's HDF5 capsules (one read each); the shallow
ones are directory listings and manifest reads. A state is 'done',
'partial', 'missing' or 'n/a', with have/want counts and a detail.

A policy per step -- 'existing' (run only where not done), 'redo'
(run everywhere), 'skip' -- turns a status table into a plan.
"""
import os

from codelab_pipeline.io import analysis_store, paths, preprocess

STEPS = ('ingestion', 'segmentation', 'fov_alignment', 'cross_modal', 'cell_alignment',
         'localization', 'celltype', 'cellcycle', 'tracing')
DEEP_STEPS = ('cell_alignment', 'celltype', 'cellcycle', 'tracing')
POLICIES = ('existing', 'redo', 'skip')
STATES = ('done', 'partial', 'missing', 'n/a')
DEFAULT_POLICY = {'ingestion': 'existing', 'segmentation': 'existing', 'fov_alignment': 'existing',
                  'cross_modal': 'existing', 'cell_alignment': 'existing', 'localization': 'existing',
                  'celltype': 'existing', 'cellcycle': 'existing', 'tracing': 'existing'}
# segmentation is the user's own step: the runner never starts it on its
# own, a person has to ask for it explicitly
NEVER_AUTO = ('segmentation',)


# -- the project from its config -------------------------------------------

def load_project(config_path):
    """The project a config names, without Qt: root, FOV list, per
    modality the storage path, the layout path (project manifest first,
    the config's own attribute second) and the parsed hybe records, plus
    the config's parameter sections."""
    cfg = preprocess.load_xml_file(config_path)
    g = cfg.get('global', {})
    root = g.get('project_root', '')
    manifest = paths.read_manifest(root) if root else None
    modalities = {}
    for name, fields in cfg.get('modalities', {}).items():
        lp = ((manifest or {}).get('modalities', {}).get(name, {}) or {}).get('layout_path', '') or fields.get('layout_path', '')
        records = []
        if lp and os.path.exists(lp):
            try:
                records = preprocess.parse_experiment_layout(lp)
            except Exception as exc:                            # noqa: BLE001
                records = []
                fields = dict(fields, layout_error=f'{type(exc).__name__}: {exc}')
        modalities[name] = {'storage_path': os.path.join(root, name), 'layout_path': lp, 'records': records,
                            'fields': fields}
    names = [c.strip() for c in str(g.get('celltype_names', '')).split(',') if c.strip()]
    return {'config': config_path, 'project_root': root, 'fovs': list(g.get('fov_list') or []),
            'modalities': modalities, 'params': cfg.get('params', {}), 'celltype_names': names}


def hybes_of(records):
    return [str(r['folder']) for r in records]


def readout_sources(records):
    """[(hybe, channel)] of every readout channel (the fiducial channel
    of each hybe left out)."""
    out = []
    for r in records:
        fid = r.get('fiducial_channel')
        for ch in r.get('channels') or []:
            if fid is None or int(ch) != int(fid):
                out.append((str(r['folder']), int(ch)))
    return out


def _parse_source_label(text):
    """'Hyb_101 (RNA)' -> ('RNA', 'Hyb_101'); 'Hyb_101' -> (None, 'Hyb_101'); '' -> None."""
    text = str(text or '').strip()
    if not text:
        return None
    if text.endswith(')') and '(' in text:
        hybe, mod = text[:-1].rsplit('(', 1)
        return mod.strip(), hybe.strip()
    return None, text


def localization_targets(project):
    """{modality: [(hybe, channel)]} the store should hold spot slices
    for: the config's spot_localization hybe/channel (the anchor round
    for tracing), the cell-cycle panel's gene sources when the config
    carries them ('MOD|HYBE|CH|GENE|ROLE', comma separated), and every
    countable readout round (an _mRNA / _exon readout that is not a
    Rep_/Toe_ round) at its readout channels. Nascent, intron, repeat,
    toe and anonymous (DNA) rounds are not expected unless named."""
    from codelab_pipeline.analysis import cellcycle as CC
    params = project.get('params', {}) or {}
    out = {name: [] for name in project['modalities']}

    def add(mod, hybe, ch):
        if mod in out and (str(hybe), int(ch)) not in out[mod]:
            out[mod].append((str(hybe), int(ch)))
    loc = params.get('spot_localization', {}) or {}
    src = _parse_source_label(loc.get('hybe', ''))
    if src and str(loc.get('channel', '')).strip().isdigit():
        mod, hybe = src
        if mod is None and len(out) == 1:
            mod = next(iter(out))
        if mod is not None:
            add(mod, hybe, int(loc['channel']))
    for part in str((params.get('cellcycle', {}) or {}).get('genes', '')).split(','):
        bits = part.strip().split('|')
        if len(bits) >= 3 and bits[2].strip().isdigit():
            add(bits[0].strip(), bits[1].strip(), int(bits[2]))
    for name, m in project['modalities'].items():
        for r in m['records']:
            if CC.countable_round(r.get('readout_name')):
                fid = r.get('fiducial_channel')
                for ch in r.get('channels') or []:
                    if fid is None or int(ch) != int(fid):
                        add(name, r['folder'], ch)
    return out


def reference_hybe(project, section, modality):
    """The reference hybe a config section names for a modality
    (fov_alignment / cell_alignment / cross_modal_alignment), or ''."""
    sec = project.get('params', {}).get(section, {}) or {}
    return str(sec.get(f'reference_hybe_{modality}', '') or '')


# -- per-step states ------------------------------------------------------------

def _state(have, want, none_is='missing'):
    if want <= 0:
        return 'n/a'
    if have >= want:
        return 'done'
    return 'partial' if have > 0 else none_is


def _combine(parts):
    """The FOV's state over modalities: the least advanced one wins."""
    order = {'missing': 0, 'partial': 1, 'done': 2, 'n/a': 3}
    real = [p for p in parts if p['state'] != 'n/a']
    if not real:
        return {'state': 'n/a', 'have': 0, 'want': 0, 'detail': '; '.join(p['detail'] for p in parts if p['detail'])}
    worst = min(real, key=lambda p: order[p['state']])
    return {'state': worst['state'], 'have': sum(p['have'] for p in real), 'want': sum(p['want'] for p in real),
            'detail': '; '.join(p['detail'] for p in parts if p['detail'])}


def step_ingestion(project, fov):
    parts = []
    for name, m in project['modalities'].items():
        want = hybes_of(m['records'])
        present = paths.mips_present(m['storage_path'], fov)
        have = [h for h in want if h in present]
        parts.append({'state': _state(len(have), len(want)), 'have': len(have), 'want': len(want),
                      'detail': f'{name} {len(have)}/{len(want)} MIPs'})
    return _combine(parts)


def _first_storage(project):
    return next(iter(project['modalities'].values()))['storage_path']


def step_segmentation(project, fov, cells=None):
    n = len(cells) if cells is not None else int(analysis_store.fov_counts(_first_storage(project), [fov])[int(fov)]['cells'])
    return {'state': 'done' if n > 0 else 'missing', 'have': n, 'want': n if n > 0 else 1, 'detail': f'{n} cells'}


def step_fov_alignment(project, fov):
    parts = []
    for name, m in project['modalities'].items():
        ref = reference_hybe(project, 'fov_alignment', name)
        want = [h for h in hybes_of(m['records']) if h != ref]
        aligned = analysis_store.aligned_hybes(m['storage_path'], fov)
        have = [h for h in want if h in aligned]
        parts.append({'state': _state(len(have), len(want)), 'have': len(have), 'want': len(want),
                      'detail': f'{name} {len(have)}/{len(want)} matrices' + (f' (ref {ref})' if ref else '')})
    return _combine(parts)


def step_cross_modal(project, fov):
    mods = list(project['modalities'].items())
    if len(mods) < 2:
        return {'state': 'n/a', 'have': 0, 'want': 0, 'detail': 'one modality'}
    found = []
    names = [n for n, _m in mods]
    for name, m in mods:
        # star topology: each store keys its bridge by the OTHER modality
        # (and '_' for a value migrated from the flat layout)
        for other in [o for o in names if o != name] + [None]:
            try:
                H = analysis_store.read_cross_modal_matrix(m['storage_path'], fov, modality=other)
            except Exception:                                   # noqa: BLE001
                H = None
            if H is not None:
                found.append(f'{name}<-{other or "_"}')
                break
    return {'state': 'done' if found else 'missing', 'have': len(found), 'want': 1,
            'detail': ('bridge in ' + ', '.join(found)) if found else 'no cross-modal matrix'}


def step_cell_alignment(project, fov, cells):
    if cells is None:
        return {'state': 'n/a', 'have': 0, 'want': 0, 'detail': 'no cells'}
    have = sum(1 for c in cells if c.get('matrices'))
    return {'state': _state(have, len(cells)), 'have': have, 'want': len(cells), 'detail': f'{have}/{len(cells)} cells with matrices'}


def step_localization(project, fov, targets=None):
    parts = []
    targets = targets if targets is not None else localization_targets(project)
    for name, m in project['modalities'].items():
        want = targets.get(name, [])
        slices = {(h, int(c)) for (mod, h, c) in analysis_store.spot_slices(m['storage_path'], fov) if mod == name}
        have = [s for s in want if s in slices]
        parts.append({'state': _state(len(have), len(want)), 'have': len(have), 'want': len(want),
                      'detail': f'{name} {len(have)}/{len(want)} spot slices'})
    return _combine(parts)


def step_celltype(project, fov, cells):
    if cells is None:
        return {'state': 'n/a', 'have': 0, 'want': 0, 'detail': 'no cells'}
    have = sum(1 for c in cells if str(c.get('celltype') or ''))
    return {'state': _state(have, len(cells)), 'have': have, 'want': len(cells), 'detail': f'{have}/{len(cells)} cells typed'}


def step_cellcycle(project, fov, cells):
    sp = _first_storage(project)
    cap = analysis_store.read_fov_cellcycle(sp, fov) or {}
    rows = cap.get('rows') or []
    n_cells = len(cells) if cells is not None else 0
    if n_cells == 0:
        return {'state': 'n/a', 'have': 0, 'want': 0, 'detail': 'no cells'}
    placed = {int(r['cell']) for r in rows}
    have = sum(1 for c in cells if int(c['id']) in placed)
    detail = f'{have}/{n_cells} cells placed'
    if have < n_cells and rows:
        detail += ' (the rest under the min panel total, or newer cells)'
    # a capsule that names every countable cell is done even when some
    # cells fell under the minimum panel total: the app never places those
    state = 'done' if (rows and have >= min(n_cells, len(rows))) else ('partial' if have else 'missing')
    return {'state': state, 'have': have, 'want': n_cells, 'detail': detail}


def step_tracing(project, fov, alleles, cells=None):
    if alleles is None:
        return {'state': 'n/a', 'have': 0, 'want': 0, 'detail': 'not read'}
    if cells is not None and len(cells) == 0:
        return {'state': 'n/a', 'have': 0, 'want': 0, 'detail': 'no cells'}
    if not alleles:
        return {'state': 'missing', 'have': 0, 'want': 1, 'detail': 'no alleles built'}
    traced = sum(1 for a in alleles if any(v for v in (a.get('polymer_adj') or {}).values()))
    return {'state': _state(traced, len(alleles)), 'have': traced, 'want': len(alleles),
            'detail': f'{traced}/{len(alleles)} alleles traced'}


def fov_status(project, fov, deep=True, targets=None):
    """{step: {'state', 'have', 'want', 'detail'}} for one FOV."""
    sp = _first_storage(project)
    targets = targets if targets is not None else localization_targets(project)
    cells = None
    alleles = None
    if deep:
        cells, _ = analysis_store.read_cells(sp, fov)
        alleles = analysis_store.read_fov_alleles(sp, fov)
    out = {'ingestion': step_ingestion(project, fov),
           'segmentation': step_segmentation(project, fov, cells if deep else None),
           'fov_alignment': step_fov_alignment(project, fov),
           'cross_modal': step_cross_modal(project, fov),
           'localization': step_localization(project, fov, targets)}
    if deep:
        out['cell_alignment'] = step_cell_alignment(project, fov, cells)
        out['celltype'] = step_celltype(project, fov, cells)
        out['cellcycle'] = step_cellcycle(project, fov, cells)
        out['tracing'] = step_tracing(project, fov, alleles, cells if cells is not None else [])
    else:
        for s in DEEP_STEPS:
            out[s] = {'state': 'n/a', 'have': 0, 'want': 0, 'detail': 'shallow'}
    return {s: out[s] for s in STEPS}


def status_table(project, fovs=None, deep=True, on_fov=None):
    """[{'fov': f, <step>: state-dict, ...}] over the FOVs (the config's
    when None); on_fov(i, n, fov) after each."""
    fovs = list(fovs) if fovs is not None else list(project['fovs'])
    targets = localization_targets(project)
    rows = []
    for i, f in enumerate(fovs):
        row = {'fov': int(f)}
        row.update(fov_status(project, int(f), deep=deep, targets=targets))
        rows.append(row)
        if on_fov is not None:
            on_fov(i + 1, len(fovs), int(f))
    return rows


def project_level(project):
    """The store's project-wide facts the steps depend on: whether a
    cell-cycle model and a celltype config exist."""
    sp = _first_storage(project)
    try:
        model = analysis_store.read_cellcycle_model(sp)
    except Exception:                                           # noqa: BLE001
        model = None
    try:
        ranges, channels, _cal, method = analysis_store.read_celltype_config(sp)
        ct = bool(ranges) or bool(channels)
    except Exception:                                           # noqa: BLE001
        ct = False
    return {'cellcycle_model': bool(model), 'celltype_config': ct}


# -- summaries and plans ------------------------------------------------------------

def summarize(rows):
    """{step: {'done': n, 'partial': n, 'missing': n, 'n/a': n}}"""
    out = {}
    for s in STEPS:
        counts = {st: 0 for st in STATES}
        for r in rows:
            counts[r[s]['state']] += 1
        out[s] = counts
    return out


def plan(rows, policies=None):
    """{step: [fov, ...]} the policies ask to run: 'existing' -> the FOVs
    not done (partial or missing; n/a never), 'redo' -> every FOV whose
    step applies, 'skip' -> none."""
    pol = dict(DEFAULT_POLICY)
    pol.update(policies or {})
    out = {}
    for s in STEPS:
        p = pol.get(s, 'existing')
        if p not in POLICIES:
            raise ValueError(f'{s}: unknown policy {p!r} (one of {POLICIES})')
        if p == 'skip':
            out[s] = []
        elif p == 'redo':
            out[s] = [r['fov'] for r in rows if r[s]['state'] != 'n/a']
        else:
            out[s] = [r['fov'] for r in rows if r[s]['state'] in ('partial', 'missing')]
    return out


def parse_policy(text):
    """'segmentation=skip,localization=redo' -> {step: policy}."""
    out = {}
    for part in (text or '').split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f'policy {part!r}: expected step=policy')
        k, v = (x.strip() for x in part.split('=', 1))
        if k not in STEPS:
            raise ValueError(f'unknown step {k!r} (one of {STEPS})')
        if v not in POLICIES:
            raise ValueError(f'unknown policy {v!r} (one of {POLICIES})')
        out[k] = v
    return out


def parse_fovs(text):
    """'1-5,7,9-10' -> [1,2,3,4,5,7,9,10]."""
    out = []
    for part in str(text or '').replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


_MARK = {'done': 'done', 'partial': 'part', 'missing': '----', 'n/a': ' n/a'}
LABEL = {'ingestion': 'ingest', 'segmentation': 'segment', 'fov_alignment': 'fov_align', 'cross_modal': 'crossmodal',
         'cell_alignment': 'cell_align', 'localization': 'localize', 'celltype': 'celltype', 'cellcycle': 'cellcycle',
         'tracing': 'tracing'}


def format_grid(rows, steps=STEPS):
    """The FOV x step grid as text, one FOV per line."""
    head = 'fov  ' + ' '.join(f'{LABEL.get(s, s)[:10]:>10s}' for s in steps)
    lines = [head, '-' * len(head)]
    for r in rows:
        lines.append(f'{r["fov"]:>3d}  ' + ' '.join(f'{_MARK[r[s]["state"]]:>10s}' for s in steps))
    return '\n'.join(lines)


def format_summary(rows, steps=STEPS):
    summ = summarize(rows)
    lines = []
    for s in steps:
        c = summ[s]
        lines.append(f'{s:15s} done {c["done"]:3d}  partial {c["partial"]:3d}  missing {c["missing"]:3d}  n/a {c["n/a"]:3d}')
    return '\n'.join(lines)
