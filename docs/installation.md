# Installation and launch

## Get the code

```console
git clone https://github.com/manhyukHan/CodeLabImagingPipeline.git
cd CodeLabImagingPipeline
```

Everything else — the Python environment included — can be bootstrapped
from that checkout.

## The environment

The app runs in a conda environment named `code_lab_imaging_pipeline`
whose contents are defined by **`requirements.txt` at the repo root**.
That file is the single source of truth: add a dependency there, never
into an env by hand. Two pins in it matter enough to know about:

- `cellpose>=3.1,<4` — cellpose 4 removed the `models.Cellpose` class the
  segmentation code builds; 3.x is required.
- `torch==2.7.1+cu118` — pulled from PyTorch's CUDA 11.8 index so
  segmentation runs on the GPU. Machines without an NVIDIA GPU still
  work; the same wheel falls back to CPU.

Expect the first install to download a few GB (the CUDA torch wheel) and
take several minutes, once.

## Launch

The repo ships three launchers that all resolve the interpreter the same
way:

| Launcher | Platform | Behavior |
|---|---|---|
| `launch_codelab.bat` | Windows | console window; can prompt |
| `launch_codelab.vbs` | Windows | double-click, **no console** |
| `launch_codelab.command` | macOS | double-clickable |

Resolution order:

1. `CODELAB_PYTHON` environment variable — an explicit interpreter path,
   the reliable knob when anything below guesses wrong.
2. The `code_lab_imaging_pipeline` conda env, if conda reports it exists
   (the pre-rename `cellclassifier` env is also accepted).
3. **Offer to create the env** from `requirements.txt` — so a fresh
   machine needs nothing but conda and one `y`. Set
   `CODELAB_BOOTSTRAP=yes` to skip the prompt, `=never` to stop being
   asked.
4. Conda's base env, then plain `python`, as last resorts.

Manual route, equivalent to what the launcher automates:

```console
conda create -y -n code_lab_imaging_pipeline python=3.11
conda run -n code_lab_imaging_pipeline python -m pip install -r requirements.txt
conda run -n code_lab_imaging_pipeline python main.py
```

`main.py` is the entry point either way.

## Using the analysis toolbox without the app

The analysis package (`codelab_pipeline.analysis`) imports no Qt and no
app module — a promise pinned by its own test suite. From any script or
notebook running in the same environment:

```python
import sys
sys.path.insert(0, r'path/to/CodeLabImagingPipeline')

from codelab_pipeline.analysis import population, gate, ensemble
```

See {doc}`analysis/index` for the full tour.
