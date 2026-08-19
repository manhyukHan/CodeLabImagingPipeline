#!/bin/sh
# CODE Lab Imaging Pipeline launcher -- macOS (double-clickable) / Linux.
# Same resolution order as launch_codelab.bat, so both platforms behave
# identically:
#   1. CODELAB_PYTHON env var (explicit interpreter -- the reliable knob)
#   2. conda (on PATH, or common miniforge/miniconda install locations),
#      running the cellclassifier env
#   3. plain python3 and hope the active environment has the deps
cd "$(dirname "$0")" || exit 1

if [ -n "$CODELAB_PYTHON" ]; then
    exec "$CODELAB_PYTHON" main.py
fi

for CONDA in conda \
             "$HOME/miniforge3/bin/conda" \
             "$HOME/miniconda3/bin/conda" \
             "$HOME/anaconda3/bin/conda" \
             /usr/local/Caskroom/miniforge/base/bin/conda \
             /opt/homebrew/Caskroom/miniforge/base/bin/conda; do
    if command -v "$CONDA" >/dev/null 2>&1; then
        exec "$CONDA" run -n cellclassifier --no-capture-output python main.py
    fi
done

exec python3 main.py
