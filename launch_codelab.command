#!/bin/sh
# CODE Lab Imaging Pipeline launcher -- macOS (double-clickable) / Linux.
# Same resolution order as launch_codelab.bat, so both platforms behave
# identically:
#   1. CODELAB_PYTHON env var (explicit interpreter -- the reliable knob)
#   2. conda (on PATH, or common miniforge/miniconda/anaconda install
#      locations), running the cellclassifier env -- but ONLY if that env
#      actually exists. A machine with conda but no cellclassifier used to
#      die here on "EnvironmentLocationNotFound" instead of falling
#      through, even when conda's base env had every dependency.
#   3. conda's base env, then plain python3, and hope the active
#      environment has the deps
cd "$(dirname "$0")" || exit 1

CODELAB_ENV=${CODELAB_ENV:-cellclassifier}

if [ -n "$CODELAB_PYTHON" ]; then
    exec "$CODELAB_PYTHON" main.py
fi

CONDA=
for CANDIDATE in conda \
                 "$HOME/miniforge3/bin/conda" \
                 "$HOME/miniconda3/bin/conda" \
                 "$HOME/anaconda3/bin/conda" \
                 /usr/local/Caskroom/miniforge/base/bin/conda \
                 /opt/homebrew/Caskroom/miniforge/base/bin/conda; do
    if command -v "$CANDIDATE" >/dev/null 2>&1; then
        CONDA=$CANDIDATE
        break
    fi
done

if [ -n "$CONDA" ]; then
    CONDA_ROOT=$("$CONDA" info --base 2>/dev/null)
    if [ -x "$CONDA_ROOT/envs/$CODELAB_ENV/bin/python" ]; then
        exec "$CONDA" run -n "$CODELAB_ENV" --no-capture-output python main.py
    fi
    if [ -x "$CONDA_ROOT/bin/python" ]; then
        echo "[launcher] conda env \"$CODELAB_ENV\" not found -- using conda base at $CONDA_ROOT."
        exec "$CONDA" run -n base --no-capture-output python main.py
    fi
fi

if command -v python3 >/dev/null 2>&1; then
    exec python3 main.py
fi

echo "[launcher] No usable Python found."
echo "[launcher] Point CODELAB_PYTHON at the interpreter you want, e.g."
echo "[launcher]   export CODELAB_PYTHON=\$HOME/anaconda3/bin/python"
exit 1
