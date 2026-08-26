#!/bin/sh
# CODE Lab Imaging Pipeline launcher -- macOS (double-clickable) / Linux.
# Same resolution order as launch_codelab.bat, so both platforms behave
# identically:
#   1. CODELAB_PYTHON env var (explicit interpreter -- the reliable knob)
#   2. the code_lab_imaging_pipeline conda env, if conda says it exists
#      (or the pre-rename "cellclassifier" env, if only that one exists)
#   3. offer to CREATE that env from requirements.txt and then use it
#   4. conda's base env, then plain python3, and hope the active
#      environment has the deps
#
# Step 3 exists because base is not a safe fallback for this project any
# more: requirements.txt pins cellpose below 4 (4.x deleted the
# models.Cellpose class segment.py builds), and a base env carrying
# cellpose 4 imports fine and only fails once someone runs segmentation.
# A purpose-built env is the fix; creating it should not be a manual chore.
cd "$(dirname "$0")" || exit 1

CODELAB_ENV=${CODELAB_ENV:-code_lab_imaging_pipeline}
# Pre-rename name -- same requirements.txt, so an existing one is used
# rather than forcing a multi-GB rebuild for a rename alone.
CODELAB_LEGACY_ENV=${CODELAB_LEGACY_ENV:-cellclassifier}
CODELAB_PY_VERSION=${CODELAB_PY_VERSION:-3.11}

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

run_in_env() {
    exec "$CONDA" run -n "$CODELAB_ENV" --no-capture-output python main.py
}

base_fallback() {
    CONDA_ROOT=$("$CONDA" info --base 2>/dev/null)
    if [ -n "$CONDA_ROOT" ] && [ -x "$CONDA_ROOT/bin/python" ]; then
        echo "[launcher] using conda base at $CONDA_ROOT."
        exec "$CONDA" run -n base --no-capture-output python main.py
    fi
    plain_fallback
}

plain_fallback() {
    if command -v python3 >/dev/null 2>&1; then
        exec python3 main.py
    fi
    echo "[launcher] No usable Python found."
    echo "[launcher] Point CODELAB_PYTHON at the interpreter you want, e.g."
    echo "[launcher]   export CODELAB_PYTHON=\$HOME/anaconda3/bin/python"
    exit 1
}

bootstrap() {
    echo "[launcher] creating $CODELAB_ENV (python $CODELAB_PY_VERSION)..."
    if ! "$CONDA" create -y -n "$CODELAB_ENV" "python=$CODELAB_PY_VERSION"; then
        echo "[launcher] setting up \"$CODELAB_ENV\" FAILED -- see above."
        base_fallback
    fi
    echo "[launcher] installing requirements.txt..."
    if ! "$CONDA" run -n "$CODELAB_ENV" --no-capture-output python -m pip install -r requirements.txt; then
        echo "[launcher] installing requirements FAILED -- see above."
        base_fallback
    fi
    echo "[launcher] environment ready."
    run_in_env
}

if [ -z "$CONDA" ]; then
    plain_fallback
fi

# Ask CONDA whether the env exists, never the filesystem: envs_dirs can put
# it on any drive, and this project's own env does not live under the conda
# install root.
if "$CONDA" env list 2>/dev/null | grep -q "^$CODELAB_ENV[[:space:]]"; then
    run_in_env
fi

# Pre-rename env: same requirements.txt, so use it rather than rebuilding.
if "$CONDA" env list 2>/dev/null | grep -q "^$CODELAB_LEGACY_ENV[[:space:]]"; then
    echo "[launcher] \"$CODELAB_ENV\" not found; using the pre-rename env \"$CODELAB_LEGACY_ENV\"."
    CODELAB_ENV=$CODELAB_LEGACY_ENV
    run_in_env
fi

echo "[launcher] conda env \"$CODELAB_ENV\" does not exist yet."
case "$CODELAB_BOOTSTRAP" in
    never|NEVER) base_fallback ;;
    yes|YES)     bootstrap ;;
esac
echo "[launcher] It can be created now from requirements.txt. That downloads"
echo "[launcher] a few GB (CUDA torch) and takes several minutes, once."
echo "[launcher] Set CODELAB_BOOTSTRAP=yes to skip this prompt, or =never to"
echo "[launcher] stop being asked at all."
printf 'Create it now? [y/N] '
read ANSWER
case "$ANSWER" in
    y|Y|yes|YES) bootstrap ;;
    *)           base_fallback ;;
esac
