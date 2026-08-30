# Sphinx configuration for the CodeLabImagingPipeline documentation.
# Built by Read the Docs from .readthedocs.yaml at the repo root; the
# doc-build dependencies live in docs/requirements.txt (NOT the app's
# requirements.txt -- the app env never needs sphinx).

project = 'CodeLabImagingPipeline'
author = 'CODE Lab'
copyright = '2026, CODE Lab'

extensions = [
    'myst_parser',
]

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

source_suffix = {'.md': 'markdown', '.rst': 'restructuredtext'}
master_doc = 'index'

exclude_patterns = ['_build', 'perf']

html_theme = 'furo'
html_static_path = ['_static']
html_title = 'CodeLabImagingPipeline'
