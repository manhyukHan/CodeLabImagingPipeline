"""
THE celltype -> colour rule, shared by every view that draws celltypes.

It exists because there were two of them. The Barcode Channel Overview
coloured by the order its channel dict happened to be built in (the
order celltypes were assigned), while the Celltype Determination Result
coloured by sorted name -- same palette, different index. With WT, 4A3
and 8B1 assigned in that order, WT drew red in one window and green in
the other, side by side on screen, which makes the two views impossible
to read against each other.

Colour is keyed on the CELLTYPE NAME and the order is always sorted, so
it depends on nothing but the set of names present: assignment order,
dict insertion order and which window drew first cannot change it.
"""

# Distinct categorical colours -- celltypes are unordered categories, not
# a scale, so a categorical palette is right here (unlike the alignment
# all-readouts overlay in canvas/pipeline_canvas.py, which deliberately
# uses a sequential red-cyan gradient for a before/after comparison of
# ONE transform).
CATEGORICAL_COLORS = [
    (0.90, 0.10, 0.10), (0.10, 0.65, 0.90), (0.15, 0.80, 0.15), (0.95, 0.75, 0.10),
    (0.70, 0.20, 0.90), (0.95, 0.45, 0.10), (0.10, 0.85, 0.75), (0.85, 0.10, 0.55),
]

UNCLASSIFIED_COLOR = (0.55, 0.55, 0.55)


def colors_for_names(names):
    """
    {celltype name: (r, g, b)} for a set/iterable of names.

    Sorted, so the same name always lands on the same colour for a given
    set of celltypes no matter who asks or in what order.
    """
    ordered = sorted({str(n) for n in names if n})
    return {name: CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
            for i, name in enumerate(ordered)}


def hex_of(color):
    """'#rrggbb' for a 0..1 RGB triple -- for the HTML legends."""
    return '#%02x%02x%02x' % tuple(int(c * 255) for c in color)
