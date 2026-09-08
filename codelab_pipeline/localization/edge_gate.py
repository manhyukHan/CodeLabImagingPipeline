"""
A spot can be real and still sit somewhere its position cannot be believed.

THIS IS NOT A MODEL INPUT, and keeping it out of one is the whole point.
"Is this a real spot?" is what the classifier answers. "Now that we know
where it is, do we keep it?" is a different question asked at a different
time -- after pass/fail, after the matched filter has placed it to
sub-voxel precision, with the final position in hand.

Handing distance-to-the-end-of-the-stack to the classifier makes it
learn "spots near the end of a stack are less likely to be real", which
is false and biases the model against exactly the real spots it should
be finding there. So the number is computed here, where it gates an
answer, and nowhere near features.NAMES.

IT IS ALSO NOT z_status. That field records whether a spot's Z has been
FITTED -- accepted, rejected, or not_fit -- and is where a learned
pass/fail verdict belongs. A spot denied here has a perfectly good fit;
what it lacks is a stack that extends far enough either side of it for
the fit to mean anything.

WHY 11 PLANES. Inter-round drift in this pipeline runs to 11 planes
(CLAUDE.md, the cell-alignment Z bound that sat at 5 and was wrong). An
emitter within 11 planes of an end may therefore have been outside the
imaged range in some other round, so its axial position is not
comparable across rounds even when this round's fit is clean.

MEASURED on 2,057 human-gated MP58/RNA candidates in 105-plane stacks,
by distance from the nearer end:

     0-5  planes      4 candidates    0.0% confirmed
     5-11 planes     29               3.4%
    11-20 planes     46              39.1%
    20+  planes   1,978              57.5%

so the default denies 33 of 2,057 and costs ONE confirmed spot. The
reviewer was already refusing almost everything down there; this makes
the refusal explicit and applies it to the position rather than to the
question of whether the emitter exists.
"""

DEFAULT_MARGIN_PLANES = 11


def planes_from_end(z, depth):
    """How many planes separate `z` from the nearer end of the stack.

    Returns 0.0 at either face. Fractional z is kept -- the matched
    filter reports sub-voxel positions and rounding them here would
    quietly move the boundary by half a plane.
    """
    z = float(z)
    d = int(depth)
    if d <= 0:
        return 0.0
    return float(min(max(z, 0.0), max(d - 1 - z, 0.0)))


def deny(z, depth, margin_planes=DEFAULT_MARGIN_PLANES):
    """True when this spot's axial position should not be believed.

    Say `deny`, not `reject`: the spot may well be real, and a caller
    that wants only its (y, x) is entitled to keep it. What this refuses
    is the z.
    """
    return planes_from_end(z, depth) < float(margin_planes)


def annotate(spots, depth, margin_planes=DEFAULT_MARGIN_PLANES):
    """[(spot, planes_from_end, denied), ...] for a list of LocalizedSpot.

    Returns the number alongside the verdict so a caller can log or plot
    what the gate did rather than only what it removed.
    """
    out = []
    for s in spots:
        d = planes_from_end(s.z, depth)
        out.append((s, d, d < float(margin_planes)))
    return out
