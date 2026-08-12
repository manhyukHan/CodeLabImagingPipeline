"""
Shared scroll-to-zoom + drag-to-pan + view-persistence-across-redraw for
every interactive displayer in this app.

Matplotlib gives no free zoom/pan by default -- a plain FigureCanvasQTAgg
only resizes with its window (the image content scales to fill, but
there's no way to zoom into a sub-region). The standard fix,
NavigationToolbar2QT, was deliberately NOT used here: its pan/zoom tools
capture LEFT-click-drag, which would fight with this app's manual-mode
click-to-place interactions (cell polygon vertices, spot placement:
left-click=add, right-click=remove) -- exactly the conflict CellClassifier
needed a dedicated "F" pan/zoom-toggle shortcut to work around with
pyqtgraph. Scroll-wheel zoom (this module) needs no click at all, so it
never competes with those handlers; drag-to-pan (install_drag_pan, per
explicit request) uses the MIDDLE mouse button specifically, since
neither manual-click handler (left/right) ever touches it -- pan works
in every mode, manual or not, no toggle needed.

Every displayer's _redraw() does fig.clear() + fig.subplots(...) on every
call (new mask overlay, new scale, a moved vertex, ...), which would
otherwise silently reset any zoom the user had set back to full extent on
every single interaction -- install_scroll_zoom alone isn't enough,
capture_view/restore_view must be threaded through each displayer's own
_redraw so genuinely cosmetic updates (scale change, remove one cell ID,
place one spot) preserve the current zoom, while a real set_data() call
(new FOV, new crop, a different image entirely) resets to full extent.
"""


def _apply_zoom(ax, canvas, cx, cy, scale):
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    new_width = (cur_xlim[1] - cur_xlim[0]) * scale
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale
    relx = (cur_xlim[1] - cx) / (cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - cy) / (cur_ylim[1] - cur_ylim[0])
    ax.set_xlim([cx - new_width * (1 - relx), cx + new_width * relx])
    ax.set_ylim([cy - new_height * (1 - rely), cy + new_height * rely])
    canvas.draw_idle()


def install_scroll_zoom(canvas, zoom_factor=1.2):
    """
    macOS trackpads report scroll as pixel-precise deltas (QWheelEvent.
    pixelDelta), not the fixed +-1 "one click" a real mouse wheel sends --
    matplotlib's Qt backend passes that raw magnitude through as
    event.step (see FigureCanvasQT.wheelEvent), which the earlier version
    of this function completely ignored, applying the same fixed
    zoom_factor regardless of whether the gesture was one full click or a
    single barely-perceptible trackpad tick. A light trackpad touch could
    produce a step of e.g. 0.3, which still triggered a full 20% zoom --
    on some trackpads/OS scroll-speed settings this reportedly reads as
    "nothing happens" (misfired direction, or too fast/small to notice
    correctly), so the effective zoom-per-event is now scaled by
    min(abs(step), 10) (clamped so one big scroll-wheel or fast swipe
    can't jump too far in one callback) instead of being all-or-nothing.
    """
    def on_scroll(event):
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        magnitude = min(abs(event.step) if event.step else 1.0, 10.0)
        scale = zoom_factor ** (-magnitude if event.step > 0 else magnitude)
        _apply_zoom(ax, canvas, event.xdata, event.ydata, scale)
    return canvas.mpl_connect('scroll_event', on_scroll)


def install_keyboard_zoom(canvas, zoom_factor=1.2):
    """
    A second, platform-independent way to zoom (+/- or =/-) that doesn't
    depend on how a given trackpad/mouse/OS combination translates scroll
    gestures into Qt wheel events -- if scroll ever misbehaves on some
    setup, this still works. Zooms whichever axes the mouse last hovered
    over (event.inaxes, same as scroll); falls back to the first axes if
    the mouse hasn't been over any (e.g. keys pressed right after the
    window opens). Requires the canvas to have keyboard focus (already
    true after any click on it, via ClickFocus).
    """
    def on_key(event):
        if event.key not in ('+', '=', '-', '_'):
            return
        ax = event.inaxes
        if ax is None:
            axes = canvas.figure.axes
            if not axes:
                return
            ax = axes[0]
        cx = sum(ax.get_xlim()) / 2
        cy = sum(ax.get_ylim()) / 2
        scale = (1 / zoom_factor) if event.key in ('+', '=') else zoom_factor
        _apply_zoom(ax, canvas, cx, cy, scale)
    return canvas.mpl_connect('key_press_event', on_key)


def install_drag_pan(canvas, button=2):
    """
    Middle-click-drag (button=2 -- Qt/matplotlib's MouseButton.MIDDLE) to
    pan, grabbing whatever data point was under the cursor at press-time
    and keeping it under the cursor for the rest of the drag. Deliberately
    NOT left or right click (see module docstring): those are already
    manual-mode's add/remove buttons in every displayer this installs
    into, and using either here would make every manual click also start
    a pan.

    Converts the drag in PIXEL space (event.x/event.y -- stable widget
    coordinates, unaffected by the axes' own xlim/ylim) rather than DATA
    space (event.xdata/ydata): those are computed through the axes'
    CURRENT transform, which this function is itself changing on every
    motion event, so re-deriving a delta from consecutive event.xdata
    values would compound whatever rounding/transform error each step
    introduces. Instead, the pixel-per-data-unit conversion is captured
    ONCE at press-time and the axes' own press-time xlim/ylim are always
    the base a fresh total pixel offset is applied to -- no compounding
    possible.
    """
    state = {}

    def on_press(event):
        if event.button != button or event.inaxes is None:
            return
        ax = event.inaxes
        inv = ax.transData.inverted()
        x0d, y0d = inv.transform((event.x, event.y))
        x1d, y1d = inv.transform((event.x + 1, event.y + 1))
        state['ax'] = ax
        state['x0'], state['y0'] = event.x, event.y
        state['xlim0'], state['ylim0'] = ax.get_xlim(), ax.get_ylim()
        state['dx_per_px'], state['dy_per_px'] = x1d - x0d, y1d - y0d

    def on_motion(event):
        if state.get('ax') is None or event.inaxes is not state['ax']:
            return
        dx = (event.x - state['x0']) * state['dx_per_px']
        dy = (event.y - state['y0']) * state['dy_per_px']
        x0, x1 = state['xlim0']
        y0, y1 = state['ylim0']
        state['ax'].set_xlim(x0 - dx, x1 - dx)
        state['ax'].set_ylim(y0 - dy, y1 - dy)
        canvas.draw_idle()

    def on_release(event):
        if event.button == button:
            state.clear()

    return [canvas.mpl_connect('button_press_event', on_press),
            canvas.mpl_connect('motion_notify_event', on_motion),
            canvas.mpl_connect('button_release_event', on_release)]


def capture_view(fig):
    """[(xlim, ylim), ...] for every current axes, in order -- call BEFORE fig.clear()."""
    return [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes]


def restore_view(fig, saved_view):
    """Reapply a capture_view() result to the freshly-rebuilt axes -- call AFTER fig.subplots(...), before draw()."""
    axes = fig.axes
    if saved_view is None or len(axes) != len(saved_view):
        return  # axes count changed (e.g. first-ever draw) -- nothing sane to restore, keep the full-extent default
    for ax, (xlim, ylim) in zip(axes, saved_view):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


def reset_view(canvas):
    for ax in canvas.figure.axes:
        ax.autoscale()
    canvas.draw_idle()
