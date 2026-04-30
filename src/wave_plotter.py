"""
wave_plotter.py
Reusable 2D scalar wavefield plotting utilities
by thomas martinod
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ---------------------------------------------------------
def enable_latex_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsfonts}"
    })


# ---------------------------------------------------------
def format_power_notation(value):
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    base = value / 10**exponent
    if np.isclose(base, 1):
        return rf"10^{{{exponent}}}"
    else:
        return rf"{base:.2f} \times 10^{{{exponent}}}"


# ---------------------------------------------------------
def plot_scalar_field(
    X,
    Y,
    Z,
    *,
    cmap="viridis",
    title="",
    xlabel=r"$\hat{x}$",
    ylabel=r"$\hat{y}$",
    cbar_label="",
    filename=None,
    save_folder="figs",
    show=True,
    equal_aspect=True,
    hide_spines=True,
    center_zero=False,
    symmetric_limits=False,
    vmin=None,
    vmax=None,
    vmax_percentile=100.0,
    dpi=300,
    cylinder_radius=None,
    param_text=None,
):
    """
    Generic 2D scalar field plotter.

    Parameters
    ----------
    vmax_percentile : float, default 100.0
        Clips the color scale at this percentile of the data.
        For center_zero=True and symmetric_limits=True, the clipping is
        symmetric: vmax = percentile(|data|, p), vmin = -vmax.

    cylinder_radius : float or None
        If given, draws a thin white circle at this radius to mark the
        cylinder boundary.

    param_text : str or None
        If given, shows a small textbox in the bottom-left corner.
    """

    fig, ax = plt.subplots(figsize=(7, 6))

    # ---------------------------------------------------------
    # color normalization with percentile clipping
    # ---------------------------------------------------------
    norm = None
    _vmin, _vmax = vmin, vmax

    data = np.ma.array(Z).compressed()

    if data.size == 0:
        raise ValueError("Z contains no valid data to plot.")

    if vmin is None or vmax is None:
        if center_zero and symmetric_limits:
            m = float(np.nanpercentile(np.abs(data), vmax_percentile))
            _vmin, _vmax = -m, m
        else:
            if _vmin is None:
                _vmin = float(np.nanpercentile(data, 100.0 - vmax_percentile))
            if _vmax is None:
                _vmax = float(np.nanpercentile(data, vmax_percentile))

    if center_zero:
        norm = TwoSlopeNorm(vmin=_vmin, vcenter=0.0, vmax=_vmax)

    # ---------------------------------------------------------
    # scalar field
    # ---------------------------------------------------------
    pcm = ax.pcolormesh(
        X, Y, Z,
        shading="auto",
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else _vmin,
        vmax=None if norm is not None else _vmax,
    )

    # ---------------------------------------------------------
    # optional boundary circle
    # ---------------------------------------------------------
    if cylinder_radius is not None:
        theta_c = np.linspace(0, 2 * np.pi, 400)
        ax.plot(
            cylinder_radius * np.cos(theta_c),
            cylinder_radius * np.sin(theta_c),
            color="white",
            lw=0.8,
            zorder=10,
        )

    # ---------------------------------------------------------
    # optional parameter textbox
    # ---------------------------------------------------------
    if param_text is not None:
        ax.text(
            0.03, 0.03, param_text,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
            ),
        )

    # ---------------------------------------------------------
    # labels
    # ---------------------------------------------------------
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    if equal_aspect:
        ax.set_aspect("equal")

    if hide_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ---------------------------------------------------------
    # ticks at ±(largest multiple of 5 inside domain)
    # ---------------------------------------------------------
    def _floor5(v):
        val = int(5 * np.floor(v / 5))
        return val if val > 0 else int(np.floor(v))

    xt = _floor5(float(np.max(np.abs(X))))
    yt = _floor5(float(np.max(np.abs(Y))))

    ax.set_xticks([-xt, xt])
    ax.set_yticks([-yt, yt])
    ax.set_xticklabels([rf"$-{xt}$", rf"${xt}$"], fontsize=9)
    ax.set_yticklabels([rf"$-{yt}$", rf"${yt}$"], fontsize=9)

    # ---------------------------------------------------------
    # colorbar
    # ---------------------------------------------------------
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9)

    # ---------------------------------------------------------
    # save/show
    # ---------------------------------------------------------
    if filename is not None:
        os.makedirs(save_folder, exist_ok=True)
        fig.savefig(
            os.path.join(save_folder, f"{filename}.png"),
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=dpi,
        )

    if show:
        plt.show()
    else:
        plt.close()

# ---------------------------------------------------------
def plot_field_grid(
    X,
    Y,
    Z_list,
    *,
    cmap="viridis",
    xlabel=r"$\hat{x}$",
    ylabel=r"$\hat{y}$",
    cbar_label="",
    suptitle=None,
    row_labels=None,
    col_labels=None,
    filename=None,
    save_folder="figs",
    show=True,
    figsize=(11, 10.5),
    equal_aspect=True,
    hide_spines=True,
    center_zero=False,
    symmetric_limits=False,
    common_colorbar=True,
    vmin=None,
    vmax=None,
    vmax_percentile=100.0,
    tick_value=None,
    gap_in=0.16,
    cbar_gap_in=0.22,
    cbar_width_in=0.22,
    cbar_height_ratio=0.50,
    left_margin_in=0.70,
    right_margin_in=0.25,
    bottom_margin_in=0.60,
    top_margin_in=0.45,
    suptitle_offset_in=0.18,
    row_label_offset_in=0.26,
    col_label_offset_in=0.28,
    x_label_offset_in=0.08,
    y_label_offset_in=0.08,
    suptitle_fontsize=15,
    row_label_fontsize=11,
    col_label_fontsize=11,
    tick_fontsize=9,
    axis_label_fontsize=11,
    dpi=300,
):
    """
    plot a grid of scalar fields with a shared colorbar.

    parameters
    ----------
    vmax_percentile : float (default 100.0)
        Clip the shared color scale at this percentile of the data.
        Use e.g. 99.0 to prevent one dominant panel from washing out
        the rest. For center_zero+symmetric_limits fields the clipping
        is symmetric: vmax = percentile(|data|, p), vmin = -vmax.
        Only applied when vmin/vmax are not set manually.

    notes
    -----
    - horizontal and vertical gaps are equal in physical units (inches)
    - y ticks and labels appear only on the left column
    - x ticks and labels appear only on the bottom row
    - row labels are placed on the left
    - column labels are placed below the bottom row
    - global x/y labels are placed closer to the plots
    - the colorbar is placed in a dedicated axis
    - saved figures are cropped tightly to the content
    """

    # ---------------------------------------------------------
    # infer grid shape
    n_plots = len(Z_list)

    if row_labels is not None and col_labels is not None:
        nrows = len(row_labels)
        ncols = len(col_labels)
    else:
        ncols = int(np.ceil(np.sqrt(n_plots)))
        nrows = int(np.ceil(n_plots / ncols))

    if nrows * ncols != n_plots:
        raise ValueError(
            f"grid shape mismatch: got {n_plots} fields, "
            f"but {nrows} x {ncols} = {nrows*ncols} slots"
        )

    # ---------------------------------------------------------
    fig_w_in, fig_h_in = figsize
    fig = plt.figure(figsize=figsize)

    # ---------------------------------------------------------
    # global color limits with percentile clipping
    if vmin is None or vmax is None:
        Z_all = np.ma.concatenate([np.ma.array(Zi).ravel() for Zi in Z_list])
        data  = Z_all.compressed()

        if center_zero and symmetric_limits:
            m    = float(np.nanpercentile(np.abs(data), vmax_percentile))
            vmin = -m
            vmax =  m
        else:
            if vmin is None:
                vmin = float(np.nanpercentile(data, 100.0 - vmax_percentile))
            if vmax is None:
                vmax = float(np.nanpercentile(data, vmax_percentile))

    norm = None
    if center_zero:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # ---------------------------------------------------------
    def floor_multiple_of_5(val):
        tick = int(5 * np.floor(val / 5))
        if tick <= 0:
            tick = int(np.floor(val))
        return tick

    xmax = float(np.max(np.abs(X)))
    ymax = float(np.max(np.abs(Y)))

    if tick_value is None:
        xt = floor_multiple_of_5(xmax)
        yt = floor_multiple_of_5(ymax)
    else:
        if np.isscalar(tick_value):
            xt = tick_value
            yt = tick_value
        else:
            xt, yt = tick_value

    xticks = [-xt, xt]
    yticks = [-yt, yt]

    # ---------------------------------------------------------
    reserved_right_in = right_margin_in
    if common_colorbar:
        reserved_right_in += cbar_gap_in + cbar_width_in

    avail_w_in = fig_w_in - left_margin_in - reserved_right_in
    avail_h_in = fig_h_in - bottom_margin_in - top_margin_in

    if equal_aspect:
        panel_side_in = min(
            (avail_w_in - (ncols - 1) * gap_in) / ncols,
            (avail_h_in - (nrows - 1) * gap_in) / nrows
        )
        panel_w_in = panel_side_in
        panel_h_in = panel_side_in
    else:
        panel_w_in = (avail_w_in - (ncols - 1) * gap_in) / ncols
        panel_h_in = (avail_h_in - (nrows - 1) * gap_in) / nrows

    grid_w_in      = ncols * panel_w_in + (ncols - 1) * gap_in
    grid_h_in      = nrows * panel_h_in + (nrows - 1) * gap_in
    grid_left_in   = left_margin_in + 0.5 * (avail_w_in - grid_w_in)
    grid_bottom_in = bottom_margin_in + 0.5 * (avail_h_in - grid_h_in)
    grid_right_in  = grid_left_in + grid_w_in
    grid_top_in    = grid_bottom_in + grid_h_in

    # ---------------------------------------------------------
    axes = np.empty((nrows, ncols), dtype=object)
    pcm  = None

    for i in range(nrows):
        for j in range(ncols):
            x0_in = grid_left_in + j * (panel_w_in + gap_in)
            y0_in = grid_bottom_in + (nrows - 1 - i) * (panel_h_in + gap_in)

            ax = fig.add_axes([
                x0_in / fig_w_in,
                y0_in / fig_h_in,
                panel_w_in / fig_w_in,
                panel_h_in / fig_h_in,
            ])

            axes[i, j] = ax
            Z = Z_list[i * ncols + j]

            pcm = ax.pcolormesh(
                X, Y, Z,
                shading="auto", cmap=cmap, norm=norm,
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
            )

            if equal_aspect:
                ax.set_aspect("equal")
            if hide_spines:
                for spine in ax.spines.values():
                    spine.set_visible(False)

            ax.set_xlim(float(np.min(X)), float(np.max(X)))
            ax.set_ylim(float(np.min(Y)), float(np.max(Y)))

            is_left   = (j == 0)
            is_bottom = (i == nrows - 1)

            ax.set_xticks(xticks)
            if is_bottom:
                ax.set_xticklabels([rf"$-{xt}$", rf"${xt}$"],
                                    fontsize=tick_fontsize)
                ax.tick_params(axis="x", which="both", bottom=True,
                               labelbottom=True, top=False, labeltop=False,
                               direction="out", length=3, width=0.8,
                               pad=1, labelsize=tick_fontsize)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", which="both", bottom=False,
                               labelbottom=False, top=False,
                               labeltop=False, length=0)

            ax.set_yticks(yticks)
            if is_left:
                ax.set_yticklabels([rf"$-{yt}$", rf"${yt}$"],
                                    fontsize=tick_fontsize)
                ax.tick_params(axis="y", which="both", left=True,
                               labelleft=True, right=False, labelright=False,
                               direction="out", length=3, width=0.8,
                               pad=1, labelsize=tick_fontsize)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", which="both", left=False,
                               labelleft=False, right=False,
                               labelright=False, length=0)

    # ---------------------------------------------------------
    if row_labels is not None:
        for i, label in enumerate(row_labels):
            y_center_in = (
                grid_bottom_in
                + (nrows - 1 - i) * (panel_h_in + gap_in)
                + 0.5 * panel_h_in
            )
            fig.text(
                (grid_left_in - row_label_offset_in) / fig_w_in,
                y_center_in / fig_h_in,
                label, ha="right", va="center",
                rotation=90, fontsize=row_label_fontsize,
            )

    if col_labels is not None:
        for j, label in enumerate(col_labels):
            x_center_in = (
                grid_left_in
                + j * (panel_w_in + gap_in)
                + 0.5 * panel_w_in
            )
            fig.text(
                x_center_in / fig_w_in,
                (grid_bottom_in - col_label_offset_in) / fig_h_in,
                label, ha="center", va="top", fontsize=col_label_fontsize,
            )

    # ---------------------------------------------------------
    mid_col     = ncols // 2
    x_center_in = (
        grid_left_in + mid_col * (panel_w_in + gap_in) + 0.5 * panel_w_in
    )
    fig.text(
        x_center_in / fig_w_in,
        (grid_bottom_in - x_label_offset_in) / fig_h_in,
        xlabel, ha="center", va="top", fontsize=axis_label_fontsize,
    )

    mid_row     = nrows // 2
    y_center_in = (
        grid_bottom_in
        + (nrows - 1 - mid_row) * (panel_h_in + gap_in)
        + 0.5 * panel_h_in
    )
    fig.text(
        (grid_left_in - y_label_offset_in) / fig_w_in,
        y_center_in / fig_h_in,
        ylabel, ha="right", va="center",
        rotation=90, fontsize=axis_label_fontsize,
    )

    # ---------------------------------------------------------
    if suptitle is not None:
        fig.text(
            0.5,
            (grid_top_in + suptitle_offset_in) / fig_h_in,
            suptitle, ha="center", va="bottom", fontsize=suptitle_fontsize,
        )

    # ---------------------------------------------------------
    if common_colorbar and pcm is not None:
        cbar_h_in = cbar_height_ratio * grid_h_in
        cbar_y_in = grid_bottom_in + 0.5 * (grid_h_in - cbar_h_in)
        cbar_x_in = grid_right_in + cbar_gap_in

        cax = fig.add_axes([
            cbar_x_in / fig_w_in,
            cbar_y_in / fig_h_in,
            cbar_width_in / fig_w_in,
            cbar_h_in / fig_h_in,
        ])

        cbar = fig.colorbar(pcm, cax=cax)
        cbar.set_label(cbar_label, fontsize=axis_label_fontsize)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=tick_fontsize)

    # ---------------------------------------------------------
    if filename is not None:
        os.makedirs(save_folder, exist_ok=True)
        fig.savefig(
            os.path.join(save_folder, f"{filename}.png"),
            dpi=dpi, bbox_inches="tight", pad_inches=0.02,
        )

    if show:
        plt.show()
    else:
        plt.close()