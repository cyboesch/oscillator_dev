"""Compact journal layout for parameter schedules and a matching real trajectory.

Rendering only: all parameters and trajectory frames are verified by the
extractor. Positive and signed parameter rows share one blue-white-red palette,
with separate numerical normalizations and common limits across times/states.
"""

from pathlib import Path

import numpy as np


def render_expanded(data, metadata, output, spring_view="rms", colour_scale="linear",
                    output_stem="parameter_evolution_draft"):
    """Render a six-row main figure and additional fixed-state maps at 180 mm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm

    output.mkdir(parents=True, exist_ok=True)
    data_dir = output.parent / "figure_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if colour_scale not in ("linear", "log"):
        raise ValueError("colour_scale must be 'linear' or 'log'")
    if spring_view not in ("rms", "graph"):
        raise ValueError("spring_view must be 'rms' or 'graph'")
    if not output_stem or "/" in output_stem or "\\" in output_stem:
        raise ValueError("output_stem must be a simple file stem")

    # Bundled OFL font, metrically compatible with Arial; no silent substitution.
    font_dir = Path(__file__).resolve().parent / "fonts"
    for variant in ("Regular", "Bold", "Italic", "BoldItalic"):
        font_manager.fontManager.addfont(str(font_dir / f"LiberationSans-{variant}.ttf"))
    font_manager.findfont("Liberation Sans", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1,
        "xtick.minor.width": 1, "ytick.minor.width": 1,
        "patch.linewidth": 1, "lines.linewidth": 1.2,
        "axes.grid": False, "savefig.bbox": None,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    ink = "#202B38"
    ids = np.asarray(data["slices"], dtype=int)
    states = np.asarray(data["states"], dtype=int)
    grid = int(data["grid"])
    if len(ids) != 5 or not len(states):
        raise ValueError("The layout needs five times and at least one state")
    if (data["W_states"].shape != data["Q_states"].shape
            or data["W_states"].shape[1:] != (len(states), grid * grid)):
        raise ValueError("State arrays do not match state IDs and image geometry")
    images = np.asarray(data["trajectory_images"])
    if images.shape != (5, grid, grid) or not np.isfinite(images).all():
        raise ValueError("Five verified real image frames are required")
    if not np.allclose(data["trajectory_times"], data["generation_time"][ids], rtol=0, atol=1e-8):
        raise ValueError("Image and parameter times must match exactly")

    positive_duff = bool(np.min(data["k_duff"][ids]) >= 0)
    selected_arrays = {key: data[key][ids] for key in ("c_lin", "c_rms_map", "k_lin", "k_duff")}
    selected_arrays.update({key: data[key + "_states"][ids] for key in ("W", "Q")})
    limits = {key: max(float(np.max(np.abs(values))), 1e-15)
              for key, values in selected_arrays.items()}
    normalizations, norm_metadata = {}, {}
    for key, values in selected_arrays.items():
        positive = key == "c_rms_map" or (key == "k_duff" and positive_duff)
        if colour_scale == "log" and positive:
            strictly_positive = np.asarray(values)[np.asarray(values) > 0]
            if strictly_positive.size:
                lower, upper = float(strictly_positive.min()), limits[key]
                if lower == upper:
                    lower = upper / 10.0
                normalizations[key] = LogNorm(vmin=lower, vmax=upper)
                norm_metadata[key] = {"type": "log", "vmin": lower, "vmax": upper,
                                      "nonpositive_values": "masked grey", "base": 10}
            else:
                normalizations[key] = Normalize(0, 1)
                norm_metadata[key] = {"type": "linear_zero_fallback", "vmin": 0., "vmax": 1.}
        elif colour_scale == "log":
            threshold = .01 * limits[key]
            normalizations[key] = SymLogNorm(linthresh=threshold, linscale=1, base=10,
                                             vmin=-limits[key], vmax=limits[key])
            norm_metadata[key] = {"type": "symmetric_log", "vmin": -limits[key], "vmax": limits[key],
                                  "linthresh": threshold, "linscale": 1, "base": 10,
                                  "threshold_rule": "1% of maximum absolute displayed coefficient in this family"}
        else:
            lower = 0.0 if positive else -limits[key]
            normalizations[key] = Normalize(lower, limits[key])
            norm_metadata[key] = {"type": "linear", "vmin": lower, "vmax": limits[key]}

    metadata.update({
        "colour_scale": colour_scale, "colour_normalizations": norm_metadata,
        "colour_limits_symmetric": {key: limits[key] for key in ("c_lin", "k_lin", "W", "Q")},
        "spring_view": spring_view,
        "spring_map_colour_range": [normalizations["c_rms_map"].vmin, normalizations["c_rms_map"].vmax],
        "duffing_colour_range": [normalizations["k_duff"].vmin, normalizations["k_duff"].vmax],
        "hidden_colour_scaling": "One W scale and one Q scale across all displayed states, times and pages.",
        "parameter_colormap": "RdBu_r for every parameter family, including positive-only rows",
        "colour_interpretation": "Blue to red means low to high within each row's numerical range; blue is not a universal negative-value marker. White is zero for symmetric signed rows, but the geometric midpoint for positive log rows.",
        "edge_width_points": {"formula": "1.0 + 2.0 * abs(c_lin) / max_abs_c_lin", "common_across_frames": True},
        "artwork": {"width_mm": 180, "font_family": "Liberation Sans", "font_size_pt": 8,
                    "minimum_stroke_pt": 1, "displayed_time": "forward time t only; decreases left to right",
                    "title": None, "figure_notes": "In caption and provenance, not on the artwork"},
    })
    if not positive_duff:
        metadata["colour_limits_symmetric"]["k_duff"] = limits["k_duff"]

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D7DBE0")
    width, map_size, pitch, top_margin = 180., 24., 29., 10.
    column_lefts = 29.0 + np.arange(5) * 26.5

    def tick_label(value):
        # Compact numeric labels keep all colour-bar text at 8 pt.
        if value == 0:
            return "0"
        if abs(value) < .001 or abs(value) >= 1000:
            mantissa, exponent = f"{value:.1e}".split("e")
            return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}".replace("-", "\N{MINUS SIGN}")
        return f"{value:.2g}".replace("-", "\N{MINUS SIGN}")

    def make_page(rows, svg_stem, page_number):
        height = top_margin + (len(rows) - 1) * pitch + map_size + 4
        fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor="white")

        def text_at(x, down, text, **kwargs):
            return fig.text(x / width, 1 - down / height, text,
                            fontsize=8, color=ink, **kwargs)

        def axes_at(x, down, w, h):
            return fig.add_axes([x / width, 1 - (down + h) / height, w / width, h / height])

        for left, k in zip(column_lefts, ids):
            value = float(data["forward_time"][k])
            label = f"{value:.2f}".rstrip("0").rstrip(".")
            text_at(left + map_size / 2, 4.6, f"t = {label}", ha="center", va="center")

        for row_index, (key, label, state_position) in enumerate(rows):
            top = top_margin + row_index * pitch
            text_at(2.4, top + .3, f"{chr(ord('a') + row_index)})", weight="bold", va="top")
            text_at(8, top + map_size / 2, label, va="center", linespacing=1.2)
            for column, (left, k) in enumerate(zip(column_lefts, ids)):
                ax = axes_at(left, top, map_size, map_size)
                if key == "image":
                    ax.imshow(images[column], cmap="gray", vmin=0, vmax=1, interpolation="none")
                elif key == "c_lin":
                    nodes = data["patch_nodes"]
                    positions = {int(node): (int(node) % grid, -(int(node) // grid)) for node in nodes}
                    node_xy = np.asarray([positions[int(i)] for i in nodes])
                    values = data[key][k]
                    segments = [[positions[int(a)], positions[int(b)]] for a, b in data["patch_edges"]]
                    ax.add_collection(LineCollection(segments, colors=cmap(normalizations[key](values)),
                                                      linewidths=1 + 2 * np.abs(values) / limits[key], zorder=1))
                    ax.scatter(*node_xy.T, s=14, color="white", edgecolor=ink, linewidth=1, zorder=3)
                    ax.set_xlim(node_xy[:, 0].min() - .45, node_xy[:, 0].max() + .45)
                    ax.set_ylim(node_xy[:, 1].min() - .45, node_xy[:, 1].max() + .45)
                    ax.set_aspect("equal")
                else:
                    values = data[key + "_states"][k, state_position] if state_position is not None else data[key][k]
                    if norm_metadata[key]["type"] == "log":
                        values = np.ma.masked_less_equal(values, 0)
                    ax.imshow(values.reshape(grid, grid), cmap=cmap, norm=normalizations[key], interpolation="none")
                ax.axis("off")

            if key == "image":
                continue
            norm, info = normalizations[key], norm_metadata[key]
            cax = axes_at(162.3, top + 1.8, 2.2, map_size - 3.6)
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            if info["type"] == "log":
                ticks = [info["vmin"], np.sqrt(info["vmin"] * info["vmax"]), info["vmax"]]
            elif info["type"] == "symmetric_log":
                ticks = [-limits[key], -info["linthresh"], 0, info["linthresh"], limits[key]]
            elif info["type"] == "linear_zero_fallback":
                ticks = [0, .5, 1]
            else:
                ticks = [norm.vmin, (norm.vmin + norm.vmax) / 2, norm.vmax]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([tick_label(v) for v in ticks])
            cbar.ax.tick_params(labelsize=8, length=2, width=1, pad=2)
            cbar.ax.minorticks_off()
            cbar.outline.set_linewidth(1)
        # Keep exact canvas dimensions; no automatic tight bounding box.
        svg = data_dir / f"{svg_stem}.svg"
        fig.savefig(svg, dpi=450, facecolor="white")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        return fig, height

    spring_key = "c_rms_map" if spring_view == "rms" else "c_lin"
    first = int(states[0])
    page_one_rows = [
        (spring_key, "Springs\nRMS" if spring_view == "rms" else "Spring\ncouplings", None),
        ("k_lin", "Linear\nstiffness", None),
        ("k_duff", "Quartic\ncoefficient", None),
        ("W", f"Linear\ncoupling\nState {first}", 0),
        ("Q", f"Quadratic\ncoupling\nState {first}", 0),
        ("image", "Generated\nimage", None),
    ]
    page_two_rows = []
    for position, state in enumerate(states[1:], start=1):
        for key, label in (("W", "Linear"), ("Q", "Quadratic")):
            page_two_rows.append((key, f"{label}\ncoupling\nState {int(state)}", position))
    states_stem = "parameter_evolution_states" if output_stem == "parameter_evolution_draft" else output_stem + "_states"
    first_fig, first_height = make_page(page_one_rows, output_stem, 1)
    figures = [first_fig]
    metadata["figure_pages"] = [{
        "page": 1, "rows": ["spring", "k_lin", "k_duff", f"W_{first}", f"Q_{first}", "image"],
        "svg_file": f"{output_stem}.svg", "size_mm": [width, first_height],
    }]
    if page_two_rows:
        second_fig, second_height = make_page(page_two_rows, states_stem, 2)
        figures.append(second_fig)
        metadata["figure_pages"].append({
            "page": 2, "state_ids": states[1:].tolist(), "svg_file": f"{states_stem}.svg",
            "size_mm": [width, second_height],
        })
    # Catch label overflow at actual publication size before writing the PDF.
    for fig in figures:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in fig.findobj(matplotlib.text.Text):
            if not text.get_visible() or not text.get_text() or (text.axes is not None and not text.axes.axison):
                continue
            box = text.get_window_extent(renderer)
            if box.width > 0 and box.height > 0 and (
                    box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
                raise ValueError(f"Text outside figure canvas: {text.get_text()!r}")
    with PdfPages(output / f"{output_stem}.pdf") as pdf:
        pdf.infodict()["Title"] = "Parameter evolution and a real digit-2 trajectory"
        for fig in figures:
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
