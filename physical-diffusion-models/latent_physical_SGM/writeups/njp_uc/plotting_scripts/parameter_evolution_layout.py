"""Readable two-page layout for saved on-site and Potts coupling schedules.

This module only renders arrays supplied by the extractor. It reads no model,
dataset or checkpoint, and performs no training or sampling.
"""

import numpy as np


def render_expanded(data, metadata, output, spring_view="rms", colour_scale="linear",
                    output_stem="parameter_evolution_draft"):
    """Render visible terms and four fixed Potts states, with shared scales."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm
    from matplotlib.patches import Rectangle

    output.mkdir(parents=True, exist_ok=True)
    if colour_scale not in ("linear", "log"):
        raise ValueError("colour_scale must be 'linear' or 'log'")
    if not output_stem or "/" in output_stem or "\\" in output_stem:
        raise ValueError("output_stem must be a simple file stem")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    ink, muted = "#202c3b", "#566477"
    ids = np.asarray(data["slices"], dtype=int)
    states = np.asarray(data["states"], dtype=int)
    initial_classes = np.asarray(data.get("state_initial_classes", np.full(len(states), -1)), dtype=int)
    grid = int(data["grid"])
    if len(ids) != 5 or not len(states):
        raise ValueError("The expanded layout needs five times and at least one state")
    if data["W_states"].shape != data["Q_states"].shape:
        raise ValueError("W/Q state arrays have inconsistent shapes")
    if data["W_states"].shape[1:] != (len(states), grid * grid):
        raise ValueError("State arrays do not match state IDs and image geometry")
    limits = {
        "c_lin": max(float(np.max(np.abs(data["c_lin"][ids]))), 1e-15),
        "c_rms_map": max(float(np.max(data["c_rms_map"][ids])), 1e-15),
        "k_lin": max(float(np.max(np.abs(data["k_lin"][ids]))), 1e-15),
        "k_duff": max(float(np.max(np.abs(data["k_duff"][ids]))), 1e-15),
        "W": max(float(np.max(np.abs(data["W_states"][ids]))), 1e-15),
        "Q": max(float(np.max(np.abs(data["Q_states"][ids]))), 1e-15),
    }
    positive_duff = bool(np.min(data["k_duff"][ids]) >= 0)
    selected_arrays = {key: data[key][ids] for key in ("c_lin", "c_rms_map", "k_lin", "k_duff")}
    selected_arrays.update({key: data[key + "_states"][ids] for key in ("W", "Q")})
    normalizations, norm_metadata = {}, {}
    for key, values in selected_arrays.items():
        positive = key == "c_rms_map" or (key == "k_duff" and positive_duff)
        if colour_scale == "log" and positive:
            strictly_positive = np.asarray(values)[np.asarray(values) > 0]
            if strictly_positive.size:
                lower = float(strictly_positive.min())
                upper = limits[key]
                if lower == upper:
                    lower = upper / 10.0
                normalizations[key] = LogNorm(vmin=lower, vmax=upper)
                norm_metadata[key] = {"type": "log", "vmin": lower, "vmax": upper,
                                      "nonpositive_values": "masked grey", "base": 10}
            else:
                normalizations[key] = Normalize(0, 1)
                norm_metadata[key] = {"type": "linear_zero_fallback", "vmin": 0.0, "vmax": 1.0}
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
    metadata["colour_scale"] = colour_scale
    metadata["colour_normalizations"] = norm_metadata
    metadata["colour_limits_symmetric"] = {key: limits[key] for key in ("c_lin", "k_lin", "W", "Q")}
    if not positive_duff:
        metadata["colour_limits_symmetric"]["k_duff"] = limits["k_duff"]
    metadata["spring_view"] = spring_view
    metadata["spring_map_colour_range"] = [normalizations["c_rms_map"].vmin, normalizations["c_rms_map"].vmax]
    metadata["duffing_colour_range"] = [normalizations["k_duff"].vmin, normalizations["k_duff"].vmax]
    metadata["hidden_colour_scaling"] = "One W scale and one Q scale across every displayed state, time and page."
    metadata["edge_width_points"] = {
        "formula": "0.35 + 3.0 * abs(c_lin) / max_abs_c_lin", "common_across_frames": True,
    }

    # Coordinates are specified in inches, preserving map and text sizes when
    # the number of rows changes. Five 1.4-inch maps fill the same columns.
    fig_width, map_size, row_pitch, map_top = 11.0, 1.40, 1.82, 2.05
    column_centres = np.linspace(2.23, 8.83, 5)

    def make_page(rows, title, subtitle, stem, page_number, placeholder=False):
        height = map_top + (len(rows) - 1) * row_pitch + map_size + (1.45 if placeholder else .90)
        fig = plt.figure(figsize=(fig_width, height), facecolor="white")

        def text_at(x, down, text, **kwargs):
            return fig.text(x / fig_width, 1 - down / height, text, **kwargs)

        text_at(.45, .40, title, fontsize=17, weight="bold", color=ink)
        text_at(.45, .72, subtitle, fontsize=9.6, color=muted)
        if page_number == 1:
            text_at(.45, 1.04,
                    r"On-site energy: $E_i^{\mathrm{on}}=k_i x_i^2/2+u_i x_i^4/4+\ell_i x_i$",
                    fontsize=11, color=ink)
            text_at(.45, 1.30, "Bare coefficients are shown; k is not the full effective local curvature.",
                    fontsize=8.8, color=muted)
        else:
            text_at(.45, 1.04, r"State preference: $a_j(\mathbf{x})=b_j+\sum_i W_{ji}x_i+\sum_i Q_{ji}x_i^2$",
                    fontsize=11, color=ink)
            text_at(.45, 1.30, "Initial class denotes the initialization cluster's majority label, not a measured final class.",
                    fontsize=8.8, color=muted)
        for x, k in zip(column_centres, ids):
            text_at(x, 1.66, f"$s={data['generation_time'][k]:.2f}$", ha="center", fontsize=11, color=ink)
            text_at(x, 1.88, f"$t={data['forward_time'][k]:.2f}$", ha="center", fontsize=9, color=muted)

        for row_index, row in enumerate(rows):
            key, label, symbol, state_position, note = row
            top = map_top + row_index * row_pitch
            positive = key == "c_rms_map" or (key == "k_duff" and positive_duff)
            norm = normalizations[key]
            cmap = plt.get_cmap("magma" if positive else "RdBu_r").copy()
            cmap.set_bad("#d7dbe0")
            text_at(.23, top + .10, chr(ord("a") + row_index), fontsize=13, weight="bold", color=ink)
            text_at(.46, top + .49, label, fontsize=10, color=ink, va="center")
            text_at(.46, top + .91, symbol, fontsize=12, color=ink, va="center")
            if state_position is not None and initial_classes[state_position] >= 0:
                text_at(.46, top + 1.20, f"initial class: {initial_classes[state_position]}",
                        fontsize=7.9, color=muted, va="center")
            for x, k in zip(column_centres, ids):
                ax = fig.add_axes([(x - map_size / 2) / fig_width,
                                   1 - (top + map_size) / height,
                                   map_size / fig_width, map_size / height])
                if key == "c_lin":
                    nodes = data["patch_nodes"]
                    positions = {int(node): (int(node) % grid, -(int(node) // grid)) for node in nodes}
                    node_xy = np.asarray([positions[int(i)] for i in nodes])
                    values = data[key][k]
                    segments = [[positions[int(a)], positions[int(b)]] for a, b in data["patch_edges"]]
                    ax.add_collection(LineCollection(segments, colors=cmap(norm(values)),
                                                      linewidths=.35 + 3.0 * np.abs(values) / limits[key], zorder=1))
                    ax.scatter(*node_xy.T, s=26, color="white", edgecolor=ink, linewidth=.8, zorder=3)
                    ax.set_xlim(node_xy[:, 0].min() - .45, node_xy[:, 0].max() + .45)
                    ax.set_ylim(node_xy[:, 1].min() - .45, node_xy[:, 1].max() + .45)
                    ax.set_aspect("equal")
                else:
                    values = data[key + "_states"][k, state_position] if state_position is not None else data[key][k]
                    if norm_metadata[key]["type"] == "log":
                        values = np.ma.masked_less_equal(values, 0)
                    ax.imshow(values.reshape(grid, grid), cmap=cmap, norm=norm, interpolation="nearest")
                ax.axis("off")
            cax = fig.add_axes([10.00 / fig_width, 1 - (top + 1.23) / height,
                                .105 / fig_width, 1.04 / height])
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            norm_info = norm_metadata[key]
            if norm_info["type"] == "log":
                low, high = norm_info["vmin"], norm_info["vmax"]
                ticks = [low, np.sqrt(low * high), high]
            elif norm_info["type"] == "symmetric_log":
                ticks = [-limits[key], -norm_info["linthresh"], 0, norm_info["linthresh"], limits[key]]
            elif norm_info["type"] == "linear_zero_fallback":
                ticks = [0, .5, 1]
            else:
                ticks = [0, limits[key] / 2, limits[key]] if positive else [-limits[key], 0, limits[key]]
            cbar.set_ticks(ticks)
            cbar.ax.set_yticklabels([f"{value:.2g}" for value in ticks])
            cbar.ax.tick_params(labelsize=8, length=2)
            cbar.ax.minorticks_off()
            cbar.outline.set_linewidth(.4)
            if note:
                text_at(1.53, top + map_size + .20, note, fontsize=8.1, color=muted)

        last_bottom = map_top + (len(rows) - 1) * row_pitch + map_size
        if placeholder:
            y_top = last_bottom + .38
            box = fig.add_axes([1.53 / fig_width, 1 - (y_top + .48) / height,
                               8.0 / fig_width, .48 / height])
            box.add_patch(Rectangle((0, 0), 1, 1, transform=box.transAxes,
                                    facecolor="#f6f7f9", edgecolor="#a5aeb8",
                                    linestyle=(0, (4, 3)), linewidth=.8))
            status = "Intermediate trajectory snapshots were not saved; no trajectory was reconstructed."
            if metadata.get("snapshot_files_found"):
                status = "Snapshot files are now available; the trajectory panel remains a placeholder for review."
            box.text(.5, .69, "SAMPLE TRAJECTORY — PLACEHOLDER", ha="center", va="center",
                     fontsize=8.2, weight="bold", color=muted)
            box.text(.5, .29, status, ha="center", va="center", fontsize=8, color=muted)
            box.axis("off")
        scale_note = ("Log magnitudes; signed coefficients use symmetric log (linear near zero). Grey: nonpositive log values."
                      if colour_scale == "log" else "Linear colour scales; limits are shared across times and across states within each coefficient family.")
        text_at(.45, height - .38, scale_note, fontsize=8.0, color=muted)
        text_at(.45, height - .16,
                f"Generation time s = 5 − t. Saved coefficients, not optimizer history.  |  Page {page_number}",
                fontsize=8.2, color=muted)
        for ext in ("png", "svg"):
            fig.savefig(output / f"{stem}.{ext}", dpi=200, facecolor="white")
        return fig

    spring_key = "c_rms_map" if spring_view == "rms" else "c_lin"
    spring_symbol = r"$\sqrt{\langle c_{ij}^2\rangle_j}$" if spring_view == "rms" else r"$c_{ij}(t)$"
    spring_note = ("At each pixel: RMS of its incident springs across the full lattice; magnitude only."
                   if spring_view == "rms" else "Fixed central patch; edge width follows |c| with a small visibility floor.")
    first = int(states[0])
    page_one_rows = [
        (spring_key, "Visible\nsprings", spring_symbol, None, spring_note),
        ("k_lin", "Bare linear\nstiffness", r"$k_i(t)$", None,
         "Coefficient of x²/2; signed scale shared across all five times."),
        ("k_duff", "Duffing\ncoefficient", r"$u_i(t)$", None,
         "Coefficient of x⁴/4; this quartic run has no sextic on-site term."),
        ("W", f"Potts state {first}\nLinear", f"$W_{{{first}i}}(t)$", 0,
         "W uses the same colour scale for every displayed state on both pages."),
        ("Q", f"Potts state {first}\nQuadratic", f"$Q_{{{first}i}}(t)$", 0,
         "Q uses the same colour scale for every displayed state on both pages."),
    ]
    page_two_rows = []
    for position, state in enumerate(states[1:], start=1):
        for key, label in (("W", "Linear"), ("Q", "Quadratic")):
            page_two_rows.append((key, f"Potts state {int(state)}\n{label}",
                                  f"${key}_{{{int(state)}i}}(t)$", position,
                                  "Same state across time; shared W and Q scales across both pages." if key == "Q" else ""))
    architecture = f"All ten MNIST digits  |  {metadata['config']['K']:,}-state global Potts unit"
    if metadata["config"].get("local"):
        architecture += f" + {grid * grid} per-pixel Ising units"
    states_stem = "parameter_evolution_states" if output_stem == "parameter_evolution_draft" else output_stem + "_states"
    figures = [make_page(page_one_rows, "Visible potentials and one Potts state", architecture,
                         output_stem, 1, placeholder=True)]
    if page_two_rows:
        state_text = ", ".join(str(int(s)) for s in states[1:])
        figures.append(make_page(page_two_rows, "Three more fixed Potts states" if len(states) == 4 else "Additional fixed Potts states",
                                 f"States j = {state_text}  |  Shared {'logarithmic' if colour_scale == 'log' else 'linear'} W and Q colour scales across both pages",
                                 states_stem, 2))
    with PdfPages(output / f"{output_stem}.pdf") as pdf:
        pdf.infodict()["Title"] = "Saved physical diffusion parameters: visible potentials and fixed Potts states"
        for fig in figures:
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    metadata["figure_pages"] = [
        {"page": 1, "rows": ["spring", "k_lin", "k_duff", f"W_{first}", f"Q_{first}"],
         "preview_stem": output_stem},
    ]
    if page_two_rows:
        metadata["figure_pages"].append({
            "page": 2, "state_ids": states[1:].tolist(), "preview_stem": states_stem,
        })
