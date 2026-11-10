"""Plot measured force components supplied by the read-only extractor.

This module reads no checkpoint or dataset and runs no model or sampler.
"""

import numpy as np


def render_force_comparison(data, metadata, output):
    """Render final-state force distributions, comparison and spatial RMS maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    output.mkdir(parents=True, exist_ok=True)
    data_dir = output.parent / "figure_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"})
    ink, muted = "#202c3b", "#566477"
    norms = np.asarray(data["norms"], dtype=float)
    fields = np.asarray(data["pixel_rms"], dtype=float)
    component_names = list(metadata["component_names"])
    pixel_names = list(metadata["pixel_rms_names"])
    grid = int(metadata["grid"])
    if norms.ndim != 2 or norms.shape[1] != len(component_names) or not len(norms):
        raise ValueError("Force-norm array and component names are inconsistent")
    if fields.shape != (len(pixel_names), grid * grid):
        raise ValueError("Spatial RMS arrays do not match the image geometry")
    if not np.isfinite(norms).all() or not np.isfinite(fields).all():
        raise ValueError("Force plots require finite measurements")
    if np.any(norms < 0) or np.any(fields < 0):
        raise ValueError("Force norms and RMS magnitudes cannot be negative")
    index = {name: i for i, name in enumerate(component_names)}
    hidden_label = metadata.get("hidden_label", "Potts")
    labels = {
        "pair": "Visible\npairs", "onsite": "Visible\non-site",
        "global_hidden": f"Global\n{hidden_label}", "local_hidden": "Per-pixel\nIsing",
        "hidden_total": "All\nhidden", "net_energy_force": "Net energy\nforce",
    }
    chosen = ["pair", "onsite", "global_hidden", "local_hidden", "hidden_total", "net_energy_force"]
    values = [norms[:, index[key]] for key in chosen]
    colors = ["#4792b5", "#82b7ce", "#c66e41", "#e0aa78", "#a04c2c", "#57647a"]
    fig = plt.figure(figsize=(12.2, 9.0), facecolor="white")
    fig.text(.055, .953, "Force contributions at the end of generation", fontsize=18,
             weight="bold", color=ink)
    s, t = float(metadata["generation_time"]), float(metadata["forward_time"])
    finite_n = int(metadata.get("n_finite", len(norms)))
    original_n = int(metadata.get("n_samples", finite_n))
    model_label = f"{hidden_label} {int(metadata['K']):,}" if "K" in metadata else f"{hidden_label} model"
    fig.text(.055, .920,
             f"{model_label}  |  s = {s:g}, t = {t:g}  |  "
             f"{len(norms):,} saved generated configurations", fontsize=10, color=muted)
    fig.text(.055, .892, "Final-time measurements only; these panels do not show forces along trajectories.",
             fontsize=9.3, color=muted)

    ax = fig.add_axes([.075, .565, .50, .265])
    positive_values = [a[a > 0] for a in values]
    zero_counts = [int(np.sum(a == 0)) for a in values]
    if any(len(a) == 0 for a in positive_values):
        raise ValueError("A plotted force component is identically zero; adapt the logarithmic layout")
    boxes = ax.boxplot(positive_values, positions=np.arange(1, len(chosen) + 1), widths=.55,
                       patch_artist=True, whis=(10, 90), showfliers=False,
                       medianprops={"color": "#172331", "linewidth": 1.5},
                       boxprops={"linewidth": .8},
                       whiskerprops={"linewidth": .8}, capprops={"linewidth": .8})
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(.83)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, len(chosen) + 1), [labels[key] for key in chosen], fontsize=9)
    ax.set_ylabel(r"Per-configuration force norm $\|\mathbf{f}\|_2$", fontsize=10)
    ax.set_xlim(.35, len(chosen) + .65)
    ax.grid(axis="y", which="major", color="#e2e7ec", linewidth=.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(.045, .850, "a", fontsize=14, weight="bold", color=ink)
    fig.text(.075, .850, "Forces across saved configurations", fontsize=11, weight="bold", color=ink)
    fig.text(.075, .501, "Boxes: 25–75%; whiskers: 10–90%. Norms are computed before averaging over images.",
             fontsize=8.2, color=muted)

    scatter = fig.add_axes([.675, .555, .260, .285])
    pair = norms[:, index["pair"]]
    hidden = norms[:, index["hidden_total"]]
    global_hidden = norms[:, index["global_hidden"]]
    positive = (pair > 0) & (hidden > 0)
    global_positive = (pair > 0) & (global_hidden > 0)
    if not positive.any():
        raise ValueError("No positive force-norm pairs available for logarithmic comparison")
    plotted_values = [pair[positive], hidden[positive]]
    if global_positive.any():
        plotted_values.extend([pair[global_positive], global_hidden[global_positive]])
    low = min(float(values.min()) for values in plotted_values) * .8
    high = max(float(values.max()) for values in plotted_values) * 1.60
    scatter.scatter(pair[positive], hidden[positive], s=9, alpha=.23,
                    color="#386e9b", linewidths=0, rasterized=True, label="All hidden")
    if global_positive.any():
        scatter.scatter(pair[global_positive], global_hidden[global_positive], s=9, alpha=.23,
                        color="#c66e41", linewidths=0, rasterized=True,
                        label=f"Global {hidden_label}")
    scatter.plot([low, high], [low, high], color="#747e89", linestyle="--", linewidth=.9,
                 label="Equal norms")
    scatter.set_xscale("log")
    scatter.set_yscale("log")
    scatter.set_xlim(low, high)
    scatter.set_ylim(low, high)
    scatter.set_xlabel(r"Visible pairs: $\|\mathbf{f}_{\mathrm{pair}}\|_2$", fontsize=9.5)
    scatter.set_ylabel(r"Hidden force norm $\|\mathbf{f}\|_2$", fontsize=9.5)
    scatter.grid(which="major", color="#e2e7ec", linewidth=.7)
    scatter.set_axisbelow(True)
    scatter.legend(loc="lower right", fontsize=7.8, frameon=False)
    for side in ("top", "right"):
        scatter.spines[side].set_visible(False)
    fig.text(.640, .850, "b", fontsize=14, weight="bold", color=ink)
    fig.text(.675, .850, "Paired comparison", fontsize=11, weight="bold", color=ink)
    ratio = float(np.median(pair[positive] / hidden[positive]))
    global_ratio = float(np.median(pair[global_positive] / global_hidden[global_positive])) if global_positive.any() else np.nan
    ratio_text = f"Median pair / all hidden: {ratio:.3g}"
    if np.isfinite(global_ratio):
        ratio_text = f"Median pair / global: {global_ratio:.3g}\n" + ratio_text
    scatter.text(.035, .965, ratio_text, transform=scatter.transAxes,
                 ha="left", va="top", fontsize=7.6, color=muted,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})

    cosines = np.asarray(data["cosine_visible_hidden"], dtype=float)
    cancellation = np.asarray(data["cancellation_ratio"], dtype=float)
    valid_cos = cosines[np.isfinite(cosines)]
    valid_cancel = cancellation[np.isfinite(cancellation)]
    cos_text = f"{np.median(valid_cos):.3f}" if len(valid_cos) else "undefined"
    cancel_text = f"{np.median(valid_cancel):.3f}" if len(valid_cancel) else "undefined"
    fig.text(.055, .459,
             "Visible = pairs + on-site; hidden = global + per-pixel. Components are added as vectors.",
             fontsize=9.2, color=ink)
    fig.text(.055, .432,
             f"Visible–hidden alignment: median cos θ = {cos_text}.  "
             f"Median net / (visible norm + hidden norm) = {cancel_text}.", fontsize=9, color=muted)

    fig.text(.045, .385, "c", fontsize=14, weight="bold", color=ink)
    fig.text(.075, .385, "Per-pixel RMS force across configurations", fontsize=11, weight="bold", color=ink)
    positive_pixels = fields[fields > 0]
    if len(positive_pixels):
        vmin, vmax = float(positive_pixels.min()), float(positive_pixels.max())
        if vmin == vmax:
            vmin = vmax / 10
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d8dde3")
    positions = np.linspace(.145, .855, len(pixel_names))
    for position, name, field in zip(positions, pixel_names, fields):
        image_ax = fig.add_axes([position - .070, .155, .140, .190])
        image_ax.imshow(np.ma.masked_less_equal(field.reshape(grid, grid), 0),
                        cmap=cmap, norm=norm, interpolation="nearest")
        image_ax.set_title(labels.get(name, name).replace("\n", " "), fontsize=9.2, color=ink, pad=6)
        image_ax.axis("off")
    cax = fig.add_axes([.225, .106, .550, .015])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    if len(positive_pixels):
        ticks = [vmin, np.sqrt(vmin * vmax), vmax]
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{v:.2g}" for v in ticks])
        colorbar.ax.minorticks_off()
    colorbar.ax.tick_params(labelsize=8, length=2)
    colorbar.outline.set_linewidth(.4)
    fig.text(.055, .078, "One logarithmic colour scale for all maps; grey denotes zero RMS.",
             fontsize=8.4, color=muted)
    fig.text(.055, .043,
             "Energy forces f = −∇F in normalized coordinates. Reverse-SDE drift = x + 2f; Brownian forcing is not included.",
             fontsize=8.5, color=muted)
    exclusions = []
    if original_n != len(norms):
        exclusions.append(f"{original_n - len(norms)} input configurations excluded")
    if any(zero_counts) or not positive.all():
        exclusions.append("zero norms omitted from logarithmic distribution/scatter axes")
    if exclusions:
        fig.text(.055, .021, "; ".join(exclusions) + ".", fontsize=8, color=muted)
    metadata["plot_pixel_rms_colour_range"] = [vmin, vmax]
    metadata["plot_norm_zero_counts"] = dict(zip(chosen, zero_counts))
    metadata["plot_pair_hidden_ratio_definition"] = "median(||f_pair||_2 / ||f_global+f_local||_2)"
    metadata["plot_median_pair_global_ratio"] = global_ratio if np.isfinite(global_ratio) else None
    metadata["plot_force_convention"] = "Energy forces -grad F, without reverse-SDE factor 2 or explicit +x drift."
    fig.savefig(output / "force_comparison_final.pdf", dpi=200, facecolor="white")
    fig.savefig(data_dir / "force_comparison_final.svg", dpi=200, facecolor="white")
    plt.close(fig)
