import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


def plot_time_dependent_energy(energy_fn,T=1.0, n_lines=51, center_x=-2.0, y_min=0.5, y_max=6.0, reverse=False):
    # Create a grid of time values
    if reverse:
        t_values = jnp.linspace(1.0*T, 0.0, n_lines)
    else:
        t_values = jnp.linspace(0.0, 1.0*T, n_lines)

    x = jnp.linspace(center_x - 6, center_x + 6, 1000)

    fig = plt.figure(layout="constrained", figsize=(8, 6), dpi=400)
    mosaic = [["main"], ["colorbar"]]
    ax_dict = fig.subplot_mosaic(mosaic, height_ratios=[20, 2])

    ax_main = ax_dict["main"]
    ax_colorbar = ax_dict["colorbar"]

    # Create a reversed colormap
    if reverse:
        cmap = plt.get_cmap("plasma")
    else:
        cmap = plt.get_cmap("plasma_r")

    # mark some ratio the ts for special visualization ensuring that 0 and 1 are always marked
    ratio = 4
    marked_ts = t_values[:: len(t_values) // ratio]
    marked_ts = jnp.concatenate([marked_ts, jnp.array([0.0, 1.0])])

    for i, t in enumerate(t_values):
        energy = jax.vmap(lambda x: energy_fn(x, t))(x)

        color = cmap(i / (len(t_values) - 1))
        linestyle = "--" if t in marked_ts else "-"
        alpha = 1.0 if t in marked_ts else 0.5
        label = f"t={t:.2f}" if t in marked_ts else ""
        linewidth = 3 if t in marked_ts else 1
        zorder = 10 if t in marked_ts else 1
        ax_main.plot(
            x,
            energy,
            color=color,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )

    if reverse:
        ax_main.set_title("Potential Energy as a Function of Position (x) and Time (T-t)")
    else:
        ax_main.set_title("Potential Energy as a Function of Position (x) and Time (t)")
    ax_main.set_xlabel("Position $x$")
    ax_main.set_ylabel("Potential Energy $U(x, t)$")
    ax_main.grid(True)
    ax_main.set_ylim(y_min, y_max)

    # Create a colorbar with reversed colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    if reverse:
        cbar = fig.colorbar(sm, cax=ax_colorbar, orientation="horizontal", label="Time $T-t$")
    else:
        cbar = fig.colorbar(sm, cax=ax_colorbar, orientation="horizontal", label="Time $t$")
    cbar.set_ticks([0, 0.5, 1])
    if not reverse:
        cbar.set_ticklabels(
            [
                f"{t_values[0]:.2f}",
            f"{t_values[len(t_values)//2]:.2f}",
            f"{t_values[-1]:.2f}",
            ]
        )
    else:
        cbar.set_ticklabels(
            [
                f"{t_values[0]:.2f}",
            f"{t_values[len(t_values)//2]:.2f}",
            f"{t_values[-1]:.2f}",
            ]
        )
    ax_colorbar.xaxis.set_ticks_position("bottom")
    ax_colorbar.xaxis.set_label_position("bottom")

    plt.show()

