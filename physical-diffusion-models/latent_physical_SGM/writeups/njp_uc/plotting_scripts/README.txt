Draft figures for npj Unconventional Computing
=============================================

PDFs go to ../figures; SVGs and JSON/NPZ data go to ../figure_data.
The same sibling-directory layout applies when --output-dir is supplied.
PNG previews are not saved. Captions are in ../figure_captions.tex.
The manuscript main.tex is unchanged.
Static figure scripts only read completed results. The optional video pipeline
below generates one isolated CPU trajectory; it does not train or request a GPU.

Rebuild from this manuscript directory (writeups/njp_uc):

    python3 plotting_scripts/make_conceptual_figure.py
    bash plotting_scripts/render_parameter_evolution.sh --also-linear
    bash plotting_scripts/render_force_comparison.sh

The additional visible_coupling_evolution figure is rendered by
make_visible_coupling_evolution.py with NumPy/Matplotlib (use the same container
as the parameter script if they are not installed in your Python environment).

The conceptual renderer uses Python's standard library and pdflatex;
pdftocairo also exports SVG to figure_data when installed. Edit make_conceptual_figure.tex
to change the scientific diagram.

The parameter wrapper uses the project's existing container/container_image.sif,
with one CPU thread and no GPU binding. With NumPy and Matplotlib available in
your Python environment, the equivalent direct command is:

    python3 plotting_scripts/make_parameter_evolution.py

Optional arguments include --run PATH, --states 0,1,2,4, --slices 0,25,50,75,99,
--patch-start 12,12, --patch-size 4, --spring-view rms|graph and --output-dir PATH. The layout expects
five time slices of a completed Potts run with quadratic hidden couplings.
--state INTEGER remains available to show a single state instead of --states.
The default --colour-scale log uses logarithmic positive magnitudes and a
symmetric logarithmic scale for signed coefficients. --also-linear saves
a separate linear-scale comparison using the same extracted arrays.
The default is the completed 28x28 all-ten-digit Potts1536 run below. It is
a representative run for this draft, not a claim of an optimal architecture.

    out/latent_physical_SGM/
      mnist28_d0123456789_K1536_ema_potts_quad_local_quartic_r1_lin_pooled

Outputs
-------

figures/conceptual_draft.pdf
    Architecture alternatives, clamped local learning and open-loop generation.
    Entirely schematic: the image sequence is a specified drawing, not research
    output. Its noise pattern and digit are defined in the editable TikZ source.
    Global Ising units and one global Potts unit are alternative architectures;
    additional per-pixel Ising units are omitted from the drawing.

figures/parameter_evolution_draft.pdf
    Expanded two-page diagnostic PDF with real saved coefficients at five
    diffusion times. SVGs for pages 1 and 2 are retained in figure_data as
    parameter_evolution_draft.svg and parameter_evolution_states.svg.
    The layout is in parameter_evolution_layout.py.

    Page 1 shows the full-lattice spring magnitude, bare on-site linear
    stiffness k_lin, on-site Duffing coefficient k_duff, and W/Q for state 0.
    Page 2 shows W/Q for three additional fixed states, j=1,2,4.
    Their initial cluster majority labels are 6,2,9; these states were chosen
    to add different initial digit clusters, not by sample quality or occupancy.
    Initial cluster labels do not establish final learned-state class purity.

    The spring row shows a full
    28x28 heat map: at each pixel, sqrt(mean_j c_ij^2) over its actual neighbours.
    This summarizes magnitude; it is not a signed single-edge coefficient.
    Every hidden map shows all 784 pixel couplings. One W scale and one Q scale
    are shared across all selected states AND times on both pages. Other
    coefficient families each have their own common scale across time.
    In the default log view, positive spring RMS and Duffing magnitudes use
    logarithmic scales from their smallest to largest displayed positive value.
    Signed k_lin, W and Q use symmetric logarithmic scales, with a linear
    interval around zero extending to 1% of that family's maximum absolute
    displayed value. This handles zero and negative coefficients without
    discarding their signs. The interval is fixed across times and states.

    parameter_evolution_linear.pdf is the two-page linear-scale comparison;
    corresponding SVGs in figure_data use the same stem and _states suffix.
    In this view the positive scales begin at zero and signed scales are
    symmetric about zero. Different contrast does not indicate different data.

    On-site energy convention: k_lin*x_i^2/2 + k_duff*x_i^4/4 + bias*x_i.
    The sextic term is zero in this run. k_lin is a bare coefficient; full
    visible local curvature is k_lin + 3*k_duff*x_i^2 + 2*sum_j c_ij, and
    marginalizing hidden units adds further state-dependent curvature.
    Thus a negative k_lin alone does not imply negative total curvature.
    Spring forces vanish for any uniform x. They do not by themselves pin
    the absolute background amplitude. Raw zero intensity corresponds to
    approximately x=-0.425 in this run's standardized coordinates.

    --spring-view graph reproduces the original 42-spring central 4x4 subgraph
    (rows/columns 12..15, zero based). Width is
    0.35 + 3.0 |c_ij| / max|c_ij| points; the 0.35-point floor keeps weak edges
    visible. The graph's intermediate-time maximum is NOT representative of
    the full lattice. Global spring RMS at the five displayed times is
    0.00123, 0.00299, 0.01800, 0.02308, 0.31698: its maximum is at the data end.
    For the central patch the corresponding values are
    0.00151, 0.00567, 0.03274, 0.04239, 0.02254.

    This is a generation-time schedule s=5-t, NOT optimizer-step history.
    Stored indices 0,25,50,75,99 correspond to s=0,1.2626,2.5253,3.7879,5.
    Coefficients are raw checkpoint values: no Potts common-mode removal and
    no per-frame rescaling. c_ij multiplies (x_i-x_j)^2; mechanical spring
    stiffness under the 1/2 k (x_i-x_j)^2 convention would be 2 c_ij.

    This run also has 784 per-pixel Ising units. Their parameters, visible
    linear biases, hidden biases and the remaining Potts states are not plotted.
    The run used exact hidden statistics for training (config hidden_rule=exact).
    These images do not establish training with a finite number of hidden reads.

    Page 1 retains an explicit trajectory placeholder because no *_snapshots.npz files were
    saved in this run's samples directory at drafting time. No intermediate
    images are inferred from final samples. If snapshots appear later, rerunning
    will flag their availability but keep the placeholder pending figure editing.

figure_data/parameter_evolution_data.npz
    Small extracted dataset: all 100 times of the four selected W/Q states,
    on-site k_lin and k_duff, all 2,970 springs and per-pixel RMS maps, plus
    the original 42-spring subset, time coordinates, initial cluster labels,
    state/edge IDs and plotted slices. W_states/Q_states have shape
    (time, selected state, pixel); W/Q retain the first state for compatibility.
    This keeps the exact plotted numbers with the paper without copying the
    1.8 GB hidden checkpoint. Source arrays are memory-mapped read-only, and
    extraction aborts if source metadata changes during the read.

figure_data/parameter_evolution_provenance.json
    Run config, source size/modification times, config and extracted-data hashes,
    selection rules, colour limits/normalizations, time ordering and placeholder
    status. parameter_evolution_linear_provenance.json records the comparison.

figures/visible_coupling_evolution.pdf
    Four spatial maps retain individual signed edge coefficients: horizontal,
    vertical and the two diagonal orientations. Five columns use the same
    time slices. The strength curves compare all 2,970 edges with the original
    42-edge central patch across all 100 saved times. Each edge is counted once
    in the global curve. This diagnostic separates location-specific patterns
    from whole-lattice behaviour.

figures/force_comparison_final.pdf
    Force contributions measured on all 2,000 saved final generated images
    from sde_base_n2000.npy, at forward time t=0 / generation time s=5.
    This is a final-time comparison, not a force curve along trajectories.
    Only nonfinite configurations would be excluded and reported; the default
    dataset contains 2,000 finite configurations and none were excluded.

    The force components, using standardized coordinates, are:
      visible pairs: -2 sum_j c_ij (x_i-x_j)
      visible on-site: -(k_i x_i + u_i x_i^3 + ell_i)
      global Potts: sum_m p_m(x) [W_mi + 2 Q_mi x_i]
      per-pixel Ising: wl_i tanh[(bl_i + wl_i x_i)/T]
    Incident edge contributions are summed at each pixel before taking its
    component's vector norm over all 784 pixels. All-hidden force is the
    vector sum of global and per-pixel hidden forces, not the sum of their norms.

    The plots show per-configuration L2 norms, paired scatter plots, and
    per-pixel RMS forces across configurations on one logarithmic colour scale.
    Norms are measured before averaging across configurations. Cosine alignment
    and net/(visible+hidden norm) reveal cancellation between components.
    The energy force is -grad F; the full reverse-SDE drift is x+2*(-grad F).
    White Brownian noise is not assigned a finite instantaneous force norm.

    The on-site/global-hidden split depends on the saved parameterization:
    a common shift of all Potts W/Q, compensated by visible bias/stiffness,
    preserves total F while redistributing those two force components.
    Norm dominance is therefore not a causal importance or ablation result.

    Scripts: make_force_comparison.py and force_comparison_layout.py.
    The wrapper uses one CPU thread in the existing container. No training,
    sampling or JAX/model import is performed. Only the final hidden slice
    is copied, not the full multi-GB history. Optional arguments include
    --run PATH, --samples PATH, --batch 128 and --output-dir PATH.

figure_data/force_comparison_final_data.npz / force_comparison_final_metrics.json
    Per-sample norms, original sample indices, per-pixel RMS maps, alignment
    and cancellation statistics, exact input provenance and summary quantiles.
    Sample manifest's visible checkpoint hash is verified against the history.
    Energy finite differences independently checked all four force components
    on the first three saved samples (maximum scaled error about 2e-8).
    The sum of internal pair forces was also checked to vanish.

figure_captions.tex
    In the manuscript directory beside main.tex: suggested LaTeX figure
    environments for author review. Not yet included
    by main.tex. Paths assume compilation from the manuscript directory.

videos/potts_state_probabilities.mp4
    Animation of a real reverse-SDE trajectory: the visible image, conditional
    probabilities of Potts states, mass by initial digit group, and the
    decomposition of posterior entropy into between-group and within-group
    uncertainty. These are conditional probabilities of the averaged model,
    not a simulated sequence of discrete hidden-state switching events.
    Initial cluster labels are approximate digit associations, not independently
    measured final-state labels. Twelve states are selected once by their
    maximum probability over the recorded trajectory; all omitted probability
    is explicitly included as "all other states".

    sample_potts_video.py streams adjacent parameter knots and uses Diffrax
    SRA1 with the production tolerances (rtol=1e-4, atol=1e-6, Brownian tolerance
    1e-12), matching the base-2000 trajectory's initial state and Brownian key.
    One global Brownian tree is shared across all 99 intervals. The adaptive
    solver restarts at parameter knots; the numerical endpoint need not be
    bitwise identical to the production solve. Final differences are recorded.
    Five saves per interval produce 397 distinct frames over generation s=0..5.

    run_potts_video_cpu.sh requests a separate Slurm allocation (2 CPU cores,
    8 GB, no GPU) and runs both sampling and encoding. Pass the absolute
    oscillator_dev workspace directory as its first argument; supply a Slurm
    --output path under figure_data/potts_video when submitting. It leaves
    existing training runs, checkpoints and sample folders unchanged and
    refuses to overwrite an existing final video.

    render_potts_video.py also works independently from the saved trajectory.
    Since the container lacks ffmpeg, --raw-video streams 1280x800 RGBA frames
    directly to host ffmpeg; no PNG frames are saved. At 20 fps the default
    video is about 20 seconds. --summary emits figures/potts_video_summary.pdf.

figure_data/potts_video/
    trajectory_000.npz contains generation_time, states, posterior,
    state_initial_classes, normalization_mean and normalization_std.
    trajectory_000.json records source provenance, initial/Brownian keys,
    solver settings, per-interval step counts, independent NumPy/JAX drift
    checks and final discrepancy against the saved reference sample.
    Rounded normalization values from the training log are used for display
    only, not integration. Partial files record progress while sampling and
    are removed after successful completion. Scheduler logs and video_info.json
    remain here for reproducibility and troubleshooting.

Verification and provenance
---------------------------

Both figures were rendered and visually inspected. Independent read-only
checks matched W/Q state 0 and all 42 displayed c_ij values against the saved
slice_000.npz and slice_099.npz exactly. Independently enumerated connectivity
agreed with all 2,970 radius-1 edges and with the selected patch indices.

The diagram source and plotting scripts were prepared with OpenAI Codex
assistance. All conceptual shapes are specified in code; no text-to-image
generator was used. The quantitative figure is computed from saved coefficients.
