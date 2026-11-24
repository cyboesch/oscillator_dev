CelebA-32, exact training, noise start, seed 1, same trajectories per column; every model at its own best quench duration (Inception FID in the legend)

made 2026-11-24 05:30 by make_gallery.py, Slurm job 4310050, problem celeba32g

How to read the names:
baseline_quench1 = locked K16 + 16x16/8x8 patches, 64 states, quench 1 (20.20); spring2states_quench0.4 = + bond units with 2 states (18.40); spring4states_lr2_quench0.4 = + bond units with 4 states + 2x learning rate (18.04, best of the night); states128_quench0.7 = 128 states in both patch sets, no bond units (18.88); K784_states128_quench0.7 = the same with a 784-state global unit (18.39); spring4states_quench1 = bond units after the full quench 1 (26.36): the plateaus of an over-long quench

Files:
  00_side_by_side.png   one row per model, 16 columns; column j = trajectory j in every model (same start and noise), first row real digits
  01_real_images.png    100 random real training images
  02_baseline_quench1.png   first 100 of 2000 samples (0 diverged)
      sample file: /physical-diffusion-models/out/latent_physical_SGM/celeba32g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1/samples/sde_base_n2000_q1.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  03_spring2states_quench0.4.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K16_exact_locked-patches-64states_plus-spring-units-2states-from-t2.22_seed1/samples/sde_base_n2000_q0.4.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  04_spring4states_lr2_quench0.4.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K16_exact_locked-patches-64states_plus-spring-units-4states-from-t2.22_2x-learning-rate_seed1/samples/sde_base_n2000_q0.4.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  05_states128_quench0.7.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K16_exact_coarse16x16-every4px-128states-from-t4.49_fine8x8-every2px-128states-from-t2.22_seed1/samples/sde_base_n2000_q0.7.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:128", patch2 "8:2:128", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  06_K784_states128_quench0.7.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K784_exact_coarse16x16-every4px-128states-from-t4.49_fine8x8-every2px-128states-from-t2.22_seed1/samples/sde_base_n2000_q0.7.npy
        training: K 784, hidden "potts", seed 1, patch "16:4:128", patch2 "8:2:128", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  07_spring4states_quench1.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K16_exact_locked-patches-64states_plus-spring-units-4states-from-t2.22_seed1/samples/sde_base_n2000_q1.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  nearest_spring4states_lr2_quench0.4.png   12 samples of spring4states_lr2_quench0.4 above their nearest training images
