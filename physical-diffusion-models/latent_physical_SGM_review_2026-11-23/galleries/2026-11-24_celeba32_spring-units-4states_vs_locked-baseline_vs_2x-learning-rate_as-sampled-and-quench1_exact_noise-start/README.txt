CelebA-32: locked baseline vs + spring (bond) units, 4 states, vs 2x learning rate; SDE as sampled and after quench 1; exact training, noise start, same trajectories per column

made 2026-11-24 01:16 by make_gallery.py, Slurm job 4307786, problem celeba32g

How to read the names:
baseline = locked K16 + 16x16/8x8 patches 64 states (main out folder); spring4 = the same + bond units with 4 states from t = 2.22; lr2 = the locked recipe at 2x learning rate; _sampled = end of the reverse SDE, _quench1 = after the zero-temperature quench of duration 1

Files:
  00_side_by_side.png   one row per model, 16 columns; column j = trajectory j in every model (same start and noise), first row real digits
  01_real_images.png    100 random real training images
  02_baseline_quench1.png   first 100 of 2000 samples (0 diverged)
      sample file: /physical-diffusion-models/out/latent_physical_SGM/celeba32g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1/samples/sde_base_n2000_q1.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  03_spring4_quench1.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K16_exact_locked-patches-64states_plus-spring-units-4states-from-t2.22_seed1/samples/sde_base_n2000_q1.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  04_lr2_quench1.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K16_exact_locked-patches-64states_2x-learning-rate_seed1/samples/sde_base_n2000_q1.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  05_baseline_sampled.png   first 100 of 2000 samples (0 diverged)
      sample file: /physical-diffusion-models/out/latent_physical_SGM/celeba32g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1/samples/sde_base_n2000.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: sampler "sde", n 2000, patch_units "16:4:64+8:2:64"
  06_spring4_sampled.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K16_exact_locked-patches-64states_plus-spring-units-4states-from-t2.22_seed1/samples/sde_base_n2000.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: sampler "sde", n 2000, patch_units "16:4:64+8:2:64"
