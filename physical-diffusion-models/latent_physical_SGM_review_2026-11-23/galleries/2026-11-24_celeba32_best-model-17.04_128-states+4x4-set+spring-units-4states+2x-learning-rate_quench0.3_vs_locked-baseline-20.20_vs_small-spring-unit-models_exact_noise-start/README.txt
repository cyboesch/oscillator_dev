CelebA-32, exact training, noise start, seed 1, same trajectories per column; best model of the review (17.04) against the locked baseline (20.20) and the small spring-unit models; every model at its own best quench duration

made 2026-11-24 09:06 by make_gallery.py, Slurm job 4311364, problem celeba32g

How to read the names:
best_128states_4x4set_spring4_lr2_quench0.3 = K16 + 16x16 every 4 px and 8x8 every 2 px patches with 128 states + 4x4 every 2 px (64 states, from t = 1.21) + bond (spring) units with 4 states + 2x learning rate, quench 0.3: Inception FID 17.04; the same _as-sampled: before the quench (46.23); the same _quench1: after the lock-in's quench 1 (25.52); baseline_quench1 = locked K16 + 16x16/8x8 patches with 64 states, quench 1 (20.20), _as-sampled (76.70); spring2_late4x_quench0.4 = locked + bond units with 2 states + 4x steps in slices 85-99 (17.96; the locked recipe's size + 1%)

Files:
  00_side_by_side.png   one row per model, 16 columns; column j = trajectory j in every model (same start and noise), first row real digits
  01_real_images.png    100 random real training images
  02_baseline_quench1.png   first 100 of 2000 samples (0 diverged)
      sample file: /physical-diffusion-models/out/latent_physical_SGM/celeba32g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1/samples/sde_base_n2000_q1.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  03_best_128states_4x4set_spring4_lr2_quench0.3.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K16_exact_coarse16x16-every4px-128states-from-t4.49_fine8x8-every2px-128states-from-t2.22_plus-4x4-every2px-64states-from-t1.21_plus-spring-units-4states-from-t2.22_2x-learning-rate_seed1/samples/sde_base_n2000_q0.3.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:128", patch2 "8:2:128", patch3 "4:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  04_spring2_late4x_quench0.4.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K16_exact_locked-patches-64states_plus-spring-units-2states-from-t2.22_4x-training-steps-in-slices-85-to-99_seed1/samples/sde_base_n2000_q0.4.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06, steps_schedule "85:4.0"
        sampling: 
  05_best_128states_4x4set_spring4_lr2_quench1.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K16_exact_coarse16x16-every4px-128states-from-t4.49_fine8x8-every2px-128states-from-t2.22_plus-4x4-every2px-64states-from-t1.21_plus-spring-units-4states-from-t2.22_2x-learning-rate_seed1/samples/sde_base_n2000_q1.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:128", patch2 "8:2:128", patch3 "4:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: 
  06_baseline_as-sampled.png   first 100 of 2000 samples (0 diverged)
      sample file: /physical-diffusion-models/out/latent_physical_SGM/celeba32g_K16_ema_potts_quad_local_quartic_r1_lin_pooled_mixst_patch16s4S64on10+patch8s2S64on55_seed1/samples/sde_base_n2000.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:64", patch2 "8:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: sampler "sde", n 2000, patch_units "16:4:64+8:2:64"
  07_best_128states_4x4set_spring4_lr2_as-sampled.png   first 100 of 2000 samples (0 diverged)
      sample file: celeba32g_K16_exact_coarse16x16-every4px-128states-from-t4.49_fine8x8-every2px-128states-from-t2.22_plus-4x4-every2px-64states-from-t1.21_plus-spring-units-4states-from-t2.22_2x-learning-rate_seed1/samples/sde_base_n2000.npy
        training: K 16, hidden "potts", seed 1, patch "16:4:128", patch2 "8:2:128", patch3 "4:2:64", patch_on_slice 10, patch_init_scale 0.06
        sampling: sampler "sde", n 2000, patch_units "16:4:128+8:2:128+4:2:64"
  nearest_best_128states_4x4set_spring4_lr2_quench0.3.png   12 samples of best_128states_4x4set_spring4_lr2_quench0.3 above their nearest training images
