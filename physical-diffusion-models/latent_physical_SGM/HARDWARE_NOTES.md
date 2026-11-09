# Hardware notes: hidden units coupled to all pixels

Research summary of 2026-11-09 (workflow wf_270f1765: four literature sweeps, 113 claims checked against their sources (96 confirmed; overstated, unverifiable and wrong claims dropped), one synthesis). Parts marked "reasoning" are estimates, not measurements.

**Short answer.** Wiring hidden units to every pixel is the easy part: K units need K·N couplings, not N². The hard part is what our model needs from those couplings. Good samples need 50k–800k of them per slice. Each is signed, follows a 100-knot schedule, and must set the hidden field to within about T = 0.02. One Ising unit is easy to build in electronics or optics. Several units can be built today as a hybrid: physical pixels plus an analog crossbar that computes the averaged hidden force. Outside electronics, I found no built system with a bistable hub that has hundreds of programmable couplings.

**Connectivity (28×28, per slice).** The visible model already has 180,810 pair couplings (59% of all pairs) and 545,566 parameters.

| hidden block | couplings | vs visible | clean yield |
|---|---|---|---|
| none (paper) | 0 | – | 31.9% |
| 1 Ising | 785 | 0.14% | 43.5% |
| Potts 32 + both | 51,776 | 9.5% | 76.4% |
| Potts 128 + both | 202,400 | 37% | 86.3% |
| Potts 512 + both, power 4 | ~805k | ~148% | 92.3% |

One unit only makes the class decision. The quality comes from the M×N prototype couplings and the spring. With nearest-neighbour pixel couplings (1,512 pairs), the hidden block would be almost all of the wiring.

**Platforms.**
- *Electronics (easiest).* The hidden block is a crossbar. A forward read gives the fields and a transposed read gives the force back on the pixels. [NeuRRAM](https://pmc.ncbi.nlm.nih.gov/articles/PMC9385482) ran a dense 794×120 RBM with on-chip Gibbs sampling this way.
- *Photonics.* A [photonic RBM](https://arxiv.org/abs/2508.20472) coupled 196 hidden units to all 196 pixels at 14×14, but its sigmoid and sampling were digital. Detectors return |u|², so the sign of the field costs an extra read. Masks refresh at 2–20 kHz at best. The realistic version is a measurement-feedback loop with an electronic hidden block, as in the [100,512-spin coherent Ising machine](https://pmc.ncbi.nlm.nih.gov/articles/PMC8480917/).
- *Mechanical/optomechanical.* The coupling terms exist: a membrane couples to a cavity mode linearly or as x² ([Sankey 2010](https://arxiv.org/abs/1002.4158)). The hub is what's missing. Shared modes that couple many elements are linear, and a linear hub integrates out to −a(x)²/2k. That is a softening along W and never gives the ±W mixture (reasoning). One optical mode has coupled 7 oscillators, with only global knobs ([Zhang 2015](https://link.aps.org/doi/10.1103/PhysRevLett.115.163902)).

**Bottlenecks for our model.**
1. *Time slices.* Knots are 0.05 apart, against about 0.03 pixel relaxation time, and |W| grows 25–40× towards t = 0. Storing every slice of Potts 512 + both takes about 80M values. If the arrays are rewritten every knot, each rewrite has about 1.7 relaxation times (reasoning). Fast volatile memory such as SRAM can do that. Non-volatile cells cannot: NeuRRAM's write-verify takes about 56 µs per cell, and endurance is 10⁵–10⁹ cycles ([Haensch 2022](https://arxiv.org/abs/2206.08735)).
2. *Speed.* DESIGN §4 needs τ_c ≪ T/(2S). At t = 0, T/(2S) is about 1/70 of the pixel relaxation time (14×14, Potts 128 + both). The bound may be loose: with replicas, a slow hub adds friction but no bias. A deterministic tanh/softmax circuit meets it if the pixels relax on µs scales. A thermally switching bistable hub needs a barrier of a few T and intrawell rates 10²–10³ times the pixel rate (reasoning).
3. *Precision.* A crude estimate (reasoning): near t = 0, static errors must stay below about 0.5% of rms W and 0.1% of rms Q, i.e. 9–11 bits. PCM gives about 4 bits and drifts ([Joshi 2020](https://arxiv.org/abs/1906.03138)). Training on the device should absorb static mismatch, but not drift or read noise.
4. *Potts winner-take-all.* The wiring is only O(M). The problem is that we need a softmax at T, and built Potts hardware has only 2–4 states ([Callahan-Coray 2026](https://arxiv.org/html/2602.04200v1); Zhou 2025). Computing the average instead (forward read, analog softmax, transposed read) avoids sampling, and it is what the simulation does.

**Ways to cut the wiring or the schedule.**
- *Frozen closed-form W* (`--hidden-fixed`, tested): same clean yield at 14×14 and 28×28, and the trained spring adds about 6 points. Its time dependence factorises through the eigenbasis of S_w. Fixed arrays plus 784 time-dependent gains could then replace the 100 programmed slices (reasoning, untested).
- *K Ising units* (untested; the code refuses K > 1 at initialisation): no winner-take-all and the same K·N wiring, but blends may come back.
- *Low-rank couplings* (untested): put R linear nodes between the pixels and the states. This is exact for one Potts unit and needs 83k–166k couplings at R = 32–64, instead of about 0.8M for Potts 512.
- *Local units:* per-pixel units (tested) only stabilise. Patch units feeding the Potts unit (untested) risk slow mixing. In simulation, restricted connectivity was the most damaging constraint for RBMs ([Dumoulin 2014](https://arxiv.org/abs/1312.5258)).

**Two experiments worth running (no retraining).**
1. An SVD of the saved W and Q, across states and across slices. It shows whether low-rank couplings or a few time-basis arrays would be enough.
2. Quantise and perturb the saved couplings (4–8 bits; 1–10% static and per-read noise) and resample. This measures the precision requirement directly.
