# Scientific Notes & Open Questions

These notes are intended for Dietz Lab HiWi applicants to show the reasoning behind pipeline choices and to frame the scientific conversation. They are not results.

## Why RSO instead of discrete hallucination or diffusion?

- Discrete sequence-space hallucination (Sergey's original hallucination work) uses argmax between steps, which "forgets" the gradient direction at every step. This slows convergence especially for longer chains.
- RSO keeps sequences relaxed (PSSM/logit style) and passes them straight back to AF2. The paper reports faster, more stable convergence than prior protocols.
- Diffusion models (RFDiffusion, Chroma) are excellent for *de novo* sampling and handle motif scaffolding well, but RSO shines for flexibility on arbitrary loss functions without retraining and scales up to 1000 aa.

## Why throw away the RSO sequence and re-design with ProteinMPNN?

Because the relaxed representation never maps cleanly to a realistic 20-letter sequence, and even a hard argmax of the RSO logits yields a sequence biased by the training data distribution of the *predictor*, not by the physics-like patterns ProteinMPNN learns from real PDB backbones. ProteinMPNN was specifically trained for inverse folding and consistently produces sequences with higher experimental success rates, including the soluble-weights variant used here.

## Why AlphaFold model_4_ptm single-sequence for validation?

Matches the official notebook `designability_test()` helper for consistency. The paper additionally reports ESMFold reprediction comparisons. We do not silently swap the validation model because different predictors have different systematic biases — cross-predictor agreement is part of what we want to measure.

## Why rank by 3 separate metrics rather than a composite?

- RMSD penalizes local clashes heavily; TM-score cares about topology.
- pLDDT measures the model's own confidence, which can be optimistic on novel topologies or when the predictor was used for design.
- Combining them with arbitrary weights would produce a score with unclear physical meaning. Users can add a custom composite downstream with documented semantics.

## Open research questions to discuss

The README poses 5 specific questions. Additional talking points:

1. **Length scaling of designability under fixed-iteration budget.** Our experiment iterates exactly 100 RSO steps for every length. Does increasing iterations for longer proteins (e.g., 200 steps at 300 aa) close the quality gap? How does quality-vs-wall-clock Pareto look for RSO vs. diffusion?
2. **Helix loss contribution diversity.** The default helix weight of -0.2 de-emphasizes helices. What happens if we sweep it across values and cluster the resulting topologies by secondary-structure content?
3. **Soluble-weights bias.** The soluble MPNN weights shift sequences to more negatively charged. Could this partly explain why success rates go up in the paper's in vitro assays? To what extent does it limit the functional sequence space (e.g., for proteins that need to bind positively charged partners)?
4. **Validation model sensitivity.** AF2 single-sequence vs. ESMFold vs. AF2 with initial-guess templates give different RMSD/TM-score distributions on the same candidates. Which correlates best with wet-lab success *on RSO-designed proteins* specifically?
5. **Copy-number tasks.** The project default copies=1 (monomers). The official notebook also supports homo-oligomers, binder design, heterodimers, and site scaffolding — these share the same 4-stage pattern but with tailored losses. A natural follow-up is extending `Experiment` schema with per-protocol loss presets for each.
