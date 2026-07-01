---
name: brief-synthesize-crop-findings
description: Build or refresh writeup/literature/0_findings_summary.md — the PROJECT-CENTRIC synthesis organized around our crop-classification study's modeling choices, evaluation bar, and action items. Use when asked to refresh the findings summary, rebuild the literature synthesis, or update 0_findings_summary. (For a general academic literature review instead, use brief-literature-review.)
---

Build (or update) `writeup/literature/0_findings_summary.md` by reading every per-paper briefer `.md` in `writeup/literature/` and synthesizing across them. This SKILL.md is the canonical, self-contained source — follow the structure and rules below. The synthesis reads from the briefers, not the PDFs, because the briefers carry the evaluation-protocol audit the synthesis depends on. If briefers are missing or sparse on the evaluation-protocol line, fix those first with [brief-paper](../brief-paper/SKILL.md); don't synthesize from incomplete briefers.

## When to invoke

- User asks to "refresh the findings summary" / "rebuild the literature synthesis" / "update 0_findings_summary".
- A new briefer materially changes the set of directly comparable papers.
- Project criteria change (model menu, evaluation protocol, target metrics) — the boilerplate context paragraph and the action list need re-auditing.

## Organizing principle

**Evaluation protocol is the single most important lens.** Papers are not comparable unless they share a protocol. Our study's central claim is the *lite-learning* thesis — that classical ML (LightGBM, Random Forest) on engineered time-series features (`xr_fresh`: per-pixel statistics over `EVI`, `B2`, `B6`, `B11`/`B12`, `hue`) is competitive with or better than deep temporal/patch networks while being cheaper, interpretable (SHAP), and viable on crowdsourced labels in data-scarce sub-Saharan settings (Cohen's Kappa 0.82, F1-micro 0.85 in northern Tanzania). Our own evaluation bar is field-disjoint grouped CV (`StratifiedGroupKFold` on `field_id`, `Field_size`-weighted) within one region — NOT a spatially disjoint holdout. So the synthesis keeps protocol front and centre, both to discount over-optimistic comparator numbers and to mark which papers clear a higher bar than we do:

- Section 1 sorts every paper into Tier 1 (true spatial holdout — different tile/region/zone, or cross-year/cross-sensor transfer) / Tier 2 (field-wise FID k-fold within a single region — no spatial transfer; **this is our own study's rung**) / Tier 3 (pooled or random k-fold — spatial-autocorrelation/FID leakage) / Tier 4 (reviews and theory). Flag explicitly which papers sit ABOVE our Tier-2 bar (e.g. a same-country study with a district holdout, or a cross-sensor transfer study).
- Section 2 names the 1–5 papers credibly directly comparable to our study: sub-Saharan / East African (ideally Tanzania or Kenya) smallholder Sentinel-2 field-level crop classification, especially engineered-feature-plus-classical-ML designs or any comparator evaluated at our bar or stronger.
- Sections 3–4 extract feature-design and model-architecture findings, discounted by source tier (does the evidence support engineered time-series features over raw-sequence deep nets, our band/index choices, classical ML vs CNN/RNN/U-Net, SAR-optical fusion, non-crop-class inclusion).
- Section 5 conceptual/theoretical framing (inductive bias and transfer, interpretability/SHAP, deep-learning data-hunger and compute cost, optical-vs-SAR, crowdsourced-label noise).
- Section 6 project-specific caveats surfaced by the literature audit (e.g. our bar is field-grouped CV within one region, not a spatial holdout, so our reported Kappa/F1 are an optimistic upper bound on cross-region or cross-year transfer; single-year 2023, optical-only Sentinel-2 months 1–8; crowdsourced YouthMappers labels carry geolocation/label noise; intercropping is a known optical-RS confusion source for our minority crops cassava/millet/sunflower/sorghum/cotton).
- Section 7 prioritized action list (high lift / medium / lower), naming concrete files, features, or scripts.
- Section 8 bottom-line numerical targets the literature actually supports (Kappa / macro- and per-class F1 under field-grouped CV, and realistic minority-crop recall; flag that no target here is validated under true spatial or cross-year transfer).

## Authoring rules

1. **Discount every reported number by protocol tier.** Never quote a number without naming the protocol that produced it.
2. **Group by finding, not by paper.** Multiple papers per point, multiple findings per paper.
3. **Link every paper reference** to its briefer file (URL-encoded filename). Add a citation-key block at the top mapping cite keys to briefers.
4. **Section 2 is the most important** — direct comparators define realistic targets.
5. **Section 7 must be actionable** — name files / features / scripts (e.g. `2_xr_fresh_extraction.py`, `3_sample_framework.py`, `5_model.py`, `sklearn_helpers.py`, `final_model_features_v3/`, and the results/figures in `writeup/writeup.md`).
6. **Section 8 must take a stand** — state specific Kappa / macro-F1 / per-class-F1 targets the literature supports under our field-grouped-CV bar, and separately note how much lower they might fall under genuine spatial or cross-year transfer.
7. **Distinguish what the literature supports from what it cannot speak to** — silences (spatial transfer, minority-crop recall, cross-year robustness) are findings.
8. **Update, don't append.** Edit existing sections in place when refreshing.
9. **Markdown only. No emojis.**

## Refresh vs full rebuild

- **Full rebuild** (corpus changed substantially or first build): read all briefers in parallel, write the document fresh.
- **Refresh** (one or two new briefers): read just the new briefers, edit the relevant sections — tier assignment in section 1, possible promotion to section 2, feature/architecture points in sections 3–4, an action item in section 7.

## Reference

- Per-paper briefer skill: [brief-paper](../brief-paper/SKILL.md)
- Project context for the boilerplate paragraph: [CLAUDE.md](../../../CLAUDE.md) and [writeup/writeup.md](../../../writeup/writeup.md)
- Sibling skills: [brief-literature-review](../brief-literature-review/SKILL.md), [briefer-bibliography](../briefer-bibliography/SKILL.md)
