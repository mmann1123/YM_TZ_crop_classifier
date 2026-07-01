# Lessons learned from participatory land use planning with high-resolution remote sensing images in Tanzania: Practitioners' and participants' perspectives

**Citation:** Eilola, S., Käyhkö, N., & Fagerholm, N. (2021). Lessons learned from participatory land use planning with high-resolution remote sensing images in Tanzania: Practitioners' and participants' perspectives. *Land Use Policy*, 109, 105649. DOI: `10.1016/j.landusepol.2021.105649` (verified against Crossref: title, journal, authors, and year all match).

## Paper type

This is a **qualitative social-science / policy study, not a crop or land-cover classification study.** It investigates how practitioners and participants in Tanzania experience participatory mapping (PM) and participatory GIS (PGIS) that use high-resolution remote-sensing imagery, across six use cases. There is no classifier, no map accuracy assessment, and no quantitative remote-sensing result. Per rule 7 (and rule 7's review/theory guidance), the Relevance section focuses on what this paper contributes to our crowdsourcing thesis, the data-scarce low-ICT Tanzanian context, and the documented value *and* noise/bias of participatory ground truth. This is the **strongest anchor in the corpus for our crowdsourced-ground-truth argument** because it explicitly names the same community-mapping ecosystem our YouthMappers data comes from.

## Objectives

- Document, from the perspectives of both **practitioners** (those running PM exercises) and **participants** (community members), the perceived benefits and limitations of participatory mapping methods that engage communities on top of high-resolution remote-sensing imagery.
- Identify the enabling factors and barriers to adopting these geospatial PM methods in real-world Tanzanian land-use planning and decision-making.
- Surface context-specific factors (beyond generic enablers like supportive policy) that shape whether participatory geospatial data gets used.

## Methods

- **Design:** qualitative, multi-case study spanning **six PM use cases in Tanzania** (urban flood mapping, gendered data collection, rural settlement/health mapping, land-tenure formalization, rural spatial planning).
- **Data collection:** semi-structured **interviews with 12 practitioners** (including six government officers at national and local level, plus geospatial/information-science experts) conducted face-to-face or by phone, April–September 2018; **one group discussion**; and **feedback surveys among PM participants** across the cases.
- **Analysis:** **conventional qualitative content analysis** in `NVivo 11` — open coding of the interview transcripts into initial codes, grouped into themes and subthemes emerging from the data; the group-discussion dataset was coded and **triangulated** against the interview themes, and the data source of each perception (practitioner vs participant) is reported throughout.

**Evaluation protocol:** Not a classification study; no train/test generalization protocol applies. The analysis is validated by **qualitative methodological rigor — triangulated content analysis across interviews, a group discussion, and participant surveys, with iterative open coding and theme saturation in `NVivo`** — not by any map accuracy, Kappa, or confusion matrix. There is no quantitative remote-sensing output to place on the pooled-kfold → field-grouped-CV → spatial-holdout spectrum; the paper's "results" are perceived benefits and limitations, not measured accuracy.

## Key Findings

- **The exact crowdsourcing ecosystem behind our ground truth is named explicitly.** The paper cites **YouthMappers** (`youthmappers.org`, gendered data collection), **Crowd2Map** and the **Humanitarian OpenStreetMap Team / HOT** (`crowd2map.org`, `hotosm.org/.../tanzania`, rural settlement mapping), and **Ramani Huria** (`ramanihuria.org`, Dar es Salaam flood mapping), alongside **OpenStreetMap** as the global volunteer-mapping exemplar built on high-resolution satellite imagery. This is direct, citable evidence that the volunteer/crowdsourced mapping infrastructure our project relies on is established and operating in-country.
- **Tanzania is a data-scarce, low-ICT context where participatory/crowdsourced spatial data fills a real gap.** Practitioners report PM produces "previously non-existent" spatial knowledge; the country is undergoing large-scale digital transformation, and national plans (Five-Year Development Plans; Development Vision 2025) explicitly call for ICT-infrastructure and geospatial-skills investment. **Poor ICT infrastructure, lack of skilled geospatial experts, and weak institutional support** are the most-cited limitations.
- **Documented benefits of participatory ground truth:** higher mapping coverage with fewer data gaps, higher local relevance and content accuracy, improved practitioner work quality and professional confidence, and greater community engagement, spatial understanding, trust, and process ownership.
- **Documented noise/bias and validity concerns of participatory ground truth (load-bearing for us):**
  - One practitioner warns the **high-accuracy assumption has a downside — users may "trust the image and not do enough ground verification," and an old or wrongly interpreted image can cause inaccurate land allocations** (i.e., uncorrected reference error).
  - Practitioners note the **difficulty of validating participatory data**, which can undermine its credibility and raises the question "what is the value of local knowledge" when it cannot be independently checked.
  - Dependence on skilled facilitators and technology, and loss of community control over the process, can perpetuate **mistrust** and bias outcomes.

## Relevance to Our Crop-Classification Study

- **Direct support for our crowdsourced-ground-truth thesis.** Our YouthMappers-collected training labels sit inside exactly the participatory/volunteer mapping ecosystem this paper studies and names (YouthMappers, Crowd2Map, HOT, Ramani Huria, OSM). This is the best in-corpus citation establishing that crowdsourced geospatial data collection is an accepted, policy-relevant practice in Tanzania — grounding the "& Crowd Sourcing" half of our manuscript's title and method.
- **Motivates the data-scarcity framing.** It independently documents that Tanzania is data-scarce with limited ICT and few skilled experts, reinforcing why a *lite*, GPU-free, low-cost approach (engineered features + tree-based ML on freely available Sentinel-2) and crowdsourced labels are appropriate to the setting, rather than data- and compute-hungry deep learning trained on dense expert annotation.
- **Names the value-and-noise tradeoff we must manage in our labels.** The "trust the image, skip ground verification" and "hard to validate participatory data" findings are a documented account of the very label-noise and reference-bias risks in crowdsourced crop ground truth. This justifies our mitigations — field-size-based buffering, multi-source label merging, and field-disjoint `StratifiedGroupKFold` grouped by `field_id` — as principled responses to participatory-data noise, and is worth citing when we discuss ground-truth quality and uncertainty.
- **Not a methods or accuracy comparator.** Because there is no classifier or map evaluation, this paper informs our motivation, data-provenance, and limitations discussion only — never our quantitative benchmarking.

## Evaluation Caveats

- **Qualitative, perception-based evidence.** Findings are practitioners'/participants' *perceptions* of benefits and limitations, not measured outcomes; they establish that crowdsourced mapping is used and valued in Tanzania, but cannot quantify the resulting data accuracy.
- **Small, purposive sample.** 12 practitioners, one group discussion, and participant surveys across six cases; conclusions are context-rich but not statistically generalizable.
- **No remote-sensing accuracy.** No classification, no Kappa, no confusion matrix — nothing to place on our evaluation spectrum; do not treat any statement here as an accuracy figure.
- **Geographic/topical breadth.** The six cases span flood, health, tenure, and planning applications (largely not crop typing), so its relevance to *crop* ground truth is by analogy to the shared crowdsourcing infrastructure, not a direct crop-mapping result.
- **Self-report and facilitator-dependence bias.** The participatory-data quality concerns it raises (reference error, validation difficulty, mistrust) are themselves caveats on using such data uncritically — a point we should foreground rather than gloss when citing the crowdsourcing benefits.
