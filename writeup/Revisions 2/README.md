# JSTARS-2025-00807 Minor Revision (Round 2)

## Submission Files

| File | Description | Submit As |
|------|-------------|-----------|
| `output_JSTARS-2025-00807.pdf` | Clean revised manuscript | Main Document |
| `reviewer_response_round2.pdf` | Point-by-point response to reviewers | Supporting File |

## Changes Made in This Revision

### Reviewer 1 Comments Addressed:
1. **Computational Results**: Added new "Computational Efficiency" subsection with training times (~10-15 min) and prediction throughput details
2. **Deep Learning Comparison**: Added paragraph on MobileNetV2/EfficientNet as complementary approaches; added citations
3. **Method Weaknesses**: Expanded limitations section with method-specific weaknesses (monthly composites, interpolation assumptions, cloud sensitivity)

### Reviewer 2 Comments Addressed:
1. **Feature Selection & Parameters**: Expanded explanation of variance threshold (0.5), why 30 features chosen, LightGBM hyperparameters (early stopping, boosting rounds)
2. **Ground Truthing**: Expanded field validation section with photo-based verification details; acknowledged lack of independent follow-up visits as limitation

## Compilation Commands

```bash
# Compile manuscript (from writeup/ directory)
cd /home/mmann1123/Documents/github/YM_TZ_crop_classifier/writeup
pandoc writeup.md --template=mytemplate.tex -o "Revisions 2/output_JSTARS-2025-00807.pdf" --bibliography=refs.bib --pdf-engine=xelatex

# Compile response letter (from writeup/ directory)
pandoc "Revisions 2/reviewer_response_round2.md" --template=mytemplate.tex -o "Revisions 2/reviewer_response_round2.pdf" --bibliography=refs.bib --pdf-engine=xelatex
```

## Submission Link

https://ieee.atyponrex.com/submission/submissionBoard/REX-PROD-2-86CF0D8D-95CD-4891-A8E6-A65788D12A8F-B87126E9-7ACA-4929-8AF0-F3475413B638-42267/current?idtype=external

## Deadline Note

J-STARS policy requires minor revision manuscripts be returned within 14 days.
