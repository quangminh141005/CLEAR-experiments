Numeric reference data transcribed from the publisher's supplementary information,
Tables S9–S10 (PadChest, 55), S11 (Indiana, 30), S13 (CheXpert supervised, 13).
Source inspected 2026-10-19:
https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41551-026-01741-4/MediaObjects/41551_2026_1741_MOESM1_ESM.pdf

Article: https://doi.org/10.1038/s41551-026-01741-4 (Han et al., 2026).
The article is CC BY 4.0; numerical values are factual reference data. Preserve attribution.
AUROC and CI endpoints retain the published three-decimal precision. N is positive
finding support, NOT dataset size. CheXpert support is absent in S13 and is left blank.
The exceptionally narrow supervised CIs are quoted as printed; they must not be
interpreted as equivalent to our patient-cluster sampling uncertainty.

Canonical `atelectasis` maps to the paper's `pulmonary atelectasis` in Indiana.
For PadChest the PDF text extraction says `descendent` and `goiter`; retain those
spellings in the reference, with explicit aliases for native labels such as
`descendent aortic elongation` and `goiter`. No broad parent/child ontology merging
is applied (e.g. pulmonary emphysema and emphysema remain distinct Indiana targets).
The default prompts use the exact `paper_finding` lowercased, not a renamed canonical label.

The comparison module computes macro AUROC from these rounded per-class values and
labels it accordingly, rather than treating an approximate headline as exact.
Patient/cohort, prompt recipe, annotation provenance and preprocessing differences
prevent automatic claims of exact reproduction, even if class counts happen to match.
