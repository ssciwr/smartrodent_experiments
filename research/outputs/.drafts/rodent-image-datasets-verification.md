# Verification: Rodent Image Datasets

## Checks performed

- Confirmed required artifacts exist on disk via `stat` in prior and final checks.
- Recorded original and supplemental exact search terms in `outputs/.drafts/rodent-image-datasets-research-direct.md`.
- Supplemental peer-reviewed pass used web_search and fetch_content on publisher/DOI pages; no PDFs were fetched.
- Added a supplemental section to the cited report with peer-reviewed sources and one additional dataset candidate: Virginia MouseCam / EDI.
- Checked that the final candidate contains the new `Supplemental peer-reviewed literature pass` section and the `Virginia MouseCam` entry.

## FATAL findings

None found in the self-review.

## MAJOR findings

1. **EDI archive direct inspection blocked.** The PLOS paper’s data availability statement cites the EDI dataset, and search metadata describes the dataset, but `fetch_content` for the DOI landed on an EDI login page. The report labels this as source-confirmed via PLOS plus search metadata, not directly inspected.
2. **Some peer-reviewed studies are method papers rather than downloadable image datasets.** Littlewood et al., Clucas & McCluskey, Smith et al. 2026, and Fink & Jachowski are used to calibrate feasibility/caveats rather than added as primary downloadable datasets unless a data repository was found.
3. **alpha_search remains blocked.** Shell and tool routes returned `fetch failed`; literature pass used web/publisher pages instead.

## MINOR findings

1. One web_search call hit an Exa rate limit; other searches succeeded.
2. DOI fetch for Böhner et al. 2023 hit an anti-bot page, so it is mentioned only as context and not used for detailed claims.

## Result

Verification: PASS WITH NOTES. The final report has been updated with a peer-reviewed literature supplement and preserves unresolved access checks explicitly.

## Correction check: iRodent

- User flagged that iRodent is a pose-estimation/segmentation dataset and may not contain species labels.
- Re-read Zenodo landing page and JSON export. Evidence: the page lists 10 species in the description, but the explicit annotation field says pose keypoints and generated segmentation masks.
- Corrected final report and drafts to mark iRodent as **not recommended for species-label work** unless per-image labels are verified/reconstructed.
- The 511.3 MB tarball was not downloaded; per-image archive internals remain unverified.
