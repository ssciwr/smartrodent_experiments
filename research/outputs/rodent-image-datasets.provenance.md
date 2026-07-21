# Provenance: Published rodent / small-mammal image datasets

- **Date:** 2026-07-21
- **Rounds:** 2 direct research rounds after plan approval: initial dataset catalog + supplemental peer-reviewed literature pass
- **Sources consulted:** 30+ source/search items, including web searches, official dataset pages, dataset cards, repository/HTML pages, and peer-reviewed publisher pages.
- **Sources accepted:**
  - Nature / Scientific Data Qilian article: https://www.nature.com/articles/s41597-025-06105-2
  - Qilian Zenodo Part 1: https://doi.org/10.5281/zenodo.15730802
  - Qilian Zenodo Part 2: https://doi.org/10.5281/zenodo.15733158
  - NINA IPT small rodent monitoring: https://ipt.nina.no/resource?r=rodent_2025
  - GBIF small rodent monitoring: https://www.gbif.org/dataset/13101e81-bc62-4553-9fd9-c5c8eb3fb9ab
  - LILA BC Ohio Small Animals: https://lila.science/datasets/ohio-small-animals/
  - LILA BC California Small Animals: https://lila.science/datasets/california-small-animals/
  - LILA BC Channel Islands Camera Traps: https://lila.science/datasets/channel-islands-camera-traps/
  - LILA BC Trail Camera Images of New Zealand Animals: https://lila.science/datasets/nz-trailcams/
  - LILA BC Oregon Critters: https://lila.science/datasets/oregon-critters/
  - Zenodo iRodent: https://zenodo.org/records/8250392 ; JSON export inspected at https://zenodo.org/records/8250392/export/json
  - SAWIT GitHub: https://github.com/dtnguyen0304/sawit
  - SAWIT paper DOI: https://doi.org/10.1007/s11042-023-16673-3
  - Denmark GBIF DOI: https://doi.org/10.15468/t7827g
  - COAT Varanger pilot: https://data.coat.no/dataset/v_rodents_cameratraps_pilot_v1
  - COAT Varanger metadata: https://data.coat.no/dataset/v_rodents_cameratraps_image_metadata_lemming_blocks_v5
  - Hugging Face IDLE-OO Camera Traps: https://huggingface.co/datasets/imageomics/IDLE-OO-Camera-Traps
  - PLOS ONE MouseCam paper: https://doi.org/10.1371/journal.pone.0309252
  - EDI Virginia MouseCam dataset DOI: https://doi.org/10.6073/pasta/34c3f0f50968ec86df734239f7f2b2a5
  - California Fish and Wildlife small mammal selfie-trap article: https://doi.org/10.51492/cfwj.111.8
  - Littlewood et al. peatland restoration camera trap article: https://doi.org/10.1007/s10344-020-01449-z
  - European Journal of Wildlife Research non-native shrew camera-trap article: https://doi.org/10.1007/s10344-026-02052-4
  - Fink & Jachowski camera-trap design comparison: https://doi.org/10.1007/s13364-025-00780-7
- **Sources rejected or downgraded:**
  - Lincoln `Small mammals camera trap data 2025` DOI `10.24385/lincoln.31850983.v1`: `fetch_content` could not extract readable page content, so it was not used as a primary recommendation.
  - Böhner et al. 2023 DOI `10.1016/j.ecoinf.2023.102150`: DOI fetch hit anti-bot/robot page; used only as contextual search result.
  - EDI Virginia MouseCam DOI: direct fetch reached login page, but PLOS data-availability statement and search metadata support its existence; marked as not directly inspected.
  - SAWIT was accepted only as a weak/detection-only source because source labels are broad categories, not rodent species.
  - Oregon Critters was accepted only as detection/coarse-label source because the source says mice/voles/moles/shrews are labeled `small mammal`.
- **Verification:** PASS WITH NOTES
- **Blocked / degraded checks:**
  - `memory_remember` was not visible, so the plan could not be stored in persistent memory.
  - One LILA-specific web search failed with an Exa MCP rate-limit error.
  - `alpha_search` failed with `fetch failed`; no alpha-backed paper-search result was used. Supplemental peer-reviewed coverage used web/publisher pages.
  - No PDFs or image archives were fetched or parsed, per workflow constraint.
  - COAT image-file availability, Denmark media-file access, New Zealand exact mouse/rat species mapping, Qilian Zenodo data license, direct EDI archive access, and iRodent per-image species-label recoverability remain open checks.
- **Plan:** outputs/.plans/rodent-image-datasets.md
- **Research files:**
  - outputs/.drafts/rodent-image-datasets-research-direct.md
  - outputs/.drafts/rodent-image-datasets-draft.md
  - outputs/.drafts/rodent-image-datasets-cited.md
  - outputs/.drafts/rodent-image-datasets-verification.md
- **Final file:** outputs/rodent-image-datasets.md

## Correction: iRodent species-label status

After user challenge, I re-read the iRodent Zenodo landing page and JSON export. The page lists 10 rodent species at dataset level, but describes the annotations as manually labeled pose keypoints and generated segmentation masks. I did not download the 511.3 MB tarball to inspect annotation internals. Therefore the report was corrected: iRodent is **not recommended for species-label work** unless per-image species labels are separately verified or reconstructed from iNaturalist provenance/archive internals.
