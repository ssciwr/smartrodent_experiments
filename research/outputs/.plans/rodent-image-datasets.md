# Plan: Rodent Image Datasets

## Key Questions
1. Which published image datasets contain rodents broadly construed by the user: rats, mice, shrews, voles, and closely related small mammals?
2. Which datasets are from camera traps, drift-fence/downward-facing traps, tunnel cameras, or similar ecological monitoring settings?
3. Which datasets provide species names directly as labels, and which only provide coarser labels from which species names may be recoverable via metadata, taxonomy mapping, location/project context, or source records?
4. What are the practical access details for each dataset: URL, host, license/access terms, annotation format, image count, taxa/label count, and download mechanism?
5. Which datasets are best suited for species-level rodent classification versus detection, segmentation, or ecological abundance monitoring?

## Evidence Needed
- Official dataset pages, repositories, or data archives for published image collections.
- Dataset cards/README files for Hugging Face, LILA BC, Zenodo, GBIF/IPT, Figshare, Dryad, Kaggle, or institutional archives.
- Paper abstracts or HTML landing pages when datasets are described in publications.
- Metadata fields or label descriptions showing whether species names are present or recoverable.
- License/access statements from official pages.

## Scale Decision
**Direct-search research mode; no subagents.**

Rationale: The request is a focused dataset-discovery task. It requires several targeted searches and source checks, but not enough breadth to justify researcher subagents. I will use direct web search and dataset/source fetching, avoid PDF parsing unless unavoidable, and record exact search terms and notes in `outputs/.drafts/rodent-image-datasets-research-direct.md`.

## Task Ledger
| # | Task | Owner | Status |
|---|------|-------|--------|
| 1 | Search camera-trap and small-mammal monitoring datasets | Lead/direct | TODO |
| 2 | Search dataset platforms for rodent/small-mammal image datasets | Lead/direct | TODO |
| 3 | Search scholarly/source pages for rodent image corpora and camera-trap data descriptors | Lead/direct | TODO |
| 4 | Fetch and inspect official pages for candidate datasets | Lead/direct | TODO |
| 5 | Write direct research notes with exact search terms | Lead/direct | TODO |
| 6 | Draft dataset catalog and recommendations | Lead/direct | TODO |
| 7 | Add inline citations and Sources section | Lead/direct | TODO |
| 8 | Self-review citation support and source reachability | Lead/direct | TODO |
| 9 | Deliver final report and provenance sidecar | Lead/direct | TODO |

## Verification Log
| Step | Check | Result |
|------|-------|--------|
| Plan | `outputs/.plans/rodent-image-datasets.md` exists | PENDING |
| Evidence | Exact search terms recorded | PENDING |
| Evidence | Candidate datasets linked to official URLs | PENDING |
| Draft | Critical counts/labels/license claims mapped to sources | PENDING |
| Citation | Reachable URLs checked where practical | PENDING |
| Delivery | Required artifacts exist on disk | PENDING |

## Decision Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-21 | Slug set to `rodent-image-datasets` | Short, lowercase, hyphenated, topic-specific |
| 2026-07-21 | Direct-search mode selected | Focused dataset discovery can be handled without subagents |
| 2026-07-21 | Avoid PDF parsing by default | User workflow explicitly requests avoiding crash-prone PDF parsing |
| 2026-07-21 | Include shrews despite not being rodents taxonomically | User explicitly listed shrews in target scope |

## Capability Notes
- `memory_remember` is not visible in the current tool set, so the plan was not saved to persistent memory.

## Progress Update After Approval

### Task Ledger Update
| # | Task | Owner | Status |
|---|------|-------|--------|
| 1 | Search camera-trap and small-mammal monitoring datasets | Lead/direct | DONE |
| 2 | Search dataset platforms for rodent/small-mammal image datasets | Lead/direct | DONE |
| 3 | Search scholarly/source pages for rodent image corpora and camera-trap data descriptors | Lead/direct | DONE WITH NOTES: alpha_search failed; web/source search used |
| 4 | Fetch and inspect official pages for candidate datasets | Lead/direct | DONE WITH NOTES: Lincoln DOI extraction failed; PDFs not fetched |
| 5 | Write direct research notes with exact search terms | Lead/direct | DONE: outputs/.drafts/rodent-image-datasets-research-direct.md |
| 6 | Draft dataset catalog and recommendations | Lead/direct | DONE |
| 7 | Add inline citations and Sources section | Lead/direct | DONE |
| 8 | Self-review citation support and source reachability | Lead/direct | DONE WITH NOTES |
| 9 | Deliver final report and provenance sidecar | Lead/direct | IN PROGRESS |

### Verification Log Update
| Step | Check | Result |
|------|-------|--------|
| Plan | `outputs/.plans/rodent-image-datasets.md` exists | PASS |
| Evidence | Exact search terms recorded | PASS |
| Evidence | Candidate datasets linked to official URLs | PASS WITH NOTES |
| Draft | Critical counts/labels/license claims mapped to sources | PASS WITH NOTES |
| Citation | Reachable URLs checked where practical | PASS WITH NOTES |
| Delivery | Required artifacts exist on disk | PENDING FINAL STAT CHECK |

### Blockers / Degraded Capabilities
- `memory_remember` not visible; plan not saved to persistent memory.
- One LILA-specific `web_search` failed due Exa rate limit.
- `alpha_search` failed with `fetch failed`.
- Lincoln `Small mammals camera trap data 2025` DOI page extraction failed; it was not used as a primary recommended dataset.
- No PDFs or large image archives were downloaded or parsed.

### Final Delivery Update
- Final report copied to `outputs/rodent-image-datasets.md`.
- Provenance sidecar written to `outputs/rodent-image-datasets.provenance.md`.
- Final artifact verification completed with `stat` and `diff`.

### Supplemental Peer-Reviewed Literature Pass Update
- User requested additional peer-reviewed literature coverage after initial delivery.
- Ran 4 supplemental web searches targeting peer-reviewed small-mammal camera-trap literature.
- Added Virginia MouseCam / EDI as an additional high-priority dataset candidate, with EDI direct access marked as blocked by login.
- Added peer-reviewed method/evidence section covering Dueser et al. 2025, Clucas & McCluskey 2025, Littlewood et al. 2021, Smith et al. 2026, and Fink & Jachowski 2025.
- Updated final report, research notes, verification, and provenance.

### Correction: iRodent
- User correctly challenged iRodent as a species-label dataset.
- Re-read Zenodo landing page/JSON export: dataset-level species list exists, but advertised annotations are keypoints and generated masks, not species-class labels.
- Corrected report to demote iRodent and mark per-image species labels as unverified/not recommended.
