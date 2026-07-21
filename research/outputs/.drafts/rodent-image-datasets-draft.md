# Published Rodent / Small-Mammal Image Datasets

## Executive summary

The strongest camera-trap or camera-trap-like datasets for species-labeled rodents/small mammals are:

1. **Qilian Mountains desert-steppe camera-trap dataset** — best explicitly rodent-focused dataset found: the Scientific Data article reports 553,814 raw images, 28,881 validated wildlife images, 3,828 independent events, 26 vertebrate species, species-level annotations, and rodent identification supported by live-trapping references.[^qilian]
2. **NINA / GBIF Norwegian alpine tundra small-rodent monitoring** — a large tunnel-camera resource. The IPT page reports 734,408 media and 734,409 observations; the GBIF landing page reports 306,997 occurrence records and lists taxa including *Lemmus lemmus*, Cricetidae, and *Sorex* spp.[^nina-ipt][^nina-gbif]
3. **LILA BC Ohio Small Animals** — practical computer-vision dataset: 118,554 AHDriFT/downward-facing drift-fence camera-trap images with 45 species labels, including 14,169 meadow vole images.[^ohio]
4. **Denmark “Out of Sight out of Mind” small-mammal camera-trap dataset** — more than 150,000 photos from 202 modified Mostela-style camera boxes/tunnels, annotated to species level if possible; I verified the GBIF metadata page but not the underlying media-file access path.[^denmark]
5. **COAT Varanger small-rodent camera-trap datasets** — useful Arctic rodent/shrew monitoring metadata and classification files with scientific names recoverable; original image-file availability was not verified in this run.[^coat-pilot][^coat-meta]

Datasets that are useful but weaker for species-name recovery include **California Small Animals** (large drift-fence camera-trap dataset but many rodent labels are above species level), **Channel Islands Camera Traps** (many rodent/rat images but coarse label), **New Zealand Trail Camera Images** (huge, with mouse and rat labels, but exact species may require category/taxonomy mapping), **Oregon Critters** (explicitly collapses mice/voles/moles/shrews to “small mammal”), and **SAWIT** (camera-trap small-mammal detections but broad categories only).[^california][^channel][^nz][^oregon][^sawit]

## Dataset catalog

| Dataset | Best fit for this request | Setting | Scale | Rodent/small-mammal labels | Label recoverability | Access/license notes |
|---|---:|---|---:|---|---|---|
| Qilian Mountains rodents and sympatric vertebrates | Excellent | Infrared camera traps in Chinese desert steppe | 553,814 raw; 28,881 valid species images; 3,828 independent events | Species-level vertebrate annotations; rodent focus includes *Rhombomys opimus* and *Meriones meridianus* | Direct species metadata | Zenodo data DOIs; article is open access CC BY-NC-ND[^qilian] |
| NINA / GBIF Norwegian alpine tundra small-rodent monitoring | Very good | Tunnel camera traps | IPT: 734,408 media; GBIF: 306,997 occurrences | *Lemmus lemmus*, Cricetidae, *Sorex* spp., plus predators/birds | Mixed: species for lemming; family/genus group for others | CC-BY 4.0 data/media[^nina-ipt][^nina-gbif] |
| Ohio Small Animals / LILA BC | Very good | AHDriFT downward-facing drift-fence camera traps | 118,554 images | 45 species; meadow vole is common | Direct species labels | COCO Camera Traps; CDLA permissive; zip/cloud folders[^ohio] |
| Denmark Out of Sight out of Mind | Very good, pending access check | Modified Mostela-style camera boxes/tunnels | >150,000 photos | Small mammals; manually annotated to species if possible, otherwise genus | Direct or genus-level | GBIF metadata DOI; image/media access needs deeper check[^denmark] |
| COAT Varanger pilot + long-term metadata/classification | Good, pending image access check | Small-mammal camera traps, Varanger peninsula | Not counted from page | *Lemmus lemmus*, *Microtus oeconomus*, *Myodes rufocanus*, *Myodes rutilus*, *Sorex* sp. | Species names in metadata/classification; original images not verified | CC-BY 4.0 for long-term metadata page[^coat-pilot][^coat-meta] |
| iRodent / Zenodo | **Not recommended for species-label work** | iNaturalist-derived, non-camera-trap | 443 images | Zenodo landing page lists 10 rodent species, but annotations are described as pose keypoints plus generated segmentation masks | Species names are dataset-level/source-description metadata, not verified per-image species labels in the annotation files | COCO-format pose/segmentation dataset; Apache 2.0; tarball not downloaded[^irodent] |
| California Small Animals / LILA BC | Useful but coarse | Downward-facing drift-fence camera traps | 2,278,071 images | mouse species, Arvicolinae, rodent | Often not species-level; maybe limited recovery via taxonomy/context | COCO Camera Traps; CC-BY 4.0[^california] |
| New Zealand Trail Camera Images / LILA BC | Useful but needs mapping | Trail cameras across NZ | ~2.5M images | mouse and rat are top categories | Exact species may require category list/taxonomy mapping | COCO Camera Traps; CDLA permissive[^nz] |
| Channel Islands Camera Traps / LILA BC | Detection, not species classification | Camera traps, Channel Islands | 246,529 images | rodent class, likely rats by project context | Coarse; species not direct | COCO Camera Traps; CDLA permissive[^channel] |
| Oregon Critters / LILA BC | Detection, not target species labels | Trail + ground-facing camera traps | 99,909 images | mice/voles/moles/shrews collapsed to small mammal | Not species-level for target taxa | COCO Camera Traps; CDLA permissive[^oregon] |
| SAWIT | Detection only | Camera traps in realistic conditions | 34,434 images; 34,820 animals | broad “small mammal” category | Species names not available from README | GitHub + Springer paper[^sawit] |
| IDLE-OO Camera Traps / Hugging Face | Small benchmark | Curated LILA BC camera-trap test sets | 2,586 images total; Ohio subset 468 | scientific_name fields; Ohio subset has 39 species | Direct scientific_name for subset | CDLA permissive compilation[^idle] |

## Practical recommendations

### If you need a camera-trap dataset with direct rodent species names
Start with **Qilian Mountains** and **Ohio Small Animals**. Qilian is explicitly rodent-focused and species-level; Ohio is easy to access through LILA BC and includes a large meadow-vole class.[^qilian][^ohio]

### If you need larger ecological monitoring data for voles/lemmings/shrews
Use **NINA/GBIF Norwegian alpine tundra** and inspect **COAT Varanger**. These are closer to long-term monitoring datasets, but label granularity is mixed and some taxa are grouped.[^nina-ipt][^nina-gbif][^coat-meta]

### If you need a quick benchmark or small labeled corpus
Use **IDLE-OO Camera Traps** for camera-trap species classification. Do **not** use **iRodent** as a species-label benchmark unless you first verify or reconstruct per-image species labels from its iNaturalist provenance or archive internals; the public landing page describes pose keypoints and generated segmentation masks, not species-class annotations.[^idle][^irodent]

### Datasets to treat cautiously for species-level work
California Small Animals, Channel Islands, Oregon Critters, and SAWIT are useful for small-mammal detection or coarse rodent presence but are not ideal if the task requires mouse/rat/vole/shrew species names directly.[^california][^channel][^oregon][^sawit] New Zealand Trail Camera Images is potentially useful for rats/mice, but exact species recovery should be checked from its category file and taxonomy mapping before training a species-level classifier.[^nz]

## Supplemental peer-reviewed literature pass

After the initial dataset-first catalog, I ran an additional literature-focused pass. This changed the prioritization in one important way: **the Virginia MouseCam / EDI dataset should be added to the high-priority list**. Dueser, Porter, and Moncrief’s peer-reviewed PLOS ONE article reports a low-cost “MouseCam” platform and gives a formal data-availability citation for **Camera detections of small mammals on the coast of Virginia, 2020–2023** in the Environmental Data Initiative repository.[^mousecam-plos][^mousecam-edi] The PLOS paper reports species-labeled detections of marsh rice rat, house mouse, brown rat, white-footed mouse, and discusses meadow vole non-detection in the study context; it also reports that over 78% of images contained recognizable small mammals.[^mousecam-plos] Direct `fetch_content` of the EDI DOI reached a login page, so I treat the EDI archive as **source-confirmed via PLOS plus search metadata, but not directly inspected**.

The literature pass also strengthens the main caution: peer-reviewed studies repeatedly show that specialized close-focus/tunnel/bucket camera traps can identify some small mammals to species, but this is not universal. Littlewood et al. identified wood mouse, bank vole, field vole, common shrew, and pygmy shrew using baited close-focus tunnels in a peatland restoration study.[^littlewood] Clucas and McCluskey detected 10 small mammal species in a coastal redwood forest, including western red-backed vole, western deermouse, fog shrew, and Trowbridge’s shrew, using a selfie-trap-style tube setup.[^clucas] A 2026 Springer study uploaded 50,328 camera-trap sequences to MammalWeb and retained 30,368 species-level small-mammal-containing sequences for analysis across seven species: wood mouse, bank vole, field vole, common shrew, pygmy shrew, water shrew, and greater white-toothed shrew.[^smith2026] Conversely, Fink and Jachowski’s comparison of Mostela, Small Mammal Box, and Baited Post designs found that mice, rats, shrews, and rabbits often had to be grouped, while distinctive taxa such as hispid cotton rat and woodland vole could be identified.[^fink]

### Added dataset candidate from the literature pass

| Dataset | Best fit for this request | Setting | Scale | Rodent/small-mammal labels | Label recoverability | Access/license notes |
|---|---:|---|---:|---|---|---|
| Virginia MouseCam / EDI “Camera detections of small mammals on the coast of Virginia, 2020–2023” | Very good, but direct archive inspection blocked | Low-cost bucket-style MouseCam camera traps | PLOS reports 2,629 Hog Island photos plus 3,454 mainland forest photos; EDI dataset covers 2020–2023 | Species observations include marsh rice rat, house mouse, brown rat, white-footed mouse; meadow vole discussed | Direct species observations per PLOS; EDI archive not directly inspected | PLOS gives EDI DOI; `fetch_content` reached login page for EDI DOI[^mousecam-plos][^mousecam-edi] |


## Caveats and disagreements

- **Shrews are not rodents taxonomically**, but they are included here because the request explicitly listed them.
- “Species labels” varies by dataset. Some pages say “primarily species level” while retaining labels such as `mouse`, `rat`, `rodent`, `small mammal`, `Cricetidae`, or `Sorex spp.`; those should not be treated as species labels without mapping or expert review.[^california][^nina-gbif][^oregon]
- I did not download the image archives or parse PDFs. Evidence comes from HTML landing pages, dataset cards, repository READMEs, and search-result snippets when page extraction failed.
- Several promising small-mammal camera-trap papers have datasets or supplementary code, but not necessarily public image files; they are lower priority unless you want method references rather than trainable image data. iRodent is similarly lower priority for this request: it is rodent imagery, but its advertised annotation task is pose/segmentation, not species classification.

## Open questions

1. What exact species are included under the New Zealand `mouse` and `rat` categories in the downloadable category list?
2. Are the original image files for COAT Varanger datasets publicly downloadable, or only metadata/classification tables?
3. What license is attached to the Qilian Zenodo image archives themselves, separately from the Nature article license?
4. Does the Denmark GBIF metadata dataset expose media files directly, or only metadata/occurrence records?
5. Does the iRodent tarball contain recoverable per-image species labels in filenames or metadata? The landing page alone does not establish this.

## Sources

[^qilian]: Wei, C., Ma, Y., Fan, Y. et al. “Camera Trap Dataset of Rodents and Sympatric Vertebrates in the Desert Steppe of Qilian Mountains China.” *Scientific Data* 12, 1831 (2025). https://www.nature.com/articles/s41597-025-06105-2 ; data DOIs: https://doi.org/10.5281/zenodo.15730802 and https://doi.org/10.5281/zenodo.15733158
[^nina-ipt]: Norwegian Institute for Nature Research IPT. “Camera trap based small rodent monitoring: Data from Norwegian alpine tundra.” https://ipt.nina.no/resource?r=rodent_2025
[^nina-gbif]: GBIF dataset page for “Camera trap based small rodent monitoring: Data from Norwegian alpine tundra,” DOI 10.15468/avmbdq. https://www.gbif.org/dataset/13101e81-bc62-4553-9fd9-c5c8eb3fb9ab
[^ohio]: LILA BC. “Ohio Small Animals.” https://lila.science/datasets/ohio-small-animals/
[^denmark]: GBIF. Worsøe Havmøller R, Nørgaard Konradsen S (2025). “Out of Sight out of Mind - Camera Traps to uncover the small mammal community of Denmark.” DOI 10.15468/t7827g. https://doi.org/10.15468/t7827g
[^coat-pilot]: COAT Data Portal. “V_rodents_cameratraps_pilot.” https://data.coat.no/dataset/v_rodents_cameratraps_pilot_v1
[^coat-meta]: COAT Data Portal. “V_rodents_cameratraps_image_metadata_lemming_blocks.” https://data.coat.no/dataset/v_rodents_cameratraps_image_metadata_lemming_blocks_v5
[^irodent]: Ye, S. et al. “iRodent: a keypoint and segmentation dataset of rodents in the wild.” Zenodo, DOI 10.5281/zenodo.8250392. https://zenodo.org/records/8250392
[^california]: LILA BC. “California Small Animals.” https://lila.science/datasets/california-small-animals/
[^channel]: LILA BC. “Channel Islands Camera Traps.” https://lila.science/datasets/channel-islands-camera-traps/
[^nz]: LILA BC. “Trail Camera Images of New Zealand Animals.” https://lila.science/datasets/nz-trailcams/
[^oregon]: LILA BC. “Oregon Critters.” https://lila.science/datasets/oregon-critters/
[^sawit]: Nguyen, T. T. T. et al. “SAWIT: A small-sized animal wild image dataset with annotations.” *Multimedia Tools and Applications* (2023), and GitHub README. https://github.com/dtnguyen0304/sawit ; https://doi.org/10.1007/s11042-023-16673-3
[^idle]: Hugging Face. imageomics/IDLE-OO-Camera-Traps dataset card. https://huggingface.co/datasets/imageomics/IDLE-OO-Camera-Traps

[^mousecam-plos]: Dueser, R. D., Porter, J. H., & Moncrief, N. D. (2025). “The continuing search for a better mouse trap: Two tests of a practical, low-cost camera trap for detecting and observing small mammals.” *PLOS ONE* 20(1): e0309252. https://doi.org/10.1371/journal.pone.0309252
[^mousecam-edi]: Porter, J. H. & Dueser, R. D. (2024). “Camera detections of small mammals on the coast of Virginia, 2020–2023.” Environmental Data Initiative. https://doi.org/10.6073/pasta/34c3f0f50968ec86df734239f7f2b2a5
[^clucas]: Clucas, B. & McCluskey, S. L. (2025). “Camera trap method effectively identifies small mammal species in forested habitats.” *California Fish and Wildlife Journal* 111:e8. https://doi.org/10.51492/cfwj.111.8
[^littlewood]: Littlewood, N. A., Hancock, M. H., Newey, S., Shackelford, G., & Toney, R. (2021). “Use of a novel camera trapping approach to measure small mammal responses to peatland restoration.” *European Journal of Wildlife Research* 67:12. https://doi.org/10.1007/s10344-020-01449-z
[^smith2026]: Smith et al. (2026). “Camera trapping for small mammals: the case of a non-native shrew.” *European Journal of Wildlife Research*. https://doi.org/10.1007/s10344-026-02052-4
[^fink]: Fink, M. & Jachowski, D. (2025). “Comparison of three camera trap designs for sampling small mammals.” *Mammal Research* 70:1–8. https://doi.org/10.1007/s13364-025-00780-7
