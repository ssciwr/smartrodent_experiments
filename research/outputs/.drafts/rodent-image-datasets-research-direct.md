# Direct Research Notes: Rodent Image Datasets

## Search terms used
1. `published rodent camera trap image dataset species labels mice voles shrews rats`
2. `site:lila.science small animals camera trap mouse vole shrew rat species labels dataset`
3. `Hugging Face Zenodo rodent image dataset species labels iRodent camera trap small mammal`
4. `small mammal tunnel camera trap dataset lemming vole shrew images species labels GBIF`
5. `"Small mammals camera trap data 2025" "31850983" species voles mice shrews`
6. alpha_search query: `published camera trap image datasets rodents mice voles shrews species labels dataset small mammals` — failed with `fetch failed`.

## Candidate datasets and extracted facts

### Qilian Mountains / Scientific Data + Zenodo
- URL: https://www.nature.com/articles/s41597-025-06105-2
- Data DOIs: https://doi.org/10.5281/zenodo.15730802 and https://doi.org/10.5281/zenodo.15733158
- Published: Scientific Data 2025; article DOI 10.1038/s41597-025-06105-2.
- Setting: 22 infrared-triggered camera traps in desert steppe habitat, Qilian Mountains National Park, China, Dec 2020-Jan 2022.
- Scale: 553,814 images, 8,052 trap-days; 28,881 valid species images; 3,828 independent detection events; 26 vertebrate species.
- Rodent relevance: explicitly motivated by Rhombomys opimus and Meriones meridianus; species-level metadata; live trapping used as reference for visually similar rodents.
- Access/license: Nature article CC BY-NC-ND 4.0; data on Zenodo (dataset license not separately fetched; verify on Zenodo if license matters).

### NINA / GBIF Norwegian alpine tundra small rodent monitoring
- URL: https://ipt.nina.no/resource?r=rodent_2025 and https://www.gbif.org/dataset/13101e81-bc62-4553-9fd9-c5c8eb3fb9ab
- DOI: https://doi.org/10.15468/avmbdq
- Setting: tunnel camera traps for small rodents in Norwegian alpine tundra; 199 deployments; 2023-06-18 to 2025-09-30.
- Scale: IPT says 734,409 observations and 734,408 media; GBIF page says 306,997 occurrence records.
- Labels/taxa: Aves, Cricetidae, Lemmus lemmus, Mustela erminea, Mustela nivalis, Sorex spp.; small rodent/lemming/vole/camera trap keywords.
- License: CC-BY-4.0 for data and media.
- Caveat: Some records are not species-level (`Cricetidae`, `Sorex spp.`).

### Denmark small mammal camera traps / GBIF metadata dataset
- URL/DOI: https://doi.org/10.15468/t7827g
- Dataset: Out of Sight out of Mind - Camera Traps to uncover the small mammal community of Denmark.
- Setting: 202 camera traps in 13 Danish study areas, modified Mostela-style boxes/tunnels, 2021-2024.
- Scale: more than 150,000 photos after blanks/setup cleanup.
- Labels: manually annotated to species level if possible, otherwise genus level.
- Caveat: Tool content was a GBIF metadata page; direct image/media file access details were not deeply inspected.

### COAT Varanger pilot and long-term small rodent camera trap datasets
- URLs: https://data.coat.no/dataset/v_rodents_cameratraps_pilot_v1 and https://data.coat.no/dataset/v_rodents_cameratraps_image_metadata_lemming_blocks_v5
- Pilot DOI: 10.48425/0077746; long-term metadata DOI: 10.48425/cj2a022d.
- Setting: small mammal camera traps on the Varanger peninsula, Norway.
- Labels/taxa: Lemmus lemmus, Microtus oeconomus, Myodes rufocanus, Myodes rutilus, Sorex sp, Mustela species.
- Data: pilot includes image metadata and classification text files; long-term page includes image metadata and points to classification/manual classification datasets.
- License: long-term metadata page says CC-BY_4.0.
- Caveat: these pages emphasize metadata/classification tables; I did not verify public availability of the original image files themselves.

### LILA BC Ohio Small Animals
- URL: https://lila.science/datasets/ohio-small-animals/
- Setting: AHDriFT downward-facing drift-fence camera traps in Ohio.
- Scale: 118,554 images; 168 unique location IDs.
- Labels: 45 species; most common include meadow vole (14,169 images), Eastern garter snake, song sparrow, blank.
- Format/access: COCO Camera Traps; 28GB zip and GCP/AWS/Azure cloud folders.
- License: CDLA permissive.

### LILA BC California Small Animals
- URL: https://lila.science/datasets/california-small-animals/
- Setting: downward-facing short-focus Reconyx cameras with drift fence, California.
- Scale: 2,278,071 images.
- Labels: include small mammals, reptiles, amphibians; most common categories include `mouse species` (252,986), `arvicolinae subfamily` (124,342), `rodent` (112,248), plus blanks.
- Format/access/license: COCO Camera Traps; CC-BY 4.0; cloud folders and metadata.
- Caveat: many rodent labels are above species level; species names likely not recoverable from image labels alone.

### LILA BC Channel Islands Camera Traps
- URL: https://lila.science/datasets/channel-islands-camera-traps/
- Setting: 73 camera locations, Channel Islands, California.
- Scale: 246,529 images.
- Labels: `rodent` 82,914 images; fox, bird, skunk, other; 114,949 empty. All animals have bounding boxes.
- Caveat: rodent label is coarse; source notes all rat images were on islands known to have rat populations, but species-level rat labels are not given on the LILA page.

### LILA BC New Zealand Trail Camera Images
- URL: https://lila.science/datasets/nz-trailcams/
- Setting: ~2.5M trail-camera images across New Zealand.
- Labels: 97 categories, primarily species-level; most common are mouse (49%), possum (6.7%), rat (5.5%).
- Format/access/license: COCO Camera Traps; CDLA permissive; cloud storage; species information in folder names.
- Caveat: `mouse` and `rat` labels may need category file/taxonomy mapping for exact species.

### LILA BC Oregon Critters
- URL: https://lila.science/datasets/oregon-critters/
- Scale: 99,909 images, bounding boxes/class labels for 91,045.
- Setting: western Oregon biodiversity camera traps, including paired trail- and ground-facing cameras.
- Labels: mix of species and higher taxa; page explicitly says mice, voles, moles, and shrews are all labeled `small mammal`.
- Suitability: useful for small-mammal detection but not species-level target labels for mice/voles/shrews.

### iRodent / Zenodo
- URL: https://zenodo.org/records/8250392
- DOI: 10.5281/zenodo.8250392.
- Setting/source: iNaturalist API observations, not camera traps.
- Scale: 443 images.
- Labels/species: Muskrat, Brown Rat, House Mouse, Black Rat, Hispid Cotton Rat, Meadow Vole, Bank Vole, Deer Mouse, White-footed Mouse, Striped Field Mouse.
- Format/access/license: COCO format; 511.3 MB tarball; Apache 2.0.
- Suitability: species-labeled rodent images and pose/segmentation annotations; not camera-trap imagery.

### SAWIT
- URL: https://github.com/dtnguyen0304/sawit and paper DOI https://doi.org/10.1007/s11042-023-16673-3
- Setting: camera traps, realistic wild conditions.
- Scale: 34,434 images and 34,820 annotated animals.
- Labels: seven broad categories: frog, lizard, bird, small mammal, medium/big mammal, spider, scorpion.
- Suitability: weak for user request because rodent/small mammal species names are not provided in the README.

### IDLE-OO Camera Traps / Hugging Face
- URL: https://huggingface.co/datasets/imageomics/IDLE-OO-Camera-Traps
- Setting: balanced benchmark curated from LILA BC camera-trap sources.
- Scale: 2,586 total images. Ohio Small Animals subset has 468 images, 39 species, 12 images/species, scientific_name used.
- Suitability: small benchmark; useful for quick species-classification tests, not a primary rodent corpus.

## Blocked / degraded evidence
- One web_search call for LILA-specific query failed due Exa MCP rate limit; other queries succeeded.
- alpha_search failed with `fetch failed`; scholarly coverage was gathered from web search and HTML/source pages instead.
- Lincoln `Small mammals camera trap data 2025` DOI page could not be extracted via fetch_content; only search snippets were used, so it is not included as a primary recommended dataset.
- No PDFs were fetched or parsed, per workflow instruction.

## Supplemental peer-reviewed literature pass (requested after initial delivery)

### Search terms used
7. `peer reviewed small mammal camera trap dataset mice voles shrews species identification images`
8. `rodent camera trap dataset species identification images peer reviewed rodents camera traps machine learning`
9. `"camera trapping" "small mammals" "dataset" "species" images voles shrews mice`
10. `"MouseCam" small mammal camera trap species dataset images mice shrews voles PLOS One`

### Literature sources found and relevance

#### Dueser, Porter & Moncrief 2025, PLOS ONE — MouseCam
- Peer-reviewed open-access article: DOI 10.1371/journal.pone.0309252.
- Data availability statement points to EDI dataset: Porter & Dueser 2024, `Camera detections of small mammals on the coast of Virginia, 2020-2023`, DOI 10.6073/pasta/34c3f0f50968ec86df734239f7f2b2a5.
- Species detected/illustrated include marsh rice rat, house mouse, brown rat, white-footed mouse; article also discusses meadow vole non-detection on Hog Island.
- This is a strong addition because it explicitly provides a repository dataset with image records/species observations. Fetching the EDI DOI landed on a login page, but the PLOS data-availability statement and search result described the dataset.

#### Clucas & McCluskey 2025, California Fish and Wildlife Journal — selfie trap in redwood forest
- Peer-reviewed journal article: DOI 10.51492/cfwj.111.8.
- Detected 10 small mammal species over 30 camera trap sites, 240 survey nights, 5,534 videos.
- Species include western red-backed vole, western deermouse, fog shrew, Trowbridge’s shrew, plus other small mammals.
- Strong as peer-reviewed evidence that species-level small mammal identification is feasible; public image dataset/repository not found in this pass.

#### Littlewood et al. 2021, European Journal of Wildlife Research — peatland restoration
- Peer-reviewed article: DOI 10.1007/s10344-020-01449-z.
- Adapted close-focus camera trap tunnel; 108 deployments, 216 trap nights, 3,872 triggers; 3,071 animal triggers.
- Recorded wood mouse, bank vole, field vole, common shrew, pygmy shrew.
- Useful literature support and example images; not found as standalone image dataset in this pass.

#### Smith et al. 2026, European Journal of Wildlife Research — non-native shrew / bait comparison
- Peer-reviewed Springer article: DOI 10.1007/s10344-026-02052-4.
- 50,328 sequences uploaded to MammalWeb; 30,368 species-level small mammal-containing sequences used in analysis; seven small mammal species identified: wood mouse, bank vole, field vole, common shrew, pygmy shrew, water shrew, greater white-toothed shrew.
- Images uploaded to MammalWeb; public downloadable image dataset status not verified.

#### Fink & Jachowski 2025, Mammal Research — comparison of three camera trap designs
- Peer-reviewed open-access article: DOI 10.1007/s13364-025-00780-7.
- Compared Mostela, Small Mammal Box, and Baited Post at 120 sites in South Carolina.
- Species-level ID possible for larger or distinctive small mammals; mice, rats, shrews, rabbits often grouped. Hispid cotton rat and woodland vole were identifiable.
- Useful cautionary evidence: species-level recovery is not automatic for small mammals, especially Peromyscus/mice/shrews.

#### Böhner et al. 2023, Ecological Informatics — semi-automatic workflow
- Search result only because DOI fetch hit an anti-bot page.
- DOI 10.1016/j.ecoinf.2023.102150. Relevant because NINA/GBIF workflow says its images were processed using Böhner et al. (2023).
- Not used for additional counts beyond search snippet.

## Supplemental assessment
- The initial report underweighted peer-reviewed method papers and the Virginia MouseCam EDI/PLOS dataset.
- The dataset shortlist should be updated to include the Virginia MouseCam/EDI dataset as a strong peer-reviewed, species-labeled small-mammal camera dataset, although direct EDI landing-page access was blocked by login in `fetch_content`.
- Peer-reviewed literature supports the key caveat: species-level identification is feasible in some close-focus/tunnel/bucket settings, but morphologically similar small mammals often still require grouping, prior species knowledge, live-trapping/genetic calibration, or expert review.
