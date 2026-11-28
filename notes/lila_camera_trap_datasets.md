# LILA camera-trap datasets with at least 10 animal classes

Research date: 2026-11-26

## Executive summary

LILA currently has a large and unusually varied camera-trap collection. For a
general-purpose, species-aware model, the strongest starting points are:

1. **WCS Camera Traps** for taxonomic and geographic breadth (about 675 species,
   12 countries, 1.4 million images, about 375,000 boxes, and recommended
   location-aware splits).
2. **SWG Camera Traps 2018-2020** for a large, coherent field dataset from the
   Annamites (120 categories, 982 locations, sequence metadata, and 101,659
   boxes).
3. **Snapshot Safari 2024 Expansion** for standardized multi-site African data
   (151 categories, 15 projects, 1,824 cameras, animal counts on 1.32 million
   labels).
4. **Oregon Critters** when bounding boxes and small-animal/large-animal camera
   modalities matter (91,045 of 99,909 images have manually drawn,
   class-labelled boxes, with published train/validation/test splits).
5. **Nkhotakota**, **Orinoquia**, or **Caltech Camera Traps** when a smaller,
   well-described dataset with spatial metadata is preferable.
6. **Indiana**, **California**, and **Ohio Small Animals** for downward-facing,
   drift-fence imagery of rodents, shrews, reptiles, and amphibians. These are
   much closer to the rodent-experiments project's target domain than ordinary
   trail-camera datasets.
7. **New Zealand Wildlife Thermal Imaging** for RGB-T-adjacent work. It is
   thermal video rather than paired RGB-T, but it supplies track-level labels,
   trajectories, locations, raw HDF thermal data, and a location-based split.

Class count alone is misleading. WCS has exceptional breadth but combines
image- and sequence-level labels; Snapshot labels are reliable at sequence
level, not necessarily on every frame; Idaho mixes animal and camera-state
classes; and Duck Pictures is more than 98% empty. Location-aware splitting is
essential to avoid measuring background recognition instead of animal
generalization.

## Scope and method

I screened the [official LILA dataset catalog](https://lila.science/datasets/)
and retained datasets that:

- were captured by autonomous camera traps/trail cameras (including
  downward-facing small-animal systems and thermal camera traps); and
- contain at least ten animal/taxon labels in the public annotations.

Drone, aerial, handheld, marine-video, individual-ID, and aggregation datasets
were not treated as camera-trap datasets. “Class count” below means the source
dataset's vocabulary, not necessarily a count of biological species. It can
include broad taxa, domestic animals, `unknown` groups, life stages, empty,
human, vehicle, or camera-state labels. Where LILA states an animal-only count,
that distinction is shown. The complete and current vocabulary is the
`categories` array in the official metadata linked from each LILA page. For
cross-dataset work, LILA publishes an [iNaturalist taxonomy mapping and a
unified image table](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/).

Sizes are download-planning figures, not selection criteria. Published zip
sizes can be much smaller than the unzipped cloud inventory. Approximate
inventory figures below come from LILA's public Source Cooperative object
catalog; entries marked `est.` were estimated from object-size samples,
stratified across locations or constituent projects where the storage layout
allowed it, because the newest collections were absent from that inventory.

## Recommended datasets, ranked by annotation utility

### Tier A: strongest general training/evaluation sources

| Dataset | Scale and classes | Annotation strengths | Important caveats | Provenance / license | Planning size |
|---|---|---|---|---|---|
| [WCS Camera Traps](https://lila.science/datasets/wcscameratraps) | ~1.4M images; ~675 species from 12 countries. Common labels: white-lipped peccary, ocellated turkey, cattle. | ~375k species-labelled boxes on ~300k images; sequences; recommended train/val/test splits; class-count table and full metadata on page. | ~50% empty. Some labels were assigned at sequence rather than image level; sequences are timestamp-inferred. Public human images removed. | Wildlife Conservation Society; CDLA-Permissive. | ~595 GB unzipped inventory. |
| [SWG Camera Traps 2018-2020](https://lila.science/datasets/swg-camera-traps) | 2,039,657 images in 436,617 sequences, 982 locations; 120 mostly species-level categories. Common: Eurasian wild pig, large-antlered muntjac, unidentified murid. | 101,659 boxes on 88,135 images; sequences and locations; full class/count table. | 12.98% empty. Human entries remain in metadata but images are removed. Large download. | Saola Working Group with Asian Arks, WCS Lao, WWF Lao, ICBF, and Vinh University; cite *Northern and Central Annamites Camera Traps 2.0*; CDLA-Permissive. | ~2.79 TB inventory. |
| [Snapshot Safari 2024 Expansion](https://lila.science/datasets/snapshot-safari-2024-expansion/) | 4,029,374 images from 15 African projects; 151 categories; 1,824 camera locations. Common: empty, impala, zebra. | Standardized Snapshot Safari protocols; animal counts for 1,322,468 labels; project and location identifiers. | 2,739,081 empty instances (~68%); substantial site/class imbalance. Some projects overlap the separately published legacy Snapshot sets. | Snapshot Safari / Smithsonian contact; CDLA-Permissive. | ~5.01 TB inventory. |
| [Oregon Critters](https://lila.science/datasets/oregon-critters/) | 99,909 images; 46 source categories. Common: *Catharus* spp., black-tailed deer, Townsend's chipmunk, Douglas squirrel. | Manually drawn class-labelled boxes on 91,045 images; 8,864 verified false triggers; train/val/test split; paired trail- and downward-facing cameras. | Mice, voles, moles, and shrews are pooled as `small mammal`; the vocabulary also contains broad groups and non-animal labels. | USFS Northwest Forest Plan monitoring plus H.J. Andrews, Elliott State Forest, Oregon Dunes, Siuslaw NF, and BLM studies; cite Appel et al.; CDLA-Permissive. | ~196 GB est. |
| [Nkhotakota Camera Traps](https://lila.science/datasets/nkhotakota-camera-traps/) | 321,562 images; 46 taxa; 164,139 empty. Includes small antelope species and broader fallback groups. | Per-image labels, animal counts, and 33,813 images with manual boxes. | Mixed taxonomic ranks; human images removed though entries remain. | Nkhotakota Wildlife Reserve, African Parks Network + Malawi DNPW, with USDA Forest Service technical cooperation; CDLA-Permissive. | ~338 GB inventory. |
| [Orinoquia Camera Traps](https://lila.science/datasets/orinoquia-camera-traps/) | 104,782 images, 50 cameras; 51 classes, mainly mammals. Examples: collared peccary, black agouti, paca, white-lipped peccary, tapir, giant anteater. | Expert review in Wildlife Insights; stratified-random design; unique locations; published study and analysis code. | ~20% empty; human records remain in metadata but images were removed. | El Rey Zamuro and Las Unamas reserves, Meta, Colombia, Jan-Jul 2020; cite Velez et al. (2023); CDLA-Permissive. | 72 GB zip; ~155 GB inventory. |
| [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps) | 243,100 images, 140 Southwestern US locations; 21 animal categories + empty. Common: opossum, raccoon, coyote. | ~66k boxes; location-aware benchmark splits; CCT20 downsized benchmark available. | ~70% empty; only regional North American fauna. | Caltech; cite Beery, Van Horn & Perona, ECCV 2018; CDLA-Permissive. | 105 GB zip; ~112 GB inventory. |

### Tier A-specialist: particularly relevant modalities or taxa

| Dataset | Scale and classes | Annotation strengths | Important caveats | Provenance / license | Planning size |
|---|---|---|---|---|---|
| [Indiana Small Animals](https://lila.science/datasets/indiana-small-animals/) | 1.6M images, 302,582 detections/sequences, 102 sites; 73 taxa. Common non-empty: *Peromyscus* spp., small mammal, eastern chipmunk. | Sequence and site identifiers; lowest feasible taxonomic label; two-camera AHDriFT deployments per site; empty sequences downsampled to one frame. | Mammals received more identification effort than other taxa; `small mammal` and genus/group labels remain. Human images removed. | Purdue Fort Wayne AHDriFT survey; cite Lee (2026 MSc); CDLA-Permissive. | ~3.8 TB est. |
| [California Small Animals](https://lila.science/datasets/california-small-animals/) | 2,278,071 images; 256 source categories spanning small mammals, reptiles, and amphibians. Common: blank/misfire, mouse spp., Arvicolinae, rodent. | Standardized CDFW drift-fence/downward-camera protocol; Wildlife Insights labels; deployment and location IDs. | 1,320,489 blank/misfire images (~58%); many labels intentionally stop at higher taxonomic ranks, so 256 is not a species count. | California Department of Fish and Wildlife Cannabis and Sentinel Site projects; **CC BY 4.0**. | ~791 GB inventory. |
| [Ohio Small Animals](https://lila.science/datasets/ohio-small-animals/) | 118,554 AHDriFT images; 45 species. | Downward-facing drift-fence imagery; species tags and location IDs in COCO Camera Traps format. | Specialized close-focus viewpoint; confirm per-class support in metadata before building a 45-way classifier. | Ohio AHDriFT research / associated smart-camera publication; CDLA-Permissive. | ~61 GB inventory. |
| [New Zealand Wildlife Thermal Imaging](https://lila.science/datasets/new-zealand-wildlife-thermal-imaging/) | 121,190 heat-triggered thermal videos; 45 categories. Common non-false-positive: bird, rodent, possum. | Human-reviewed track labels; start/end frames and trajectories; camera IDs; recommended mostly location-disjoint split; background-subtracted and normalized MP4 plus raw HDF thermal data. | ~24k false-positive videos; tracks are machine-generated then labelled; not paired RGB-T. A few rare classes cross split locations. | The Cacophony Project, multiple New Zealand regions; CDLA-Permissive. | 33 GB ML zip; ~500 GB raw HDF (~677 GB full inventory). |
| [ENA24-detection](https://lila.science/datasets/ena24detection) | ~10,000 Eastern North American images; 23 classes. Common: American crow, American black bear, dog. | Bounding box on every image; compact detection dataset. | Small and geographically narrow; public human images removed. | ENA24 providers/citation on official page; CDLA-Permissive. | 3.6 GB zip; ~7.8 GB inventory. |

### Tier B: useful, but with stronger balance, granularity, or metadata caveats

| Dataset | Scale / class vocabulary | Best use and caveats | Provenance / license | Planning size |
|---|---|---|---|---|
| [Trail Camera Images of New Zealand Animals](https://lila.science/datasets/nz-trailcams) | ~2.5M images; 110 mostly species-level categories. Common: mouse (49%), possum (6.7%), rat (5.5%). | Excellent scale and rodent relevance; labels were mostly project-labelled then volunteer-verified. Strong class imbalance and heterogeneous projects/camera models. Location IDs are in metadata and EXIF. | Multiple New Zealand organizations; contact NZ Department of Conservation; CDLA-Permissive. | ~2.11 TB inventory. |
| [Biome Health Project Maasai Mara 2018](https://lila.science/datasets/biome-health-project-maasai-mara) | 37,075 images from 176 sites; 100 categories, mostly species with some groups such as `shoat`. | Clean non-empty multi-taxon subset with explicit train/test/validation use, but only one image per five-minute period was labelled and it is relatively small. | WWF-UK/UCL Biome Health Project; cite Connolly et al. (2025); CDLA-Permissive. | ~64 GB inventory. |
| [Felidae Conservation Fund 2020-2025](https://lila.science/datasets/felidae-conservation-fund) | 357k+ images, 216 San Francisco Bay Area cameras; 66 labels. Common: mule deer, gray fox, bobcat. | Useful urban-wildland carnivore/prey data with locations and no blanks. Vocabulary mixes species, unknown groups, `prey-mammal`, `prey-bird`, invertebrate, and reptile. | Felidae Conservation Fund, Bay Area Puma/Bobcat projects; cite Granados et al. (2024); CDLA-Permissive. | ~76 GB est. |
| [WSU Lynx](https://lila.science/datasets/wsu-lynx/) | 1,205,453 images in 280,485 sequences, 637 Washington locations; 26 categories including empty. Common non-empty: cattle, mule deer. | Strong sequence/location design and 4-5-frame bursts from unbaited high-elevation trail cameras. Very empty-heavy (215,659 sequences; ~77%); no boxes. Human/vehicle images removed. | Washington State University, public lands, 2016-2017; cite Thornton et al. (2026); CDLA-Permissive. | ~1.0 TB est.; sampling uncertainty makes this borderline. |
| [Desert Lion Conservation Camera Traps](https://lila.science/datasets/desert-lion-conservation-camera-traps/) | 65,959 images + 199 videos; 46 mostly species-level categories; no empty-labelled images. | Good Namibia/desert domain data with unique locations, but the no-empty curation changes deployment prevalence and makes it unsuitable alone for empty-vs-animal calibration. | Desert Lion Conservation Project, northern Namibia; CDLA-Permissive. | ~155 GB inventory. |
| [Idaho Camera Traps](https://lila.science/datasets/idaho-camera-traps/) | ~1.5M images; 62 categories; common animal labels deer, elk, cattle. | Large regional dataset, but vocabulary includes state labels such as snow/foggy lens; 70.5% empty; labels are reliable only at sequence level. Human, vehicle, and dog images removed. | Idaho Department of Fish and Game. **Special terms:** images may not be sold; scientific-publication use and acknowledgement requested. | 1.45 TB published zips; ~3.10 TB inventory. |
| [Island Conservation Camera Traps](https://lila.science/datasets/island-conservation-camera-traps/) | ~123k images, 123 cameras, seven islands in six countries; >10 classes. Common: rabbit, rat, petrel, iguana, cat, goat, pig. | ~65k boxes on ~50k images and rare island taxa. ~60% empty. Original timestamps were absent and inferred by OCR, so temporal analyses need validation. | Island Conservation eradication/conservation projects; CDLA-Permissive. | 76 GB zip; ~164 GB inventory. |
| [North American Camera Trap Images (NACTI)](https://lila.science/datasets/nacti) | 3.7M images from five US locations; 28 animal categories; common: cattle, boar, red deer. | Broad benchmark-scale regional data, but multiple source locations and substantial empties require location-held-out evaluation. | Multiple US camera-trap sources described on official page; CDLA-Permissive. | ~1.47 TB published archives. |
| [Missouri Camera Traps](https://lila.science/datasets/missouricameratraps) | ~25k images; 20 species. Common: red deer, mouflon, white-tailed deer. | Challenging clutter and long sequences; ~900 boxes. Sequence labels may apply to frames without a visible animal; 79 multi-animal images have known incomplete unique boxes. | Cite Zhang et al. (2016); CDLA-Permissive. | 10 GB zip; ~20.5 GB inventory. |
| [Wellington Camera Traps](https://lila.science/datasets/wellingtoncameratraps) | 270,450 images, 187 locations; 15 animal classes + empty + unclassifiable. Common: bird, cat, hedgehog. | Useful urban/citizen-science New Zealand data. All three frames in a burst share the sequence label, even if an animal is absent from one frame. | Victoria University of Wellington citizen scientists/ecologists; cite Anton et al. (2018); CDLA-Permissive. | 176 GB zip; ~198 GB inventory. |
| [Duck Pictures in Wetlands](https://lila.science/datasets/duck-pictures-in-wetlands/) | 674,426 Finland wetland images; 10 non-empty label classes plus `empty`. Common: unidentified waterbird, adult mallard, adult goldeneye. | Useful wetland niche with bird age labels and counts, but 660,935 images are empty (~98%). Several classes are adult/duckling variants of the same species, so true taxonomic diversity is below ten. | University of Helsinki contact; CDLA-Permissive. | ~1.9 TB est. |
| [AMMonitor Camera Traps](https://lila.science/datasets/ammonitor-camera-traps/) | ~1.2M images from ten northeastern US projects; >10 animal classes. Common non-empty: moose, white-tailed deer, fisher. | Valuable multi-project monitoring corpus with dataset and location IDs. Roughly half empty; precise coordinates were removed from LILA; users must cite the relevant constituent ScienceBase releases. | Ten AMMonitor/USGS, state, university, refuge, and USFS releases; constituents are public domain and request notification/individual citation. | Likely >2 TB public objects; rough full-source estimate ~3.4 TB. |

## Snapshot Safari legacy sets

These are genuine, standardized field-camera datasets and all exceed ten animal
classes. They are useful compact African test domains, but labels are attached
to images while being reliable only at the sequence level. Human images were
removed, and all use CDLA-Permissive. Their projects also contribute additional
data to the 2024 Expansion, so deduplicate by filename/project before combining.

| Dataset | Images / sequences | Classes and examples | Empty | Published zip |
|---|---:|---|---:|---:|
| [Snapshot Karoo](https://lila.science/datasets/snapshot-karoo) | 38,074 / 14,889 | 38; gemsbok/oryx, red hartebeest, kudu | 83.02% | See page (shared legacy inventory ~542 GB) |
| [Snapshot Kgalagadi](https://lila.science/datasets/snapshot-kgalagadi) | 10,222 / 3,611 | 31; gemsbok/oryx, other bird, ostrich | 76.14% | 10 GB |
| [Snapshot Enonkishu](https://lila.science/datasets/snapshot-enonkishu) | 28,544 / 13,301 | 39; impala, warthog, zebra | 64.76% | 32 GB |
| [Snapshot Camdeboo](https://lila.science/datasets/snapshot-camdeboo) | 30,227 / 12,132 | 43; kudu, springbok, ostrich | 43.74% | 33 GB |
| [Snapshot Mountain Zebra](https://lila.science/datasets/snapshot-mountain-zebra) | 73,034 / 71,688 | 54; Cape mountain zebra, kudu, springbok | 91.23% | 109 GB |
| [Snapshot Kruger](https://lila.science/datasets/snapshot-kruger) | 10,072 / 4,747 | 46; impala, elephant, buffalo | 61.60% | 28 GB |
| [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti) | 7.1M / 2.65M | 61; wildebeest, zebra, Thomson's gazelle | ~76% | ~6.13 TB inventory; seasons downloadable separately |

Snapshot Serengeti additionally has ~150,000 boxes on ~78,000 images and
recommended location splits. Its scale, published consensus-label addendum, and
landmark provenance (Swanson et al., *Scientific Data*, 2015) make it stronger
than the smaller legacy sets, despite its sequence-label caveat.

## Derived, mixed, or hobby datasets to keep separate

### iWildCam 2022

[iWildCam 2022](https://lila.science/datasets/iwildcam-2022/) contains about
260,000 camera-trap images and 204 public training categories. It is a competition
package derived substantially from WCS Camera Traps, with extra training data,
metadata, and satellite imagery. It is useful for reproducing the competition
or its domain-generalization task, but it should **not** be combined naively
with WCS: image overlap would create leakage and double counting. The official
image archive is about 169 GB. Use its official metadata for the full class
list, splits, and competition-specific license/citation details.

### Seattle(ish) Camera Traps

[Seattle(ish) Camera Traps](https://lila.science/datasets/seattleish-camera-traps/)
is technically qualifying camera-trap data: roughly 20,000 images and 4,500
videos from consumer trail cameras, a ~150 GB archive, and more than ten animal
labels. Animal labels include deer, heron, rabbit, raccoon, bear, dog, bird,
squirrel, owl, bobcat, crow, bee, junco, northern flicker, robin, sparrow,
spotted towhee, varied thrush, otter, goose, cat, opossum, chicken, insect, and
coyote; other labels include empty, person (`dan`), unknown, and vehicle.

It comes from one private yard and was assembled partly to fill technical test
gaps. It is useful for video ingestion, consumer-camera artifacts, and backyard
species, but it is not an ecological field survey and should not drive claims
about population prevalence or geographic generalization.

## Excluded camera-trap datasets

These are real camera-trap datasets but fail the ten-animal-class threshold:

- [UNSW Predators](https://lila.science/datasets/unsw-predators/): five classes
  (dingo, fox, goanna, possum, quoll).
- [Channel Islands Camera Traps](https://lila.science/datasets/channel-islands-camera-traps/):
  five animal groups (rodent, fox, bird, skunk, other), despite useful boxes.
- [Lindenthal Camera Traps](https://lila.science/datasets/lindenthal-camera-traps/):
  rich RGB/IR/depth video and instance masks, but only a small labelled animal
  vocabulary (four annotation classes in the released labelled subset).

Datasets tagged aerial/drone, individual ID, marine, or aggregation in the LILA
catalog were outside this review even if they have many classes.

## Class lists and acquisition

For reproducible class selection, use the official metadata rather than copying
names from this report:

1. Follow the dataset link in the tables and download its metadata. In COCO
   Camera Traps JSON, `categories` is the complete source vocabulary and
   `annotations` supplies counts.
2. Remove non-biological classes (`empty`, human/person, vehicle, camera-state,
   unclassifiable) explicitly. Do not assume their IDs are consistent across
   datasets.
3. Decide whether broad groups (`bird`, `rodent`, `small mammal`, unknown
   murid), life stages, and domestic taxa count as target classes.
4. Use LILA's [common taxonomy mapping](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/)
   to reconcile synonyms and scientific/common names, but retain the original
   label too: LILA warns that mappings can lose information for visual groups
   such as `bird of prey`.
5. Split by `location`, deployment, or project before generating frames or
   crops. Keep all frames from one sequence in one split.
6. For detection/cropping, check LILA's [published MegaDetector results](https://lila.science/megadetector-results-for-camera-trap-datasets/)
   in addition to native boxes. Native human-drawn species boxes are preferable
   when available.

## Practical shortlist for rodent-experiments

For the current project's small-animal and RGB/RGB-T focus, a pragmatic order is:

1. **Oregon Critters**: native species boxes and both trail/downward views.
2. **Indiana Small Animals**: largest targeted small-bodied-wildlife source and
   73 taxa.
3. **Ohio Small Animals**: compact AHDriFT source with 45 species.
4. **California Small Animals**: very large standardized CDFW drift-fence source
   including reptiles and amphibians, but many coarse taxon labels and blanks.
5. **New Zealand Thermal**: the best LILA source for testing thermal-track and
   modality pipelines, while recognizing it is thermal-only video.
6. **New Zealand Trailcams**: huge rodent-heavy RGB source, useful after
   aggressive balancing and project/location-held-out splitting.
7. **SWG** and **WCS**: add broad environmental, geographic, and negative-domain
   diversity after the specialized small-animal core is working.

The official LILA taxonomy mapping has exact matches for several species already
configured in this repository: California Small Animals maps to *Mus musculus*,
*Rattus norvegicus*, and *Rattus rattus*; Indiana to *Mus musculus*; Ohio to
*Rattus norvegicus*; Missouri to *Apodemus sylvaticus*; NACTI to *Rattus
rattus*; Wellington to *Rattus norvegicus* and *Rattus rattus*; and WCS to
*Rattus rattus* and *Suncus murinus*. This proves that a mapped source label is
present, not that it has enough images or locations for training; inspect counts
and location support in metadata before acquisition.

Before downloading terabytes, first fetch only metadata, compute per-class and
per-location support, select a location-disjoint subset, and stream individual
objects from GCP/AWS/Azure. LILA's [partial-download guide](https://lila.science/image-access)
documents this workflow.

## Primary sources

- [LILA dataset catalog](https://lila.science/datasets/)
- [LILA taxonomy mapping and unified camera-trap table](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/)
- [LILA image access guide](https://lila.science/image-access)
- [LILA camera-trap metadata format FAQ](https://lila.science/faq)
- [LILA MegaDetector outputs](https://lila.science/megadetector-results-for-camera-trap-datasets/)
- Individual official LILA dataset pages linked throughout the tables; these
  pages contain the provider citation, contact, metadata, class-count table (if
  published separately), and cloud download locations.
