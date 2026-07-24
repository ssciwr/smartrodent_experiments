### Datasets (collection, not evaluated yet)
- ~~[iRodent](https://zenodo.org/records/8250392)~~
- [kaggle thermal images of rats/mice](https://www.kaggle.com/datasets/ronneiborges/thermal-images-of-rats-mice-for-segmentation) but this is only for *Rattus norvegicus* (wistar rat)
- [nature published dataset from camera traps sepcifically](https://www.nature.com/articles/s41597-025-06105-2)
- [snakeclef 2021](https://lindat.mff.cuni.cz/repository/items/327ccc1f-a3c5-426a-9cf4-1b329cb019fa)
- [snakes (found on roboflow)](https://universe.roboflow.com/chris-workspace/snake-species-identification/dataset/7)
- [BioTrove](https://huggingface.co/datasets/BGLab/BioTrove) Gigantic dataset covering more or less the entire multicellular macroscopic tree of life, including rodent, snake and bird species
- [LiLA-BC: Labeled Information Library of Alexandria: Biology and Conservation](http://lila.science/) Collection of datasets for ML in biology and conservation. Labels inconsistent between datasets.
- [wildlife-insights](https://app.wildlifeinsights.org/explore) Worldwide collection of camera trap projects with available data. Would be ideal for our case.

**Camera-trap / small-mammal box-camera datasets (genuine CT domain):**
- [Ohio Small Animals / AHDriFT](https://lila.science/datasets/ohio-small-animals/) (LILA) — 118k RGB images, downward-facing box camera, **45 species at species level**, COCO Camera Traps format, images pre-sorted into per-species folders, permissive license (CDLA). Species are North American (meadow vole, chipmunks, etc.), so **not our Palearctic targets**, but it's the closest analog for the small-mammal *camera-trap* domain + a clean labelled set.
- [California Small Animals](https://lila.science/datasets/california-small-animals/) (LILA) — 2.28M images, CDFW small-mammal camera protocol, species-level. North American.
- [Oregon Critters](https://lila.science/datasets/oregon-critters/) (LILA) — ~100k images, US Forest Service small-animal cameras, species-level. North American.
- [LILA taxonomy-mapping trick](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/) — lets you filter *all* LILA CT datasets down to Rodentia and pull only the rodent images across sets in one pass. Worth scripting.
- [Böhner "Small mammal classification model"](https://zenodo.org/records/7801786) (Zenodo 7801786, **CC-BY**, images bundled ~480MB) + [camera_trap_workflow repo](https://github.com/hannaboe/camera_trap_workflow) — Arctic Norway box/IR cameras. Classes = voles, lemmings, mustelids, **shrews**. Useful for our Sorex→"shrew" class + vole distractors + real CT domain. **No Apodemus/Rattus/Mus** (wrong biome).
- [DeepFaune](https://www.deepfaune.cnrs.fr/en/) — European CT classifier/dataset, well-labelled, but all small mammals collapse into a single `micromammal` class → **not species-level**. Model useful, labels not.
- [Democratising Camera Trap AI / Trap Tracker](https://arxiv.org/pdf/2606.10940) (arXiv 2606.10940, 2026) — open model over 28 UK mammal/bird classes (incl. wood mouse & rat). Model is open; the 48k-instance labelled set is **not clearly deposited** → request from authors (Conservation AI / Nottingham Trent).
- UK small-mammal box-camera citizen science — [BucketBuddy / Littlewood Box comparison](https://link.springer.com/article/10.1007/s13364-026-00874-w) (Mammal Research 2026) and [Cambridge peatland small-mammal CT study](https://link.springer.com/article/10.1007/s10344-020-01449-z) (EJWR 2020). These specifically image **wood mouse + Sorex araneus + S. minutus** on camera — the rare designs that actually resolve our species. Images usually **not deposited** → email authors.
- [SAWIT small-animal CT dataset](https://doi.org/10.5281/zenodo.14927692) (Zenodo 14927692) — check taxa fit (non-Palearctic-leaning).
- [Böe et al. semi-automatic small-mammal CT workflow](https://www.sciencedirect.com/science/article/pii/S1574954123001796) (Ecol. Informatics) — companion to the Böhner model above.

**RGB photo repositories for the target species (NOT camera-trap, but license-clean-able):**
- **iNaturalist bulk export** — primary source for species-level RGB of all six target taxa (Rattus rattus/norvegicus, Mus musculus, Apodemus spp., Sorex araneus/minutus). Only CC0 / CC-BY / CC-BY-NC photos are exported; pull via GBIF multimedia extension, the iNat AWS Open Data mirror, or the iNat export tool. (BioTrove is already derived from this — so a direct iNat pull mainly buys us fresher/uncropped images + the license field.)
- [Observation.org](https://observation.org) — European citizen-science, real API, strong coverage of Apodemus/Sorex/Rattus/Mus. Many photos are per-photographer rights-reserved → filter to the openly-licensed subset (smaller yield, but genuinely new vs iNat).
- [Roboflow Universe rat-detection sets](https://universe.roboflow.com/search?q=class%3Arat) — several urban *Rattus* RGB sets (400–1000 imgs each). Labelled `rat` at genus level only, but real synanthropic imagery.

**Dead ends (recorded so we don't re-check):**
- [Artportalen](https://www.artportalen.se/) (Swedish Species Observation System) — occurrence records are CC0, **but the photos are per-photographer copyright**. No bulk image download, no media API, and images are **not** shipped in its GBIF multimedia extension (GBIF pull returns points, not pictures). Good for distribution data only. **Not viable for images.**
- [Agouti](https://agouti.eu/) / [Snapshot Europe](https://www.ab.mpg.de/snapshot-europe) / EUROMAMMALS — the big European CT infrastructure, but explicitly scoped to mammals **>~200 g**, which excludes Mus (~20g), Apodemus (~25g) and Sorex (~5–10g) by design. Not useful for our target size class.

- not directly relevant, but perhaps interesting because we might see traces of something that has run by while the machinery was off:
[AnimalClue](https://dahlian00.github.io/AnimalCluePage/)
- this is interesting as a modeling project in itself for other/future projects perhaps

## Dataset evaluation
- **iRodent**: Species labels are stripped and only the class 'rodent' is retained, not useful for us unless for pose estimation, for which the dataset is intended. good thing: it's scrapped from iNaturalist, which we might be able to use, but I am not entirely sure how to get the species labels back.
- **snakeclef21**: Have not read the paper yet, but many images retain copyright notices and the usage of flikr images is unclear wrt licensing. Provenance unclear, hence not trusted.
- **nature published dataset**: is too specific for the area the images are from (qilian mountains, China), no relevant species
- **BioTrove**: can be filtered on various taxonomic levels depending on what is needed. Started out with "[Muridae](https://en.wikipedia.org/wiki/Muridae)" (mouse-likes), which can further be filtered based on need. Because it spans essentially the entire animal tree of life (to one degree or another), we can filter the dataset for various animal species we need, and then apply OpenAI's clip model to filter out specific image types (like dead animals, animals caught in traps, bones/skulls or museum specimens) to build a dataset.
Question here: The angles and distance at which the animals are photographed might be not representative in general of what a camera trap does?
- **LiLA-BC**: tbd.
- **Wildlife-insights**: Download hasn't worked for me yet?
### Models
- [SpeciesNet](https://research.google/blog/where-wild-things-roam-identifying-wildlife-with-speciesnet/) Can give species classifications and goes up the taxonomic hierarchy if not sure
- [Yolo](https://docs.ultralytics.com/models/yolo11#overview) 'The' object detection model. Retrainable, but we need data of the species in question
- We need to find a way to sensor-fuse the thermals + rgb images?
- [Megadetector](https://github.com/agentmorris/MegaDetector) The detector backbone that SpeciesNet is built upon. Only finds animals in images, doesn't classify anything.
### Remarks
- we could perhaps try to become a member of wildlifeinsights to help the community with camera trap data?

### Questions
- uncertainty -> inference = 'accumulating evidence for hypothesis and give out likelihood', not binary is/is not. Reason: We are not guaranteed that 1st/nth image taken of an animal is a good one.
	- model calibration?
	- inference over multiple images?
	- uncertainty quantification?
- data
	- would it be enough to build a image augmentation pipeline to create low-res thermal images from the data we can find?
        - No, probably not. We only can use the thermal images realistically as an indicator if something is there or not, and turning RGB into *realistic* thermals is not productive b/c distribution shifts from real thermals are impossible to detect but relevant.
- integration of thermal images
	- sensor fusion for low light conditions even during the day? (shaded area)
	- thermal image as prior for rgb object detection? -> this is what we need probably.
	- what to do at night? can we use posture, movement from thermal image for identification?
        - thermal image alone will not be enough b/c resolution is too low
- ecology
	- onthogenetic niche shift in snake predators
		- size of snake might play a role (young snakes eat different rodents)
        - is that relevant in Sri Lanka?
	- what influence does prior knowledge of local ecology have on the detection?
- inference and deployment
	- how big will the surveillance system become?
        - we know now it's gonna 7x7 trap grid, but cameras is yet to be determined?
	- how many images do we expect to arrive on the server per hour
	- can we allow for future GPU availability even if that means we are near-overtaxing the current deployment hardware now?

## Software architecture ideas (very loose, needs to be made more workable and scaled perhaps)
- generally, it would make sense to co-opt faunanet for this, b/c we have the infrastructure already for much of what we need.
- faunanet should be generalized to not be audio centric -> sensor channel agnostic
- make small faunanet-cam repo that is based on Hammad's work + faunanet-record and can be used with faunanet in a docker setup
- iff faunanet: add a faunanet-comms repository which can be run to receive the data and send it. usual SOLID principles apply to make it work with all kinds of comms backends. Build on top of Hammad's comms repo?
- use docker-compose to run record + comms on the raspberry
- dashboard -> James
- on the server side, do away with the faunanet REPL I think, but coopt the watcher system for inference. Test with YOLO. Have faunanet and dashboard run on the server
- run server with docker-compose: comms + watch + inference
- run devices with docker-compose: cams+comms+(record if bioacoustics is needed)
- What should we simplify, remove or add?

## Machine learning ideas (very loose, needs to be made workable and scaled down perhaps)
- Yolo26 is a good starting point for detection/classification, cropping, pose estimation
- SpeciesNet is more powerful for species detection and perhaps should be used there
    - doesn't seem to perform well with the small animals we have
- We can use some of the available dataset for finetuning or retraining
    - Problem: Many images are too close-up perhaps? we need to try
- We may be able to 'thermalize' images for finetuning/retraining, but that is rather difficult
    - no we are not, at least not in a way that would work well and could be used for training
- The thermal camera they use: https://www.amazon.de/-/en/Waveshare-Long-Wave-Interfaces-Communication-Development/dp/B0FW3XZY7N/
    - has 80x62 pixels, which is extremely low and not good enough to make out details
- we could try, depending on time of day and quality if we have time and resources for it:
	- YOLO pose estimation from thermal as indicator of what we are looking at?
	- thermal as extra channels for YOLO inference and/or speciesnet, to guide where the thing is we want.
        - that was the original idea and what we should go for first
	- only rgb if good enough, whatever good enough means atm
    - thermal will not help much I (HM) believe atm at night or for snakes. We would need an active infrared camera for that I think atm., at least that is the most low-risk path
- We need a solid estimation of inference uncertainty. How confident are we about having detected something? A single estimate is not enough. Read up on that:
   - take a set of species -> hypothesis -> update ->...
- Model calibration?
- If we need to do finetuning/training we need to make sure we handle long-tailed class distributions properly, which we always have in ecology, unless we are focusing only on the dominant species for this stage.
- We need a 'not what we are after' class.
    - 2-stage pipeline:
        - check if interesting species with confidence level
            - confident: discard
            - not confident enough -> expert checks
        - if confident it's what we are after:
            - classify what it is

### Model evaluation

- early June 2026:
    - preliminary tests show that speciesnet at least is not good enough for the biotrove data that we have. It appears to be great for larger species commonly seen in camera traps (nutria, muskrat, beaver), but it seems to struggle with the small animals we have here.
    - The data that they currently produce is a very particular set: top down perspective, one angle, only one species, so from this we cannot generalize to real world deployments as far as I see?
    - I think atm that retraining a yolo model makes more sense.
    - We can check out Biotrove-CLIP and YOLOE26 as vision language models for detecting species. For species-detection, some finetuning will be needed likely, at least Biotrove-CLIP says that on their benchmark their models underperformed (huggingface biotrove-clip site).
        - Just like with microscopes, it might turn out that transfering images from one setup to another doesn't cut it and we will have to wait until data is produced to build a proper detection model. Until then, we would probably be limited to 'rodent or not' or perhaps the Genus level (Rattus, Mus) at best?
        - Even if that is the case, the development work can be done, so it's not a roadblock for the project as it is now.

- Final eval:
    - Summary:
        - biotrove-clip: vision language, open vocabulary. trained for species identification
        - speciesnet: specifically camera trap model, trained for species identification, but goes up the phylogenetic tree when its not confident in the lower levels it finds (configurable). uses geofencing (had it set to DE for this test). Crop extraction possible
        yoloe: open vocabulary vision language model. Crop extraction possible
        yolo26: object detection + classfication model, uses COCO classes. Crop extraction possible
        Results
        - biotrove-clip is rather bad for everything other than identifying 'animal', and doesn´t understand species well (that's expected, the model is reported as such. only in there for testing purposes).
        - speciesnet is the best at identifying animals in it (~100%) gets rodents occassionally (but not quantified here b/c of the phylogenetic tree mechanic it uses, I didn´t aggregate all the families and stuff that would entail 'rodent' or something rodent-like like shrews), but isn't great at identifying species either. It's trained for camera trap usage, but seems to be trained on 'charismatic megafauna' -> beaver, muskrat, elk, not small critters like rats, mice and shrews.
        - yoloe (open vocabulary vision language) works well for identifying animals , but worse than speciesnet (85-99% or so, best for detection threshold ~0.2). Works for rodent identification specifically but not as well (~60 -76%)
        - yolo26 is struggling more (70% correct or so) with animal identification with default classes, would need retraining
        Hardware: not really checked in detail, yolo26 needed ~800 - 900MB on the gpu, vision language needs more. Speciesnet ~3.5gb

    - Caveats:
        I aggregated classes that correspond to 'animal' in YOLO26 and Speciesnet (where it did not right away default to 'animal') post hoc. Good results derived from that do not mean that the model separates animals from non-animals cleanly in its latent space as far as I can understand it, so if we change the species the results might jump unexpectedly.
        I tried to mitigate by including more species than we would most likely be interested in for the rodents at least, so the ones we need should be in the current test and separate out well.

        Biotrove dataset is derived from a citizen scientist database (inaturalist). Images from species associated with human dwellings (house mouse, black and brown rats) can have different biases than images of species mostly encountered in the wild (wood mouse, bicolored shrew), so an object detection might detect non-animal objects more often in the former b/c there's more artificial stuff on them. Also, the former class of animals is substantially overrepresented (good for us though).

    - Possible ways forward I (HM) see:
        use qwen and call it a day if that's good enough for them (no species identification though, no crops afaik and no ability to modify the model to include IR or thermal unless we can solve that with a prompt a la 'here is the same image in thermal and visible light, find the animalX... )
        fine tune Yolo26 on the dataset we have using crops derived from Speciesnet. Probably the most tractable and the one that carries us furthest if it works well(!), but also the most resource intensive for us.
        build a 2 stage pipeline: identify animal first, then check what it is in a separate step with something specialized. thermal would come in in the first for the most part I suppose, we won´t be able to do more with that than separate bird from rodent (snakes will be difficult to spot there).
        (edited)
        Integration of thermal images into YOLO:
        there's a rgb-t version of YOLO12 or so in the literature and a couple of other versions that we could try and use. It always boils down to have a second channel in the model for the thermal images and then a cross-attention/gated-attention neck that merges them before they go to the classifier/object detection head.