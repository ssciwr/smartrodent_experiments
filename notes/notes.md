### Datasets (collection, not evaluated yet)
- [iRodent](https://zenodo.org/records/8250392)
- [kaggle thermal images of rats/mice](https://www.kaggle.com/datasets/ronneiborges/thermal-images-of-rats-mice-for-segmentation) but this is only for *Rattus norvegicus* (wistar rat)
- [nature published dataset from camera traps sepcifically](https://www.nature.com/articles/s41597-025-06105-2)
- [snakeclef 2021](https://lindat.mff.cuni.cz/repository/items/327ccc1f-a3c5-426a-9cf4-1b329cb019fa)
- [snakes (found on roboflow)](https://universe.roboflow.com/chris-workspace/snake-species-identification/dataset/7)
- [BioTrove](https://huggingface.co/datasets/BGLab/BioTrove) Gigantic dataset covering more or less the entire multicellular macroscopic tree of life, including rodent, snake and bird species

- not directly relevant, but perhaps interesting because we might see traces of something that has run by while the machinery was off:
[AnimalClue](https://dahlian00.github.io/AnimalCluePage/)
- this is interesting as a modeling project in itself

## Dataset evaluation
- **iRodent**: Species labels are stripped and only the class 'rodent' is retained, not useful for us unless for pose estimation, for which the dataset is intended. good thing: it's scrapped from iNaturalist, which we might be able to use.
- **snakeclef21**: Have not read the paper yet, but many images retain copyright notices and the usage of flikr images is unclear wrt licensing. Provenance unclear
- **nature published dataset**: is too specific for the area the images are from (qilian mountains, China)
- **BioTrove**: can be filtered on various taxonomic levels depending on what is needed. Started out with "[Muridae](https://en.wikipedia.org/wiki/Muridae)" (mouse-likes), which can further be filtered based on need. Because it spans essentially the animal tree of life, we can filter the dataset for various animal species we need, and then apply OpenAI's clip model to filter out specific image types (like dead animals, animals caught in traps, bones/skulls or museum specimens) to build a dataset.

## Code



## List of interesting species beyond just black rat and mouse
researched with the help of gemini flash
### Central Europe: The Complete Small Mammal Pest Matrix

In European settlements, the crossover between true rodents, dormice, and shrews is extensive. They all raid human structures for heat and food, especially in autumn.
### 1. True Rodents (Muridae & Cricetidae)

- **`Rattus norvegicus` (Brown Rat):** Subterranean king of cities, sewers, and livestock farms. Prime vector for _Leptospira_.

- **`Rattus rattus` (Roof Rat):** Less common than the brown rat but heavily colonized in older granaries, shipping ports, and high rafters of agricultural buildings.

- **`Mus musculus` (House Mouse):** Classic indoor commensal pest found globally inside walls, kitchens, and urban warehouses.

- **`Mus spicilegus` (Steppe Mouse):** Found in southeastern Central Europe (e.g., Austria/Hungary borders). They build massive, human-disruptive seed mounds right on the edges of agricultural plots.

- **`Apodemus sylvaticus` (Wood Mouse):** The main "invader" of suburban gardens and garages during winter.

- **`Apodemus flavicollis` (Yellow-necked Mouse):** Lives near woodland borders but aggressively enters rural attics and houses. A critical vector for **Tick-Borne Encephalitis (TBE)** and various hantaviruses.

- **`Apodemus agrarius` (Striped Field Mouse):** Easily identified by its black dorsal stripe. It heavily bridges rural fields and urban parks, acting as the primary reservoir for the deadly **Dobrava-Belgrade hantavirus**.

- **`Microtus arvalis` (Common Vole):** Explodes in agricultural population cycles, frequently overrunning barns and grain bins, bringing **Tularemia** into contact with humans.

- **`Myodes glareolus` (Bank Vole):** Inhabits rural woodpiles and gardens. The absolute primary reservoir for the **Puumala hantavirus**.

- **`Arvicola amphibius` (European Water Vole):** A massive, rat-sized vole that acts as a major pest in suburban orchards and gardens by destroying root systems.


### 2. Dormice (Gliridae — The Rodent Analogs)

- **`Glis glis` (Edible Dormouse):** Looks like a miniature squirrel-rat hybrid. They are notorious for nesting inside the roof insulation and attics of suburban houses, causing severe structural damage and noise.

- **`Eliomys quercinus` (Garden Dormouse):** Frequently found in human orchards, fruit gardens, and outbuildings.


### 3. Shrews & Moles (Soricidae & Talpidae — The Insectivore Analogs)

- **`Crocidura russula` (Greater White-toothed Shrew):** Highly synanthropic. They move out of compost piles and directly into basements and crawlspaces. *not sure about risks*

- **`Crocidura leucodon` (Bicolored Shrew):** Visually distinct but highly dangerous from a public health standpoint; they are the primary reservoir for **Borna disease virus 1 (BoDV-1)**, which causes fatal encephalitis in humans.

- **`Sorex araneus` (Common Shrew):** Frequently caught on camera traps near rural sheds and woodpiles; known carriers of Seewis and Altai hantaviruses.


## Sri Lanka: The Tropical Commensal & Field Community

In Sri Lanka, open-air architecture and tropical agricultural cycles create a highly fluid environment where large burrowing rats, climbing mice, arboreal squirrels, and highly vocal shrews constantly overlap.
*Not all of them might be useful for our purposes, b/c their health impact is unclear*

### 1. True Rodents (Muridae)

- **`Rattus rattus` (House Rat / Roof Rat):** The absolute dominant urban pest across Sri Lanka, occupying ceilings, rafters, and thatch.

- **`Rattus norvegicus` (Brown Rat):** Primarily restricted to massive urban port cities like Colombo, occupying subterranean drains.

- **`Mus musculus` (House Mouse — Subspecies _castaneus_):** Pervasive indoor pest in both high-density urban settings and rural villages.

- **`Bandicota indica` (Greater Bandicoot Rat):** A massive, aggressive, cat-sized rodent that infests urban sewers, markets, and drainage ditches. High-risk vector for **Leptospirosis**.

- **`Bandicota bengalensis` (Lesser Bandicoot Rat):** Smaller but vastly more destructive to infrastructure. They dig massive burrow networks right up against the foundations of rural kitchens, village homes, and rice mills.

- **`Millardia meltada` (Soft-furred Field Rat):** Tracks human harvesting schedules closely, moving en masse from grain fields into village crop stores and homes.

- **`Mus booduga` (Little Indian Field Mouse):** Tiny mouse that floods rice paddies and frequently infiltrates mud-brick homes and rural floor boards.

- **`Mus cervicolor` (Fawn-colored Mouse):** Co-habitates with humans in rural spaces, frequently nesting in thatched roofs.

- **`Vandeleuria oleracea` (Asiatic Long-tailed Climbing Mouse):** A tiny, agile climber that transitions from banana and coconut plantations straight into the ceilings and roofs of rural houses.

- **`Golunda ellioti` (Indian Bush Rat):** Lives in the dense, brushy borders and living fences surrounding semi-rural properties.


### 2. Large Rodents & Arboreal Analogs (Sciuridae & Hystricidae)

- **`Funambulus palmarum` (Three-striped Palm Squirrel):** Functionally behaves like a rodent pest in Sri Lanka, nesting inside roof tiles, raiding kitchens, and contaminating food prep areas.

- **`Hystrix indica` (Indian Crested Porcupine):** A massive rodent that frequently invades rural home gardens and agricultural fringes, causing major agricultural conflict.


### 3. Shrews (Soricidae — The Major Vector Analogs)

- **`Suncus murinus` (Asian House Shrew):** Locally known as the _Hik-meeya_. It is large, grey, musky, makes sharp chattering noises, and lives 100% inside human dwellings. It is a dominant carrier of **Leptospirosis** and flea-borne typhus. Your model _must_ separate this from true rats.

- **`Suncus montanus` (Sri Lanka Highland Shrew):** Takes over the synanthropic niche of _S. murinus_ in the colder, high-altitude human settlements of the central hill country.
### Models
- [SpeciesNet](https://research.google/blog/where-wild-things-roam-identifying-wildlife-with-speciesnet/) Can give species classifications and goes up the taxonomic hierarchy if not sure
- [Yolo](https://docs.ultralytics.com/models/yolo11#overview) 'The' object detection model. Retrainable, but we need data of the species in question
- We need to find a way to sensor-fuse the thermals + rgb images?

### Remarks
- we could perhaps try to become a member of wildlifeinsights to help the community with camera trap data?

### Questions
- uncertainty -> inference = 'accumulating evidence for hypothesis and give out likelihood', not binary is/is not.
	- model calibration
	- inference over multiple images
	- uncertainty quantification
- data
	- would it be enough to build a image augmentation pipeline to create low-res thermal images from the data we can find?
- integration of thermal images
	- sensor fusion for low light conditions even during the day?
	- when to run thermal image?
	- thermal image as prior for rgb object detection?
	- what to do at night? can we use posture, movement from thermal image for identification? thermal image alone will not be enough b/c res. is too low
- ecology
	- onthogenetic niche shift in snake predators
		- size of snake might play a role (young snakes eat different rodents)
	- is there any value in extending our involvement towards this
	- how does the risk model look like?
	- what influence does prior knowledge of local ecology have on the detection?
- inference and deployment
	- how big will the surveillance system become?
		- how many images do we expect to arrive on the server per hour
	- can we allow for future GPU availability even if that means we are overtaxing the current deployment hardware now?
	-

## Software architecture

- faunanet-record should be generalized to be not audio centric -> sensor channel agnostic
- faunanet should be generalized to not be audio centric -> sensor channel agnostic
- both should support multi-recorder systems: audio, video, perhaps lidar or whatever else
- add a faunanet-comms repository which can be run to receive the data and send it. usual SOLID principles apply to make it work with all kinds of comms backends. Build on top of Hammad's comms repo
- use docker-compose to run record + comms.
- add faunanet-watch repo to have a dashboard for a fleet of raspberrys. Usual SOLID principles apply here
- on the server side, do away with the faunanet REPL, but coopt the watcher system for inference. Test with YOLO.
- run server with docker-compose: comms + watch+inference
- What should we simplify, remove or add?

## Machine learning
- Yolo26 is a good starting point for detection, cropping, pose estimation
- SpeciesNet is more powerful for species detection and perhaps should be used there
- We can use some of the available dataset for finetuning
- We may be able to 'thermalize' images for finetuning/retraining
- we could try, depending on time of day and quality:
	- YOLO pose estimation from thermal as indicator of what we are looking at
	- thermal as extra channels for YOLO inference and/or speciesnet
	- only rgb if good enough, whatever good enough means atm
- We need a solid way of estimating which approach works best in the wild! This is the most difficult part of the whole thing, and outside our current reach
- We need a solid estimation of inference uncertainty. How confident are we about having detected something. A single estimate is not enough. Read up on that:
	- Bayesian approach: hypothesis -> update -> do until done
	- for the speciesnet model, we could use the thermal images as preconditioner for the object detection system -> would avoid false blanks
	- ... or should we do that in the preprocessing side? probably not, some finetuning would be necessary either way
- If we need to do finetuning/training we need to calibrate properly, i.e., need some idea of how common certain species are in the area?
- If we need to do finetuning/training we need to make sure we handle long-tailed class distributions properly, which we always have in ecology
### Model evaluation
preliminary tests show that generic models don't seem to be good enough out of the box.