### Datasets (collection, not evaluated yet)
- [iRodent](https://zenodo.org/records/8250392)
- [kaggle thermal images of rats/mice](https://www.kaggle.com/datasets/ronneiborges/thermal-images-of-rats-mice-for-segmentation) but this is only for *Rattus norvegicus* (wistar rat)
- [nature published dataset from camera traps sepcifically](https://www.nature.com/articles/s41597-025-06105-2)
- [snakeclef 2021](https://lindat.mff.cuni.cz/repository/items/327ccc1f-a3c5-426a-9cf4-1b329cb019fa)
- [snakes (found on roboflow)](https://universe.roboflow.com/chris-workspace/snake-species-identification/dataset/7)
- [BioTrove](https://huggingface.co/datasets/BGLab/BioTrove) Gigantic dataset covering more or less the entire multicellular macroscopic tree of life, including rodent, snake and bird species

- not directly relevant, but perhaps interesting because we might see traces of something that has run by while the machinery was off:
[AnimalClue](https://dahlian00.github.io/AnimalCluePage/)
- this is interesting as a modeling project in itself for other/future projects perhaps

## Dataset evaluation
- **iRodent**: Species labels are stripped and only the class 'rodent' is retained, not useful for us unless for pose estimation, for which the dataset is intended. good thing: it's scrapped from iNaturalist, which we might be able to use, but I am not entirely sure how to get the species labels back.
- **snakeclef21**: Have not read the paper yet, but many images retain copyright notices and the usage of flikr images is unclear wrt licensing. Provenance unclear, hence not trusted.
- **nature published dataset**: is too specific for the area the images are from (qilian mountains, China), no relevant species
- **BioTrove**: can be filtered on various taxonomic levels depending on what is needed. Started out with "[Muridae](https://en.wikipedia.org/wiki/Muridae)" (mouse-likes), which can further be filtered based on need. Because it spans essentially the entire animal tree of life (to one degree or another), we can filter the dataset for various animal species we need, and then apply OpenAI's clip model to filter out specific image types (like dead animals, animals caught in traps, bones/skulls or museum specimens) to build a dataset.
Question here: The angles and distance at which the animals are photographed might be not representative in general of what a camera trap does?

### Models
- [SpeciesNet](https://research.google/blog/where-wild-things-roam-identifying-wildlife-with-speciesnet/) Can give species classifications and goes up the taxonomic hierarchy if not sure
- [Yolo](https://docs.ultralytics.com/models/yolo11#overview) 'The' object detection model. Retrainable, but we need data of the species in question
- We need to find a way to sensor-fuse the thermals + rgb images?

### Remarks
- we could perhaps try to become a member of wildlifeinsights to help the community with camera trap data?

### Questions
- uncertainty -> inference = 'accumulating evidence for hypothesis and give out likelihood', not binary is/is not. Reason: We are not guaranteed that 1st/nth image taken of an animal is a good one.
	- model calibration?
	- inference over multiple images?
	- uncertainty quantification?
- data
	- would it be enough to build a image augmentation pipeline to create low-res thermal images from the data we can find?
        - No, probably not. We only can use the thermal images realistically as an indicator if something is there or not.
- integration of thermal images
	- sensor fusion for low light conditions even during the day?
	- thermal image as prior for rgb object detection? -> this is what we need probably.
	- what to do at night? can we use posture, movement from thermal image for identification?
        - thermal image alone will not be enough b/c resolution is too low
- ecology
	- onthogenetic niche shift in snake predators
        - is that relevant in Sri Lanka?
		- size of snake might play a role (young snakes eat different rodents)
	- what influence does prior knowledge of local ecology have on the detection?
- inference and deployment
	- how big will the surveillance system become?
        - we know now it's gonna 7x7 trap grid, but cameras is yet to be determined
	- how many images do we expect to arrive on the server per hour
	- can we allow for future GPU availability even if that means we are overtaxing the current deployment hardware now?

## Software architecture ideas (very loose, needs to be made more workable and scaled perhaps)
- generally, it would make sense to coopt faunanet for this, b/c we have the infrastructure already for much of what we need.
- faunanet should be generalized to not be audio centric -> sensor channel agnostic
- make small faunanet-cam repo that is based on Hammad's work and can be used with faunanet in a docker setup
- both should support multi-recorder systems in the long run (not this project, but at the moment I don't see anything that would make this hard): audio, video, perhaps lidar or whatever else
- add a faunanet-comms repository which can be run to receive the data and send it. usual SOLID principles apply to make it work with all kinds of comms backends. Build on top of Hammad's comms repo?
- use docker-compose to run record + comms on the raspberry
- add faunanet-manage repo to have a dashboard for a fleet of raspberrys. Usual SOLID principles apply here. perhaps base it on the frontend of heiplanet?
- on the server side, do away with the faunanet REPL, but coopt the watcher system for inference. Test with YOLO. Have faunanet and dashboard run on the server
- run server with docker-compose: comms + watch + inference
- What should we simplify, remove or add?

## Machine learning ideas (very loose, needs to be made workable and scaled down perhaps)
- Yolo26 is a good starting point for detection/classification, cropping, pose estimation
- SpeciesNet is more powerful for species detection and perhaps should be used there
    - doesn't seem to perform well with the small animals we have
- We can use some of the available dataset for finetuning or retraining
    - Problem: Many images are too close-up
- We may be able to 'thermalize' images for finetuning/retraining, but that is rather difficult
    - no we are not, at least not in a way that would work well and could be used for training
- The thermal camera they use: https://www.amazon.de/-/en/Waveshare-Long-Wave-Interfaces-Communication-Development/dp/B0FW3XZY7N/
    - has 80x62 pixels, which is extremely low and not good enough to make out details
- we could try, depending on time of day and quality if we have time and resources for it:
	- YOLO pose estimation from thermal as indicator of what we are looking at?
	- thermal as extra channels for YOLO inference and/or speciesnet, to guide where the thing is we want.
        - that was the original idea
	- only rgb if good enough, whatever good enough means atm
    - thermal will not help much I (HM) believe atm at night or for snakes. We would need an active infrared camera for that I think atm.
- We need a solid estimation of inference uncertainty. How confident are we about having detected something? A single estimate is not enough. Read up on that:
   - take a set of species -> hypothesis -> update -> do until images used up
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
- preliminary tests show that speciesnet at least is not good enough for the biotrove data that we have. It appears to be great for larger species commonly seen in camera traps (nutria, muskrat, beaver), but it seems to struggle with the small animals we have here.
- The data that they currently produce is a very particular set: top down perspective, one angle, only one species, so from this we cannot generalize to real world deployments as far as I see?
- I think atm that retraining a yolo model makes more sense.
- We can check out Biotrove-CLIP and YOLOE26 as vision language models for detecting species. For species-detection, some finetuning will be needed likely, at least Biotrove-CLIP says that on their benchmark their models underperformed (huggingface biotrove-clip site).
    - Just like with microscopes, it might turn out that transfering images from one setup to another doesn't cut it and we will have to wait until data is produced to build a proper detection model. Until then, we would probably be limited to 'rodent or not' or perhaps the Genus level (Rattus, Mus) at best?
    - Even if that is the case, the development work can be done, so it's not a roadblock for the project as it is now.
