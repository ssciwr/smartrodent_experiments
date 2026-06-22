## List of interesting species beyond just black rat and mouse

- Species lists have been derived from Gemini/gpt-5.5 powered internet searches about health relevant rodent species in central europe and sri lanka, so take it with a grain of salt.
- Species list is more general than needed for our current project, and are just there to check how we do with a mixed set of common and uncommon species, and what data is available.
- For starters we can just use Rattus and Mus perhaps.

### Central Europe: configured BioTrove target list

### 1. Rodents (Muridae & Cricetidae)

- **`Rattus norvegicus` (Brown Rat / Norway Rat):** Strongly human-associated in many cities, farms, food-storage settings, sewers, and other built environments. It is a public-health target because commensal rats in Europe have been reported with many zoonotic infectious agents. Refs: [CE-RN1], [CE-RAT].

- **`Rattus rattus` (Black Rat / Roof Rat):** A climbing, globally invasive commensal rat that is often associated with buildings, ports, storage, and other human-modified habitats. In Europe it is generally less widespread than `R. norvegicus`, but still relevant as a pest and zoonotic-host species. Refs: [CE-RR1], [CE-RAT].

- **`Mus musculus` (House Mouse):** A globally distributed commensal species, frequently associated with residential, agricultural, and commercial structures; also occurs in feral populations. Relevant for food contamination, infrastructure nuisance, and as a visual/model confounder for other small rodents. Refs: [CE-MM1], [CE-MM2].

- **`Apodemus sylvaticus` (Wood Mouse):** A widespread and adaptable European mouse found in woodlands, shrublands, arable fields, parks, gardens, and other human-modified habitats. It should be treated mainly as a peri-domestic or outdoor species rather than a strictly indoor commensal. Refs: [CE-AS1], [CE-AS2].

- **`Apodemus flavicollis` (Yellow-necked Mouse):** Primarily associated with woodland and forest-edge habitats, but can occur near human habitation where woodland and buildings meet. It is also one of the small mammals discussed in European tick/pathogen ecology. Refs: [CE-AF1], [CE-TICK].

- **`Apodemus agrarius` (Striped Field Mouse):** Originally associated with steppe and grassland, but also common in anthropogenic habitats such as meadows, croplands, field edges, road verges, gardens, and orchards in parts of its range. It has documented relevance in European orthohantavirus ecology, but the public-health implication is regional and pathogen-specific. Refs: [CE-AA1], [CE-HANTA].

- **`Microtus arvalis` (Common Vole):** A common vole of European agricultural landscapes. Population outbreaks can damage crops, and outbreaking farm-vole populations have been linked to increased concern about zoonotic pathogen transmission, including tularemia in some regions. Refs: [CE-MA1], [CE-MA2].

- **`Microtus agrestis` (Field Vole):** A grassland, wetland, heathland, cropland, and agricultural-mosaic vole in Europe. Included as a target species for visual/taxonomic coverage rather than because it is a classic indoor pest. Refs: [CE-MAG1].

- **`Myodes glareolus` (Bank Vole):** A woodland-associated vole that is important in European public-health context because it is the reservoir host of Puumala orthohantavirus, which can cause nephropathia epidemica in humans. Refs: [CE-MG1], [CE-MG2].


### 2. Shrews (Soricidae — non-rodent visual/pathogen confounder)

- **`Crocidura leucodon` (Bicolored White-toothed Shrew):** Not a rodent. Included because it can be visually confused with small rodents and because it is a documented reservoir host of Borna disease virus 1 in parts of Central Europe. Refs: [CE-CL1], [CE-CL2].

### Sri Lanka: configured BioTrove target list

### 1. Rodents (Muridae)

- **`Rattus norvegicus` (Brown Rat / Norway Rat):** A globally commensal rat and plausible urban/peri-urban pest target in Sri Lanka. In one Sri Lankan storage-facility study, it was not the dominant trapped small mammal; `R. rattus` and `Suncus` spp. were much more common. Refs: [SL-L1], [CE-RN1].

- **`Rattus rattus` (Black Rat / House Rat / Roof Rat):** A major Sri Lankan target. In the cited storage-facility study, `R. rattus` was the dominant captured small mammal across sampled districts and was discussed in the context of Leptospira reservoir ecology. Refs: [SL-L1], [SL-L2].

- **`Mus musculus` (House Mouse):** A globally commensal mouse associated with residential, agricultural, and commercial structures. It remains a useful class for model coverage, although the Sri Lankan reservoir study cited here emphasized `R. rattus`, `Suncus` spp., `Bandicota` spp., and `Mus booduga` rather than `M. musculus`. Refs: [CE-MM1], [SL-L1].

- **`Bandicota indica` (Greater Bandicoot Rat):** A large South/Southeast Asian murid recorded from Sri Lanka. It was rare in the cited Sri Lankan storage-facility trapping data but remains relevant because `Bandicota` spp. are part of the local rodent/reservoir community. Refs: [SL-BI1], [SL-L1].

- **`Bandicota bengalensis` (Lesser Bandicoot Rat):** Occurs in Sri Lanka and the broader Indian subcontinent. It was also rare in the cited storage-facility trapping data, so claims about dominance should be avoided unless local data support them. Refs: [SL-BB1], [SL-L1], [SL-BART].

- **`Mus booduga` (Little Indian Field Mouse):** A small South Asian mouse included in the Sri Lanka config. In the cited Sri Lankan storage-facility study it was captured at low frequency and only in the wet zone, so it should be treated as a local field/peri-domestic target rather than a universally dominant indoor pest. Refs: [SL-L1].

- **`Vandeleuria` (Long-tailed climbing mice, genus-level target):** The config uses the genus name rather than a species name. `Vandeleuria oleracea` is recorded from Sri Lanka and occupies forested and human-modified habitats including cropland; use genus-level labels carefully when evaluating model outputs. Refs: [SL-VO1]. *This has been fixed and we use the 'Vandleuria oleracea' now.

### 2. Shrews (Soricidae — non-rodent visual/pathogen confounder)

- **`Suncus murinus` (Asian House Shrew / Asian Musk Shrew):** Not a rodent. It is important as a Sri Lankan visual confounder and reservoir-context species: in the cited storage-facility study, `Suncus` spp. were the second most common captured small mammals after `R. rattus`, and `S. murinus` was reported from all three sampled zones. Refs: [SL-L1], [SL-SM1].

### References for this species section

- [CE-RAT] Strand, T. M. et al. 2019. “Rat-borne diseases at the horizon. A systematic review on infectious agents carried by rats in Europe 1995–2016.” `https://pmc.ncbi.nlm.nih.gov/articles/PMC6394330/`
- [CE-RN1] IUCN Global Invasive Species Database: `Rattus norvegicus`. `https://www.iucngisd.org/gisd/speciesname/Rattus+norvegicus`
- [CE-RR1] IUCN Global Invasive Species Database: `Rattus rattus`. `https://www.iucngisd.org/gisd/pdf.php?sc=19`
- [CE-MM1] CABI Compendium: `Mus musculus` (house mouse). `https://www.cabidigitallibrary.org/doi/10.1079/cabicompendium.35218`
- [CE-MM2] IUCN Global Invasive Species Database: `Mus musculus`. `https://www.iucngisd.org/gisd/pdf.php?sc=97`
- [CE-AS1] IUCN Red List: `Apodemus sylvaticus`. `https://www.iucnredlist.org/species/pdf/197270811`
- [CE-AS2] EUNIS: `Apodemus sylvaticus`. `https://eunis.eea.europa.eu/species/11233`
- [CE-AF1] EUNIS: `Apodemus flavicollis`. `https://eunis.eea.europa.eu/species/11231`
- [CE-TICK] Mihalca, A. D. & Sándor, A. D. 2013. “The role of rodents in the ecology of Ixodes ricinus and associated pathogens in Central and Eastern Europe.” `https://www.frontiersin.org/journals/cellular-and-infection-microbiology/articles/10.3389/fcimb.2013.00056/full`
- [CE-AA1] IUCN Red List PDF for `Apodemus agrarius`. `https://www.iucnredlist.org/species/pdf/111875852`
- [CE-HANTA] Hönig, V. et al. 2022. “Orthohantaviruses in Reservoir and Atypical Hosts in the Czech Republic.” `https://journals.asm.org/doi/10.1128/spectrum.01306-22`
- [CE-MA1] Jacob, J. et al. 2020. “Europe-wide outbreaks of common voles in 2019.” `https://link.springer.com/article/10.1007/s10340-020-01200-2`
- [CE-MA2] Luque-Larena, J. J. et al. 2021. “Common Vole Populations and Tularemia Outbreaks in NW Spain.” `https://pmc.ncbi.nlm.nih.gov/articles/PMC8397442/`
- [CE-MAG1] EUNIS: `Microtus agrestis`. `https://eunis.eea.europa.eu/species/11288`
- [CE-MG1] ECDC factsheet on orthohantavirus infections. `https://www.ecdc.europa.eu/en/infectious-disease-topics/hantavirus-infection/factsheet-orthohantavirus-infections`
- [CE-MG2] Reil, D. et al. 2017. “Puumala hantavirus infections in bank vole populations: host and virus dynamics in Central Europe.” `https://link.springer.com/article/10.1186/s12898-017-0118-z`
- [CE-AV1] EUNIS: `Arvicola amphibius`. `https://eunis.eea.europa.eu/species/11822`
- [CE-AV2] Animal Diversity Web: `Arvicola amphibius`. `https://animaldiversity.org/accounts/Arvicola_amphibius/`
- [CE-CL1] CDC Emerging Infectious Diseases: “Shrews as Reservoir Hosts of Borna Disease Virus.” `https://wwwnc.cdc.gov/eid/article/12/4/05-1418_article`
- [CE-CL2] Puorger, M. E. et al. 2014. “The Bicolored White-Toothed Shrew Crocidura leucodon is an Indigenous Host of Mammalian Borna Disease Virus.” `https://pmc.ncbi.nlm.nih.gov/articles/PMC3974811/`
- [SL-L1] Cosson, J.-F. et al. 2022. “Ecology and distribution of Leptospira spp., reservoir hosts and environmental interaction in Sri Lanka.” `https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010757`
- [SL-L2] Same article in PMC full text. `https://pmc.ncbi.nlm.nih.gov/articles/PMC9518908/`
- [SL-BI1] National Red List: `Bandicota indica`. `https://www.nationalredlist.org/assessments/nrld-94573`
- [SL-BB1] Mammal Diversity Database: `Bandicota bengalensis`. `https://www.mammaldiversity.org/taxon/1003501/`
- [SL-BART] Mühldorfer, K. et al. 2021. “First Detection of Bartonella spp. in Small Mammals from Rice Storage and Processing Facilities in Myanmar and Sri Lanka.” `https://pmc.ncbi.nlm.nih.gov/articles/PMC8004705/`
- [SL-VO1] National Red List: `Vandeleuria oleracea`. `https://www.nationalredlist.org/assessments/nrld-95011`
- [SL-SM1] IUCN Global Invasive Species Database: `Suncus murinus`. `https://www.iucngisd.org/gisd/pdf.php?sc=162`
