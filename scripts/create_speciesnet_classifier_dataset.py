import pandas as pd
from pathlib import Path
import yaml

path_to_data = (
    "/home/hmack/projects/smartrodent_experiments/datasets/full_dataset/filtered_kept"
)
out_path = "/home/hmack/projects/smartrodent_experiments/datasets/full_dataset/speciesnet_training_dataset"

Path(out_path).mkdir(parents=True, exist_ok=True)

specieslist = [
    "Rattus norvegicus",
    "Rattus rattus",
    "Mus musculus",
    "Apodemus agrarius",
    "Apodemus flavicollis",
    "Apodemus sylvaticus",
    "Arvicola amphibius",
    "Microtus arvalis",
    "Clethrionomys glareolus",
    "Myodes glareolus",
    "Microtus agrestis",
    "Crocidura leucodon",
    "Sorex araneus",
    "Sorex minutus",
    "Sorex coronatus",
    # rodents in sri lanka
    "Suncus murinus",
    "Bandicota indica",
    "Bandicota bengalensis",
    "Mus booduga",
    "Vandeleuria oleraceus",
    # predators of snakes and rodents (birds, mammals )
    "Vulpes vulpes",  # red fox
    "Buteo buteo",  # common buzzard
    "Meles meles",  # european badger
    "Ciconia ciconia",  # white stork
    "Ardea cinerea",  # grey heron
    # snakes
    "Vipera berus",  # common european adder
    "Coronella austriaca",  # smooth snake
    "Natrix natrix",  # grass snake
    "Natrix helvetica",  # barred grass snake
    # snakes in sri lanka
    "Ptyas mucosa",
    "Bungarus caeruleus",
    "Naja naja",
    "Daboia russelii",
    "Coelognathus helena",
    # predators from sri lanka
    "Spilornis cheela",
    "Varanus bengalensis",
    "Centropus sinensis",
    "Pavo cristatus",
]
datalists = {"filename": [], "category": [], "location": []}
for dir in Path(path_to_data).iterdir():
    if dir.name in specieslist:
        for imgfile in dir.iterdir():
            if imgfile.suffix == ".jpg":
                datalists["filename"].append(str(imgfile.resolve()))
                datalists["category"].append(dir.name)
                datalists["location"].append("unknown")

specieslist_df = pd.DataFrame(datalists)
specieslist_df.to_csv(Path(out_path) / "species_data.csv")

# map species to various higher orders
biological_mapping = {
    "Rattus norvegicus": "Rattus",
    "Rattus rattus": "Rattus",
    "Mus musculus": "Mus",
    "Apodemus agrarius": "Apodemus",
    "Apodemus flavicollis": "Apodemus",
    "Apodemus sylvaticus": "Apodemus",
    "Arvicola amphibius": "Arvicola",
    "Microtus arvalis": "Microtus",
    "Clethrionomys glareolus": "Clethrionomys",
    "Myodes glareolus": "Myodes",
    "Microtus agrestis": "Microtus",
    "Crocidura leucodon": "Crocidura",
    "Sorex araneus": "Sorex",
    "Sorex minutus": "Sorex",
    "Sorex coronatus": "Sorex",
    # rodents in sri lanka
    "Suncus murinus": "Suncus",
    "Bandicota indica": "Bandicota",
    "Bandicota bengalensis": "Bandicota",
    "Mus booduga": "Mus",
    "Vandeleuria oleraceus": "Vandeleuria",
    # predators of snakes and rodents (birds, mammals )
    "Vulpes vulpes": "Vulpes",  # red fox
    "Buteo buteo": "Buteo buteo",  # common buzzard
    "Meles meles": "Meles",  # european badger
    "Ciconia ciconia": "Ciconia",  # white stork
    "Ardea cinerea": "Ardea",  # grey heron
    # snakes
    "Vipera berus": "Vipera",  # common european adder
    "Coronella austriaca": "Coronella",  # smooth snake
    "Natrix natrix": "Natrix",  # grass snake
    "Natrix helvetica": "Natrix",  # barred grass snake
    # snakes in sri lanka
    "Ptyas mucosa": "Ptyas",
    "Bungarus caeruleus": "Bungarus",
    "Naja naja": "Naja",
    "Daboia russelii": "Daboia",
    "Coelognathus helena": "Coelognathus",
    # predators from sri lanka
    "Spilornis cheela": "Spilornis",
    "Varanus bengalensis": "Varanus",
    "Centropus sinensis": "Centropus",
    "Pavo cristatus": "Pavo",
}

biological_mapping_lists = {"input": [], "output": []}
for species, family in biological_mapping.items():
    biological_mapping_lists["input"].append(species)
    biological_mapping_lists["output"].append(family)

biological_mapping = pd.DataFrame(biological_mapping_lists)
biological_mapping.to_csv(Path(out_path) / "family_data.csv")

# map species to general common names
common_name_mapping = {
    "Rattus norvegicus": "rat",
    "Rattus rattus": "rat",
    "Mus musculus": "mouse",
    "Apodemus agrarius": "mouse",
    "Apodemus flavicollis": "mouse",
    "Apodemus sylvaticus": "mouse",
    "Arvicola amphibius": "vole",
    "Microtus arvalis": "vole",
    "Clethrionomys glareolus": "vole",
    "Myodes glareolus": "vole",
    "Microtus agrestis": "vole",
    "Crocidura leucodon": "shrew",
    "Sorex araneus": "shrew",
    "Sorex minutus": "shrew",
    "Sorex coronatus": "shrew",
    # rodents in sri lanka
    "Suncus murinus": "shrew",
    "Bandicota indica": "rat",
    "Bandicota bengalensis": "rat",
    "Mus booduga": "mouse",
    "Vandeleuria oleraceus": "mouse",
    # predators of snakes and rodents (birds, mammals )
    "Vulpes vulpes": "fox",  # red fox
    "Buteo buteo": "buzzard",  # common buzzard
    "Meles meles": "badger",  # european badger
    "Ciconia ciconia": "stork",  # white stork
    "Ardea cinerea": "heron",  # grey heron
    # snakes
    "Vipera berus": "snake",  # common european adder
    "Coronella austriaca": "snake",  # smooth snake
    "Natrix natrix": "snake",  # grass snake
    "Natrix helvetica": "snake",  # barred grass snake
    # snakes in sri lanka
    "Ptyas mucosa": "snake",
    "Bungarus caeruleus": "snake",
    "Naja naja": "snake",
    "Daboia russelii": "snake",
    "Coelognathus helena": "snake",
    # predators from sri lanka
    "Spilornis cheela": "eagle",
    "Varanus bengalensis": "lizard",
    "Centropus sinensis": "coucal",
    "Pavo cristatus": "peafowl",
}

commonname_mapping_lists = {"input": [], "output": []}
for species, family in common_name_mapping.items():
    commonname_mapping_lists["input"].append(species)
    commonname_mapping_lists["output"].append(family)

common_name_mapping = pd.DataFrame(commonname_mapping_lists)
common_name_mapping.to_csv(Path(out_path) / "common_name_data.csv")
