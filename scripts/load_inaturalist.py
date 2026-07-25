from smartrodent import inaturalist

config_path = (
    "/home/hmack/Development/rodent_experiments/configs/data_config_central_europe.yaml"
)

inaturalist.download_inat_data(config_path, maxlen=2000, seed=42, page=None)
