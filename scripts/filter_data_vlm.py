from pathlib import Path
import argparse
from smartrodent import VLMFilter

# Configs express dataset paths relative to the directory that *contains* the
# repo (e.g. "smartrodent_experiments/datasets/..."), so the same config works
# whether the repo is checked out under .../Development/ or .../projects/.
# dvc runs stages from the repo root, so cwd alone can't resolve that prefix.
PARENT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    default_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "filter_data_vlm_config_animal.yaml"
    )
    parser = argparse.ArgumentParser(
        description="Filter image data using a VLM backend."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_config,
        help=f"Path or list of paths to YAML config file (default: {default_config})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # chain configs together in the order given if we have multiple.
    # we can run a filter pipeline in this way.
    detector = VLMFilter.from_config(args.config)
    res_df = detector.filter_data()
    detector.save_results(res_df)
