"""Export the registered classifier as an ONNX bundle from one YAML configuration."""

import argparse
from pathlib import Path

import yaml

from smartrodent.onnx_bundle import OnnxBundleExporter


def main() -> None:
    """Write the bundle described by the configuration and print its directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args()
    settings = yaml.safe_load(args.configuration.read_text(encoding="utf-8"))
    bundle = OnnxBundleExporter(**settings).export()
    print(f"Bundle written: {bundle}")


if __name__ == "__main__":
    main()
