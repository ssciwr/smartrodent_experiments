"""Image preprocessing and CLIP-based filtering utilities.

The helpers in this module support the exploratory BioTrove workflow in
``notebooks/process_biotrove.ipynb``. They load an OpenAI CLIP model, preprocess
image batches, compare images against short text prompts, separate confident from
ambiguous prompt matches, and visualize the results for manual review.
"""

import json
import shutil
from collections import defaultdict
from copy import deepcopy
from math import ceil, floor
from pathlib import Path

import torch
import yaml
from ultralytics.utils.metrics import box_iou

from .base import YoloDatasetCreatorBase
from .utils import path_component


class YoloDetectorDatasetCreatorFromSpeciesnet(YoloDatasetCreatorBase):
    def __init__(
        self,
        path_to_image_data: str,
        path_to_labels: str,
        dataset_output_path: str,
        class_names: list[str],
        train_val_test_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        rng_seed: int = 42,
        confidence_threshold: float = 0.1,
        IoU_threshold: float = 0.45,
        labels_to_filter: list[str] = [
            "animal",
        ],
        create_detection_dirs: bool = True,
        background_image_dir: str | Path | None = None,
        background_class_names: list[str] | tuple[str, ...] = ("empty",),
    ):
        self.background_image_dir = (
            Path(background_image_dir) if background_image_dir else None
        )
        self.background_class_names = set(background_class_names)
        detection_class_names = [
            class_name
            for class_name in class_names
            if class_name not in self.background_class_names
        ]
        super().__init__(
            path_to_image_data,
            path_to_labels,
            dataset_output_path,
            detection_class_names,
            train_val_test_split,
            rng_seed=rng_seed,
            confidence_threshold=confidence_threshold,
            IoU_threshold=IoU_threshold,
            create_detection_dirs=create_detection_dirs,
        )
        self.allowed_classes = labels_to_filter
        self.labels = self._load_speciesnet_predictions()

    def _load_speciesnet_predictions(self) -> dict:
        """Merge per-species SpeciesNet ``predictions.json`` files.

        ``path_to_image_data`` is expected to point at a directory whose immediate
        children are species folders, each containing images plus a
        ``predictions.json`` file produced for that species.
        """
        merged_labels = {"predictions": []}
        data_root = Path(self.path_to_labels)

        for species_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
            predictions_path = species_dir / "predictions.json"
            if not predictions_path.exists():
                continue

            with open(predictions_path, "r") as f:
                species_labels = json.load(f)

            merged_labels["predictions"].extend(species_labels.get("predictions", []))

        if not merged_labels["predictions"]:
            raise FileNotFoundError(
                f"No SpeciesNet predictions found under species folders in {data_root}"
            )

        return merged_labels

    def _filter_label_items(self, detections: list) -> list[tuple[int, dict]]:
        """Filter SpeciesNet detections while preserving original detection indices.

        The detector dataset only needs the filtered detection dictionaries, but the
        classifier dataset must map each accepted detection back to a crop filename.
        SpeciesNet crop names include the original detection index, so this helper keeps
        ``(original_index, detection)`` pairs through confidence filtering, allowed-class
        filtering, and NMS.
        """

        # this uses non-maximum suppression (NMS) to filter out overlapping detections based on their confidence scores.
        # The assumption is that allowed classes represent **the same** class which got conflated by
        # speciesnet, so there is no extra class equality check below anymore.

        _detections = [
            (idx, deepcopy(detection)) for idx, detection in enumerate(detections)
        ]

        # keep only those detections in which we are confident enough and which are in
        # the classes we assume represent detections of relevant animals (allowed_classes)
        _detections = [
            (idx, detection)
            for idx, detection in _detections
            if detection.get("conf", detection.get("confidence", 0.0))
            >= self.confidence_threshold
            and detection.get("label", detection.get("class")) in self.allowed_classes
        ]
        _detections.sort(
            key=lambda x: x[1].get("conf", x[1].get("confidence", 0.0)), reverse=True
        )

        def bbox_xywh_to_xyxy_tensor(bbox: list[float]) -> torch.Tensor:
            """SpeciesNet bbox is [x_min, y_min, width, height]; box_iou expects [[x1, y1, x2, y2]]."""
            x_min, y_min, width, height = bbox
            return torch.tensor(
                [[x_min, y_min, x_min + width, y_min + height]], dtype=torch.float32
            )

        if len(_detections) == 0:
            return []

        keep = []
        suppressed = set()
        # Go over detections sorted by confidence. If a later/lower-confidence detection
        # overlaps enough with an earlier/higher-confidence detection, suppress it.
        for i, d1 in enumerate(_detections):
            if i in suppressed:
                continue

            keep.append(d1)
            bbox_d1 = bbox_xywh_to_xyxy_tensor(d1[1]["bbox"])

            for j in range(i + 1, len(_detections)):
                if j in suppressed:
                    continue

                bbox_d2 = bbox_xywh_to_xyxy_tensor(_detections[j][1]["bbox"])
                iou = box_iou(bbox_d1, bbox_d2).item()

                if iou > self.IoU_threshold:
                    suppressed.add(j)

        return keep

    def _filter_labels(self, detections: list) -> list:
        """Return only filtered SpeciesNet detection dictionaries.

        This is the detector-dataset-facing compatibility wrapper around
        ``_filter_label_items``. It keeps existing detector behavior unchanged while
        allowing the classifier dataset to use original detection indices.
        """
        return [detection for _, detection in self._filter_label_items(detections)]

    def _filter_by_observation(self, labels: dict) -> dict:

        retained_predictions = []
        for item in labels["predictions"]:
            path = Path(item["filepath"]).name
            splitpath = path.split("_")

            if len(splitpath) > 1:
                print("splitpath: ", splitpath)
            # use only the _0 (i.e., first) observation
            if len(splitpath) > 1 and splitpath[1][0] != "0":
                print("skipping: ", path)
                continue
            else:
                retained_predictions.append(item)

        labels["predictions"] = retained_predictions

        return labels

    def _preprocess_labels(self, raw_labels: dict) -> dict:
        label_data = {}
        for pred in raw_labels["predictions"]:
            filepath = Path(pred["filepath"])

            key = filepath.name  # Use the filename as the key for label_data
            label = filepath.parent.name  # Parent directory name is the species name
            if label in self.background_class_names:
                continue

            if key not in label_data:
                label_data[key] = {}

            filtered_detections = self._filter_labels(pred["detections"])

            detections_for_key = []

            for detection in filtered_detections:
                if detection is None:
                    # TODO: how to classify things in which there is nothing?
                    print(f"No detection found for {key}")
                    continue

                # SpeciesNet stores normalized (x_min, y_min, width, height),
                # while YOLO labels need normalized (x_center, y_center, width, height).

                bbox = detection["bbox"]
                width = bbox[2]
                height = bbox[3]
                x_min = bbox[0]
                y_min = bbox[1]
                x_center = x_min + width / 2
                y_center = y_min + height / 2

                label_bbox = {
                    "bbox": [x_center, y_center, width, height],
                    "label": label,
                }

                detections_for_key.append(label_bbox)

            label_data[key] = detections_for_key

        return label_data

    def _background_image_paths(self) -> list[Path]:
        """Return background/empty full-image examples, if configured.

        For YOLO detection these images are copied with empty ``.txt`` label files;
        background is not added as a detector class. For classification subclasses the
        same paths are copied into the background class directory.
        """
        if self.background_image_dir is None:
            return []

        if not self.background_image_dir.exists():
            raise ValueError(
                f"Background image directory {self.background_image_dir} does not exist"
            )

        return sorted(
            p
            for p in self.background_image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.img_types
        )

    def _split_paths(self, img_paths: list[Path]) -> dict[str, list[Path]]:
        """Shuffle and split image paths according to the configured fractions."""
        img_paths = list(img_paths)
        self.rng.shuffle(img_paths)
        n_images = len(img_paths)
        n_train = int(ceil(n_images * self.train_frac))
        n_val = int(floor(n_images * self.val_frac))
        n_test = n_images - n_train - n_val

        if n_train + n_val + n_test != n_images:
            raise ValueError(
                "Image split counts do not sum to total: "
                f"{n_train} + {n_val} + {n_test} != {n_images}"
            )

        return {
            "train": img_paths[:n_train],
            "val": img_paths[n_train : n_train + n_val],
            "test": img_paths[n_train + n_val :],
        }

    def _split_train_val_test(
        self, processed_labels
    ) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
        paths: dict[str, list[Path]] = {
            "img_paths": [],
            "label_paths": [],
        }
        for name in ["train", "val", "test"]:
            img_path = Path(self.dataset_output_path) / "images" / name

            labels_path = Path(self.dataset_output_path) / "labels" / name

            (img_path).mkdir(parents=True, exist_ok=True)
            (labels_path).mkdir(parents=True, exist_ok=True)

            paths["img_paths"].append(img_path)
            paths["label_paths"].append(labels_path)

        assignments = {"train": [], "val": [], "test": []}

        # Split independently within each label directory. This keeps the label folders
        # in each split and prevents an image path from entering more than one split.
        # Only include images retained in ``processed_labels``. ``__call__`` passes labels
        # through ``_filter_by_observation`` first, so this excludes X_1, X_2, ...
        # observations and prevents observation-level data leakage. Requiring at least
        # one processed label also means images whose detections were removed by the
        # confidence/class/NMS filters are not copied into the split.
        retained_labels_by_image = {
            image_name: {
                label_info["label"]
                for label_info in label_list
                if label_info is not None and label_info != {}
            }
            for image_name, label_list in processed_labels.items()
        }
        for label_dir in Path(self.path_to_image_data).iterdir():
            if label_dir.name in self.background_class_names:
                continue

            img_paths = [
                p
                for p in Path(label_dir).iterdir()
                if p.is_file()
                and p.suffix.lower() in self.img_types
                and label_dir.name in retained_labels_by_image.get(p.name, set())
            ]

            split_images = self._split_paths(img_paths)

            if (
                set(split_images["train"]).intersection(split_images["val"])
                or set(split_images["train"]).intersection(split_images["test"])
                or set(split_images["val"]).intersection(split_images["test"])
            ):
                raise ValueError(
                    f"Data leakage detected: image paths overlap between splits for {label_dir}"
                )

            all_assigned = (
                set(split_images["train"])
                .union(split_images["val"])
                .union(split_images["test"])
            )

            if all_assigned != set(img_paths):
                raise ValueError(
                    f"Some image paths were not assigned to any split for {label_dir}"
                )

            for split_name, split_paths in split_images.items():
                for src in split_paths:
                    dst = (
                        Path(self.dataset_output_path)
                        / "images"
                        / split_name
                        / src.name
                    )
                    shutil.copy(src, dst)
                    assignments[split_name].append(dst)

        background_split_images = self._split_paths(self._background_image_paths())
        for split_name, split_paths in background_split_images.items():
            for src in split_paths:
                dst = Path(self.dataset_output_path) / "images" / split_name / src.name
                shutil.copy(src, dst)
                assignments[split_name].append(dst)

        return paths, assignments

    def _write_labels(
        self,
        paths: dict[str, list[Path]],
        assignments: dict[str, list[Path]],
        preprocessed_labels: dict,
    ) -> Path:

        # write the labels in parallel to the images in the train/val/test splits

        for split_name in ["train", "val", "test"]:
            for img_path in assignments[split_name]:
                img_name = img_path.name
                print("image name: ", img_name)
                label_list = preprocessed_labels.get(img_name, [])
                yolo_labels = []
                for label_info in label_list:
                    if label_info is None or label_info == {}:
                        print(f"No label info found for {img_name}, skipping.")
                        continue
                    bbox = label_info["bbox"]
                    label = label_info["label"]

                    # Convert to YOLO format: class_index x_center y_center width height
                    class_index = self.class_names.index(label)
                    yolo_labels.append(
                        [
                            class_index,
                            bbox[0],
                            bbox[1],
                            bbox[2],
                            bbox[3],
                        ]
                    )

                label_file_path = (
                    Path(self.dataset_output_path)
                    / "labels"
                    / split_name
                    / f"{img_path.stem}.txt"
                )

                with open(label_file_path, "w") as f:
                    f.writelines(
                        " ".join(map(str, yolo_label)) + "\n"
                        for yolo_label in yolo_labels
                    )
        datayaml = {
            "path": self.dataset_output_path,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.classes,
        }

        with open(Path(self.dataset_output_path) / "data.yaml", "w") as f:
            yaml.safe_dump(datayaml, f)

        return Path(self.dataset_output_path)

    def create(self) -> Path:
        """Create the detector dataset and return its output directory."""
        filtered_labels = self._filter_by_observation(self.labels)
        preprocessed_labels = self._preprocess_labels(filtered_labels)
        paths, assignments = self._split_train_val_test(preprocessed_labels)
        return self._write_labels(paths, assignments, preprocessed_labels)


class YoloClassifierDatasetCreatorFromSpeciesnet(
    YoloDetectorDatasetCreatorFromSpeciesnet
):
    """Create an Ultralytics YOLO classification dataset from SpeciesNet crops.

    This assumes that SpeciesNet writes crops under per-species folders such as
    ``<species>/crops/animal/<source_stem>_<hash>_<detection_index>.jpg``. This class
    reuses te YoloDetectorDatasetCreatorFromSpeciesnet's confidence, allowed-label, and NMS filtering, then
    copies only the accepted crop files into the standard classification layout:
    ``train/<species>/``, ``val/<species>/``, and ``test/<species>/``.

    The split is performed by original source image rather than by crop. That prevents
    multiple crops from the same source image appearing in different splits, which would
    leak near-duplicate visual context into validation or test data.
    """

    def __init__(
        self,
        path_to_image_data: str,
        dataset_output_path: str,
        class_names: list[str],
        path_to_labels: str | None = None,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        rng_seed: int = 42,
        confidence_threshold: float = 0.1,
        IoU_threshold: float = 0.45,
        labels_to_filter: list[str] = [
            "animal",
        ],
        image_directory: str = "crops",
        background_image_dir: str | Path | None = None,
        background_class_name: str = "empty",
    ):
        """Configure classifier dataset creation from a SpeciesNet output tree.

        Args:
            path_to_image_data: SpeciesNet output root containing one folder per
                species, each with a ``predictions.json`` and crop directory.
            dataset_output_path: Destination root for the YOLO classification dataset.
            class_names: Species names to include. Folders not listed here are ignored.
            path_to_labels: Optional root for SpeciesNet ``predictions.json`` files.
                Defaults to ``path_to_image_data`` because crops and predictions usually
                live in the same SpeciesNet output tree.
            train_val_test_split: Fractions for train, validation, and test splits.
            rng_seed: Seed used for reproducible source-image-level splits.
            confidence_threshold: Minimum SpeciesNet detection confidence to accept.
            IoU_threshold: NMS overlap threshold for duplicate SpeciesNet detections.
            labels_to_filter: SpeciesNet detector labels to keep, for example
                ``["animal", "rodent"]``.
            image_directory: Name of the crop folder inside each species directory.
            background_image_dir: Directory containing full-frame background/empty
                examples.
            background_class_name: Classification class directory used for background
                examples. Defaults to ``"empty"``.
        """
        self.image_directory = image_directory
        self.background_class_name = background_class_name
        self.crop_records: dict[str, list[dict]] = {}
        self.missing_crops: list[dict] = []
        super().__init__(
            path_to_image_data=path_to_image_data,
            path_to_labels=path_to_labels or path_to_image_data,
            dataset_output_path=dataset_output_path,
            class_names=class_names,
            train_val_test_split=train_val_test_split,
            rng_seed=rng_seed,
            confidence_threshold=confidence_threshold,
            IoU_threshold=IoU_threshold,
            labels_to_filter=labels_to_filter,
            create_detection_dirs=False,
            background_image_dir=background_image_dir,
            background_class_names=(),
        )

    def _find_crop_path(
        self,
        species: str,
        source_path: Path,
        detection: dict,
        detection_index: int,
    ) -> Path | None:
        """Return the crop path for one accepted SpeciesNet detection if it exists.

        The crop filename is produced by ``utils.extract_crop`` and contains the source image
        stem, a source-path hash, and the original detection index. We can derive the
        stem and index here, but the hash is intentionally opaque, so this method uses a
        glob with the known stem and index.
        """
        detection_label = path_component(
            str(detection.get("label", detection.get("class", "unknown")))
        )
        crop_dir = (
            Path(self.path_to_image_data)
            / species
            / self.image_directory
            / detection_label
        )
        if not crop_dir.exists():
            return None

        source_stem = path_component(source_path.stem)
        matches = sorted(crop_dir.glob(f"{source_stem}_*_{detection_index:03d}.*"))
        matches = [p for p in matches if p.suffix.lower() in self.img_types]
        if len(matches) == 0:
            return None
        return matches[0]

    def _preprocess_labels(self, raw_labels: dict) -> dict:
        """Build accepted crop records grouped by species.

        Unlike the detector dataset, classification labels are represented by directory
        names. The preprocessing output is therefore a list of crop-copy records instead
        of YOLO bbox rows. Missing crop files are skipped and recorded in
        ``self.missing_crops`` so an exploratory run can continue while still exposing
        data-layout mismatches for review.
        """
        records_by_species: dict[str, list[dict]] = {
            species: [] for species in self.class_names
        }

        for pred in raw_labels["predictions"]:
            source_path = Path(pred["filepath"])
            species = source_path.parent.name
            if species not in records_by_species:
                continue

            # Keep the original SpeciesNet detection index through filtering because it
            # is part of the crop filename written by ``extract_crop``.
            for detection_index, detection in self._filter_label_items(
                pred.get("detections", [])
            ):
                crop_path = self._find_crop_path(
                    species=species,
                    source_path=source_path,
                    detection=detection,
                    detection_index=detection_index,
                )
                if crop_path is None:
                    self.missing_crops.append(
                        {
                            "species": species,
                            "source_path": str(source_path),
                            "detection_index": detection_index,
                            "label": detection.get(
                                "label", detection.get("class", "unknown")
                            ),
                        }
                    )
                    continue

                records_by_species[species].append(
                    {
                        "species": species,
                        "source_path": source_path,
                        "crop_path": crop_path,
                    }
                )

        if self.background_class_name in records_by_species:
            for image_path in self._background_image_paths():
                records_by_species[self.background_class_name].append(
                    {
                        "species": self.background_class_name,
                        "source_path": image_path,
                        "crop_path": image_path,
                    }
                )

        self.crop_records = records_by_species
        return records_by_species

    def _split_train_val_test(
        self,
    ) -> tuple[dict[str, list[Path]], dict[str, list[dict]]]:
        """Split accepted crop records by original source image within each species.

        Grouping by ``source_path`` keeps all crops from one original image together.
        This is more important than matching exact crop counts because SpeciesNet may
        emit several crops from one image, and splitting those independently would leak
        near-duplicates across train/val/test.
        """
        assignments: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

        for split_name in ["train", "val", "test"]:
            for species in self.class_names:
                (Path(self.dataset_output_path) / split_name / species).mkdir(
                    parents=True, exist_ok=True
                )

        for species, records in self.crop_records.items():
            records_by_source: dict[Path, list[dict]] = defaultdict(list)
            for record in records:
                records_by_source[record["source_path"]].append(record)

            source_paths = list(records_by_source)
            self.rng.shuffle(source_paths)

            n_images = len(source_paths)
            n_train = int(ceil(n_images * self.train_frac))
            n_val = int(floor(n_images * self.val_frac))
            n_test = n_images - n_train - n_val

            if n_train + n_val + n_test != n_images:
                raise ValueError(
                    f"Image split counts do not sum to total for {species}: "
                    f"{n_train} + {n_val} + {n_test} != {n_images}"
                )

            split_sources = {
                "train": source_paths[:n_train],
                "val": source_paths[n_train : n_train + n_val],
                "test": source_paths[n_train + n_val :],
            }

            for split_name, split_source_paths in split_sources.items():
                for source_path in split_source_paths:
                    assignments[split_name].extend(records_by_source[source_path])

        paths = {
            "train": [Path(self.dataset_output_path) / "train"],
            "val": [Path(self.dataset_output_path) / "val"],
            "test": [Path(self.dataset_output_path) / "test"],
        }
        return paths, assignments

    def _write_labels(
        self,
        paths: dict[str, list[Path]],
        assignments: dict[str, list[dict]],
        preprocessed_labels: dict,
    ) -> Path:
        """Copy accepted crops into YOLO classification folders and write metadata.

        YOLO classification datasets do not use per-image ``.txt`` labels. The class is
        encoded by the destination directory name, so this method only copies images and
        writes a small ``data.yaml`` for downstream training convenience.
        """
        for split_name, records in assignments.items():
            for record in records:
                src = record["crop_path"]
                dst = (
                    Path(self.dataset_output_path)
                    / split_name
                    / record["species"]
                    / src.name
                )
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        datayaml = {
            "path": self.dataset_output_path,
            "train": "train",
            "val": "val",
            "test": "test",
            "names": self.classes,
        }

        with open(Path(self.dataset_output_path) / "data.yaml", "w") as f:
            yaml.safe_dump(datayaml, f)

        return Path(self.dataset_output_path)

    def _filter_by_observation(self) -> dict:
        filtered_elements: dict[str, list[Path]] = {
            species: [] for species in self.class_names
        }

        for species, records in self.crop_records.items():
            for record in records:
                path = record["source_path"].name
                splitpath = path.split("_")
                if len(splitpath) > 1:
                    print("splitpath: ", splitpath)
                # use only the _0 (i.e., first) observation
                if len(splitpath) > 1 and splitpath[1][0] != "0":
                    print("skipping: ", record["source_path"])
                    continue
                else:
                    filtered_elements[species].append(record)

        return filtered_elements

    def create(self) -> Path:
        """Create the classifier dataset and return its output directory."""
        preprocessed_labels = self._preprocess_labels(self.labels)
        self.crop_records = self._filter_by_observation()
        paths, assignments = self._split_train_val_test()
        return self._write_labels(paths, assignments, preprocessed_labels)
