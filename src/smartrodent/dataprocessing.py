"""Image preprocessing and CLIP-based filtering utilities.

The helpers in this module support the exploratory BioTrove workflow in
``notebooks/process_biotrove.ipynb``. They load an OpenAI CLIP model, preprocess
image batches, compare images against short text prompts, separate confident from
ambiguous prompt matches, and visualize the results for manual review.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import json
import shutil
from pathlib import Path
import clip
import matplotlib.pyplot as plt
import numpy as np
import open_clip
from PIL import Image
import torch
from math import ceil, floor
import yaml
from ultralytics import YOLOE
from ultralytics.utils.metrics import box_iou

from .base import YoloDatasetCreatorBase, ImageFilterBase


class ImageFilterCLIP(ImageFilterBase):
    """Filter and inspect images by comparing CLIP image/text embeddings.

    Args:
        model: CLIP model name accepted by ``clip.load`` (for example,
            ``"RN50x16"`` or ``"ViT-B/32"``).
        prompts: Text labels to compare against each image. The model sees each
            prompt as ``"This is {prompt}"`` when computing similarities.
        id_tol: Similarity spread threshold used to mark images as undecided. If
            the best and worst prompt scores for an image differ by less than or
            equal to this value, the image is treated as ambiguous.
    """

    def __init__(
        self,
        model: str,
        prompts: list[str],
        id_tol: float = 0.02,
    ):
        """Load a CLIP model and store filtering prompts/settings."""
        # ``clip.load`` returns both the model and the matching image transform.
        # Keeping the transform on the instance ensures preprocessing matches the
        # chosen model architecture.
        self.id_tol = id_tol
        self.prompts = prompts
        self._load_model(model)

    def _load_model(self, model):
        self.model, self.preprocess = clip.load(model)

    def print(self):
        """Print a compact summary of the loaded CLIP model."""
        input_resolution = self.model.visual.input_resolution
        context_length = self.model.context_length
        vocab_size = self.model.vocab_size

        print(
            "Model parameters:",
            f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}",
        )
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)

    def preprocess_images(self, impage_paths):
        """Load image files and convert them into CLIP-ready tensors.

        Args:
            impage_paths: Iterable of image paths readable by Pillow.

        Returns:
            A tuple ``(original_images, images)`` where ``original_images`` are
            RGB ``PIL.Image`` objects for plotting and ``images`` are tensors
            produced by the CLIP preprocessing transform.
        """
        original_images = []
        images = []
        for filename in impage_paths:
            # Convert all inputs to RGB so grayscale or palette images have the
            # channel layout expected by CLIP.
            image = Image.open(filename).convert("RGB")

            original_images.append(image)
            images.append(self.preprocess(image))
        return original_images, images

    def compute_similarity(self, images):
        """Compute CLIP cosine similarities between prompts and images.

        Args:
            images: Preprocessed image tensors, such as the second return value
                from ``preprocess_images``.

        Returns:
            A tensor shaped ``(len(prompts), len(images))``. Each column contains
            the prompt scores for one image.
        """
        # Stack the preprocessed images into a batch and move both images and text
        # tokens to CUDA because CLIP inference is expensive on CPU.
        image_input = torch.tensor(np.stack(images)).cuda()
        text_tokens = clip.tokenize(["This is " + desc for desc in self.prompts]).cuda()

        # Inference only: gradients are unnecessary and would waste memory.
        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()

        # Normalize embeddings so the matrix multiplication below is cosine
        # similarity rather than an unnormalized dot product.
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = text_features.cpu().detach() @ image_features.cpu().detach().T
        return similarity

    def filter_similarities(
        self,
        similarity: torch.Tensor,
    ):
        """Split image similarity columns into decided and undecided groups.

        A column is considered undecided when the spread between its maximum and
        minimum prompt scores is less than or equal to ``self.id_tol``.

        Args:
            similarity: Tensor shaped ``(num_prompts, num_images)``.

        Returns:
            ``(decided_idx, undecided_idx, decided, undecided)`` where the index
            lists refer to original image positions and the tensors contain the
            corresponding similarity columns.
        """
        # Store columns separately first so we can return both the filtered score
        # tensors and the original image indices needed for plotting/inspection.
        undecided = []
        decided = []
        undecided_idx = []
        decided_idx = []
        for i in range(similarity.shape[1]):
            minimum_col = similarity[:, i].min()
            maximum_col = similarity[:, i].max()

            if (maximum_col - minimum_col) <= self.id_tol:
                undecided.append(similarity[:, i].unsqueeze(1))
                undecided_idx.append(i)
            else:
                decided.append(similarity[:, i].unsqueeze(1))
                decided_idx.append(i)

        # Keep empty results as tensors so downstream plotting and decision code
        # can check shapes consistently.
        decided = torch.cat(decided, dim=1) if len(decided) > 0 else torch.empty(0)
        undecided = (
            torch.cat(undecided, dim=1) if len(undecided) > 0 else torch.empty(0)
        )

        return decided_idx, undecided_idx, decided, undecided

    def decisions(self, imgs, decided):
        """Group decided images by their highest-scoring prompt.

        Args:
            imgs: Images corresponding to the columns in ``decided``.
            decided: Similarity tensor containing only confident image columns.

        Returns:
            A dictionary mapping each prompt to the list of images assigned to it.
        """
        best = decided.argmax(dim=0).tolist()
        labeled_data = {}
        for p in self.prompts:
            labeled_data[p] = []

        for i, img in zip(best, imgs):
            labeled_data[self.prompts[i]].append(img)

        return labeled_data

    def decided_label(
        self, preprocessed_images: list
    ) -> tuple[str | None, torch.Tensor]:
        """Return the winning prompt only when the prompt decision is clear.

        This is a convenience wrapper around ``compute_similarity`` and
        ``filter_similarities`` for the common one-image-at-a-time filtering workflow.
        It returns ``None`` when the image lands in the undecided bucket.
        """
        similarity = self.compute_similarity(preprocessed_images)
        decided_idx, _, decided, _ = self.filter_similarities(similarity)
        if len(decided_idx) != 1:
            return None, similarity
        return self.prompts[int(decided.argmax(dim=0).item())], similarity

    def copy_with_structure(
        self,
        src: str | Path,
        dst_root: str | Path,
        source_root: str | Path | None = None,
    ) -> Path:
        """Copy one image to ``dst_root`` while preserving relative structure.

        If ``source_root`` is supplied, the destination path preserves the source path
        below that root. Otherwise only the filename is copied.
        """
        src = Path(src)
        dst_root = Path(dst_root)
        if source_root is None:
            relative_path = Path(src.name)
        else:
            try:
                relative_path = src.relative_to(Path(source_root))
            except ValueError:
                relative_path = Path(src.name)
        dst = dst_root / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    def copy_by_decision(
        self,
        image_paths: list[str | Path],
        accepted_prompt: str,
        filtered_root: str | Path,
        rejected_root: str | Path,
        undecided_root: str | Path,
        source_root: str | Path | None = None,
    ) -> dict:
        """Copy images into filtered, rejected, and undecided directories.

        Images whose clear winning prompt equals ``accepted_prompt`` go to
        ``filtered_root``. Images with a clear different prompt go to
        ``rejected_root``. Images whose prompt scores are within ``id_tol`` or that
        raise an image/model error go to ``undecided_root`` instead of being mixed
        into rejected images.

        Returns:
            A dictionary with ``summary``, ``by_prompt``, and ``errors`` entries for
            quick inspection of the copy run.
        """
        filtered_root = Path(filtered_root)
        rejected_root = Path(rejected_root)
        undecided_root = Path(undecided_root)
        filtered_root.mkdir(parents=True, exist_ok=True)
        rejected_root.mkdir(parents=True, exist_ok=True)
        undecided_root.mkdir(parents=True, exist_ok=True)

        summary = Counter()
        by_prompt = defaultdict(Counter)
        errors = []

        for image_path in image_paths:
            image_path = Path(image_path)
            try:
                _, preprocessed = self.preprocess_images([image_path])
                label, _ = self.decided_label(preprocessed)

                if label is None:
                    self.copy_with_structure(image_path, undecided_root, source_root)
                    reason = "undecided"
                elif label == accepted_prompt:
                    self.copy_with_structure(image_path, filtered_root, source_root)
                    reason = "filtered"
                else:
                    self.copy_with_structure(image_path, rejected_root, source_root)
                    reason = "rejected"

                summary[reason] += 1
                by_prompt[label if label is not None else "undecided"][reason] += 1
            except Exception as exc:
                self.copy_with_structure(image_path, undecided_root, source_root)
                summary["error"] += 1
                by_prompt["error"]["error"] += 1
                errors.append((str(image_path), repr(exc)))

        return {
            "summary": dict(summary),
            "by_prompt": {key: dict(value) for key, value in by_prompt.items()},
            "errors": errors,
        }

    def plot_sim_score(
        self,
        similarity: torch.Tensor,
        original_images: list[np.ndarray],
        figsize=(20, 14),
        yfontsize=18,
    ):
        """Plot prompt/image similarity scores as a heatmap with thumbnails.

        Args:
            similarity: Tensor shaped ``(num_prompts, num_images)``.
            original_images: Display images corresponding to the similarity
                columns.
            figsize: Figure size passed to matplotlib.
            yfontsize: Font size for prompt labels on the y-axis.

        Returns:
            ``(fig, ax)`` for the generated plot, or ``None`` if there is nothing
            to plot.
        """
        if similarity.shape == (0,) or len(original_images) == 0:
            print("nothing to plot")
            return

        count = similarity.shape[0]

        plt.figure(figsize=figsize)
        plt.imshow(similarity, vmin=0.1, vmax=0.3)

        plt.yticks(range(count), self.prompts, fontsize=yfontsize)
        plt.xticks([])

        # Draw the image thumbnails above the heatmap columns so each score column
        # can be visually checked against its source image.
        for i, image in enumerate(original_images):
            plt.imshow(
                image,
                extent=(i - 0.5, i + 0.5, -1.7, -0.7),
                origin="lower",
                aspect="equal",
                zorder=3,
                clip_on=False,
            )

        # Annotate every heatmap cell with the numeric similarity score. This is
        # useful when deciding whether ``id_tol`` should be stricter or looser.
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(
                    x,
                    y,
                    f"{similarity[y, x]:.2f}",
                    ha="center",
                    va="center",
                    size=12,
                )

        fig = plt.gcf()
        ax = plt.gca()
        for side in ["left", "top", "right", "bottom"]:
            ax.spines[side].set_visible(False)

        ax.set_xlim(-0.5, similarity.shape[1] - 0.5)
        ax.set_ylim(count - 0.5, -1.8)

        plt.title("Cosine similarity between text and image features", size=20)

        return fig, ax


class ImageFilterBiotroveClip(ImageFilterCLIP):
    """Filter images with the local BioTrove-CLIP OpenCLIP checkpoint.

    BioTrove-CLIP uses OpenCLIP rather than the original OpenAI ``clip`` package, so
    model construction, tokenization, and the text prompt template differ from
    ``ImageFilterCLIP``. The public filtering/plotting methods keep the same tensor
    shape as ``ImageFilterCLIP``: rows are prompts and columns are images.
    """

    def __init__(
        self,
        model: str,
        prompts: list[str],
        id_tol: float = 0.02,
        prompt_template: str = "This is a photo of {}",
        model_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ):
        self.prompt_template = prompt_template
        self.model_dir = (
            Path(model_dir) if model_dir is not None else self._default_model_dir()
        )
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        super().__init__(model=model, prompts=prompts, id_tol=id_tol)

    def _default_model_dir(self) -> Path:
        """Return the repository-local BioTrove-CLIP model directory."""
        return Path(__file__).resolve().parents[2] / "models" / "BioTrove-CLIP"

    def _load_model(self, model: str | Path):
        """Load BioTrove-CLIP through OpenCLIP using a local checkpoint."""
        model_dir = self.model_dir.expanduser().resolve()
        checkpoint_path = Path(model).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = model_dir / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()

        if not model_dir.exists():
            raise FileNotFoundError(
                f"BioTrove-CLIP model directory not found: {model_dir}"
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"BioTrove-CLIP checkpoint not found: {checkpoint_path}"
            )

        openclip_model_name = f"local-dir:{model_dir}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            openclip_model_name,
            pretrained=str(checkpoint_path),
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(openclip_model_name)
        self.model.eval()

    def compute_similarity(self, images):
        """Compute BioTrove-CLIP prompt scores for each image.

        Detection uses BioTrove-CLIP logits followed by a softmax over prompts, so
        this image-filter adapter returns those same prompt probabilities in the
        ``(num_prompts, num_images)`` layout expected by ``ImageFilterCLIP`` helpers.
        OpenCLIP preprocessors already return tensors, so this method stacks them with
        ``torch.stack`` rather than going through NumPy.
        """
        if len(images) == 0:
            return torch.empty((len(self.prompts), 0))

        image_input = torch.stack(images).to(self.device)
        text_tokens = self.tokenizer(
            [self.prompt_template.format(desc) for desc in self.prompts]
        ).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        return probabilities.cpu().detach().T


class ImageFilterYoloE(ImageFilterCLIP):
    """Filter images with YOLOE text-prompt detections.

    YOLOE produces boxed detections rather than whole-image embeddings. To keep the
    ``ImageFilterCLIP`` downstream API usable, this adapter converts each image's
    detections into a prompt-by-image score matrix whose entries are the maximum box
    confidence for each requested prompt in each image. Images with no detections get
    all-zero columns and are therefore treated as undecided by ``filter_similarities``.
    """

    def __init__(
        self,
        model: str,
        prompts: list[str],
        id_tol: float = 0.02,
        task: str = "detect",
        predict_conf: float = 0.001,
        device: str | int | None = None,
    ):
        self.task = task
        self.predict_conf = predict_conf
        self.device = device
        super().__init__(model=model, prompts=prompts, id_tol=id_tol)

    def _load_model(self, model: str | Path):
        """Load YOLOE and install text-prompt classes."""
        self.model = YOLOE(self.resolve_local_model(model), task=self.task)
        if self.prompts:
            self.model.set_classes(
                self.prompts,
                self.model.get_text_pe(self.prompts),
            )
        self.preprocess = None

    def preprocess_images(self, impage_paths):
        """Load originals for plotting and keep filenames for YOLOE inference."""
        original_images = []
        image_paths = []
        for filename in impage_paths:
            image = Image.open(filename).convert("RGB")
            original_images.append(image)
            image_paths.append(str(Path(filename).resolve()))
        return original_images, image_paths

    def compute_similarity(self, images):
        """Run YOLOE and return max class confidence per prompt/image.

        Args:
            images: Image paths returned by ``preprocess_images``.

        Returns:
            A tensor shaped ``(len(prompts), len(images))`` with detection
            confidences instead of cosine similarities.
        """
        if len(images) == 0:
            return torch.empty((len(self.prompts), 0))

        predict_kwargs = {
            "source": images,
            "batch": len(images),
            "save": False,
            "verbose": False,
            "conf": self.predict_conf,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)
        similarity = torch.zeros((len(self.prompts), len(images)), dtype=torch.float32)
        prompt_to_idx = {prompt: idx for idx, prompt in enumerate(self.prompts)}

        for image_idx, result in enumerate(results):
            boxes = result.boxes
            if not boxes or len(boxes) == 0:
                continue

            for cls, conf in zip(boxes.cls.cpu(), boxes.conf.cpu(), strict=False):
                label = result.names[int(cls)]
                prompt_idx = prompt_to_idx.get(label)
                if prompt_idx is None:
                    continue
                similarity[prompt_idx, image_idx] = max(
                    similarity[prompt_idx, image_idx], float(conf)
                )

        return similarity


class YoloDatasetCreatorFromSpeciesnet(YoloDatasetCreatorBase):
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
    ):
        super().__init__(
            path_to_image_data,
            path_to_labels,
            dataset_output_path,
            class_names,
            train_val_test_split,
            rng_seed=rng_seed,
            confidence_threshold=confidence_threshold,
            IoU_threshold=IoU_threshold,
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

    def _filter_labels(self, detections: list) -> list:

        # this uses non-maximum suppression (NMS) to filter out overlapping detections based on their confidence scores.
        # The assumption is that allowed classes represent **the same** class which got conflated by
        # speciesnet, so there is no extra class equality check below anymore.

        _detections = deepcopy(detections)  # don´t want to modify the og list

        # keep only those detections in which we are confident enough and which are in
        # the classes we assume represent detections of relevant animals (allowed_classes)
        _detections = [
            detection
            for detection in _detections
            if detection.get("conf", detection.get("confidence", 0.0))
            >= self.confidence_threshold
            and detection.get("label", detection.get("class")) in self.allowed_classes
        ]
        _detections.sort(
            key=lambda x: x.get("conf", x.get("confidence", 0.0)), reverse=True
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
            bbox_d1 = bbox_xywh_to_xyxy_tensor(d1["bbox"])

            for j in range(i + 1, len(_detections)):
                if j in suppressed:
                    continue

                bbox_d2 = bbox_xywh_to_xyxy_tensor(_detections[j]["bbox"])
                iou = box_iou(bbox_d1, bbox_d2).item()

                if iou > self.IoU_threshold:
                    suppressed.add(j)

        return keep

    def _preprocess_labels(self, raw_labels: dict) -> dict:
        label_data = dict()
        for pred in raw_labels["predictions"]:
            filepath = Path(pred["filepath"])

            key = filepath.name  # Use the filename as the key for label_data
            label = filepath.parent.name  # Parent directory name is the species name

            if key not in label_data:
                label_data[key] = dict()

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

    def _split_train_val_test(
        self,
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
        for label_dir in Path(self.path_to_image_data).iterdir():
            img_paths = [
                p
                for p in Path(label_dir).iterdir()
                if p.is_file() and p.suffix.lower() in self.img_types
            ]

            self.rng.shuffle(img_paths)
            n_images = len(img_paths)
            n_train = int(ceil(n_images * self.train_frac))
            n_val = int(floor(n_images * self.val_frac))
            n_test = n_images - n_train - n_val

            if n_train + n_val + n_test != n_images:
                raise ValueError(
                    f"Image split counts do not sum to total for {label_dir}: "
                    f"{n_train} + {n_val} + {n_test} != {n_images}"
                )

            split_images = {
                "train": img_paths[:n_train],
                "val": img_paths[n_train : n_train + n_val],
                "test": img_paths[n_train + n_val :],
            }

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
        return paths, assignments

    def _write_labels(
        self,
        paths: dict[str, list[Path]],
        assignments: dict[str, list[Path]],
        preprocessed_labels: dict,
    ) -> str:

        # write the labels in parallel to the images in the train/val/test splits

        for split_name in ["train", "val", "test"]:
            for img_path in assignments[split_name]:
                img_name = img_path.name
                label_list = preprocessed_labels.get(img_name)
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
                    for yolo_label in yolo_labels:
                        f.write(" ".join(map(str, yolo_label)) + "\n")
        datayaml = {
            "path": self.dataset_output_path,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.classes,
        }

        with open(Path(self.dataset_output_path) / "data.yaml", "w") as f:
            yaml.safe_dump(datayaml, f)

        return self.dataset_output_path
