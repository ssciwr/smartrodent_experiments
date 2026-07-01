"""Image preprocessing and CLIP-based filtering utilities.

The helpers in this module support the exploratory BioTrove workflow in
``notebooks/process_biotrove.ipynb``. They load an OpenAI CLIP model, preprocess
image batches, compare images against short text prompts, separate confident from
ambiguous prompt matches, and visualize the results for manual review.
"""

from itertools import product
import json
from math import sqrt
import shutil
from pathlib import Path
import clip
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from math import ceil, floor
import yaml

from .base import YoloDatasetCreatorBase, DataPreprocessorBase


class DataPreprocessorBioTrove(DataPreprocessorBase):
    def __init__(
        self,
    ):
        pass

    # TODO


class ImageFilter:
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
        self.model, self.preprocess = clip.load(model)
        self.id_tol = id_tol
        self.prompts = prompts

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

    def plot_images(self, images: list, figsize=(24, 28)):
        """Plot a list of images in a near-square matplotlib grid.

        Args:
            images: Images accepted by ``Axes.imshow``.
            figsize: Figure size passed to ``plt.subplots``.

        Returns:
            The ``(fig, axs)`` pair from matplotlib.

        Raises:
            RuntimeError: If no suitable grid shape can be found.
        """

        def get_optimal_squaring(n: float):
            """Return the smallest tested grid that fits all images."""
            if n < 0:
                raise RuntimeError("n < 0 forbidden")

            # Start from a square root estimate, then test the neighboring grid
            # sizes. This keeps the display compact without requiring a broad
            # search over all possible row/column counts.
            nx = int(sqrt(n))
            ny = nx
            nx_candidates = [nx - 1, nx, nx + 1]
            ny_candidates = [ny - 1, ny, ny + 1]

            n = 2 * len(images)
            frames = None
            for nx, ny in product(nx_candidates, ny_candidates):
                if nx * ny < len(images):
                    continue

                if nx * ny < n:
                    frames = (nx, ny)
                    n = nx * ny

            return frames

        nxny = get_optimal_squaring(len(images))

        if nxny is None:
            raise RuntimeError(
                f"Could not find optimal squaring for num. images {len(images)}"
            )

        else:
            nx, ny = nxny
            fig, axs = plt.subplots(nx, ny, figsize=figsize)
            axs = axs.flatten()
            for i, image in enumerate(images):
                axs[i].imshow(image)

            fig.tight_layout()

            return fig, axs

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
            print("i: ", i, ", prompt: ", self.prompts[i])
            labeled_data[self.prompts[i]].append(img)

        return labeled_data

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


class YoloDatasetCreatorFromSpeciesnet(YoloDatasetCreatorBase):
    def __init__(
        self,
        path_to_data: str,
        path_to_labels: str,
        dataset_output_path: str,
        class_names: list[str],
        image_directory: str = "boxed",
        confidence_threshold: float = 0.1,
        labels_to_filter: list[str] = [
            "animal",
        ],
    ):
        super().__init__(path_to_data, path_to_labels, dataset_output_path, class_names)
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = labels_to_filter
        self.labels = self._load_speciesnet_predictions()
        self.image_directory = image_directory

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
        # Placeholder for label filtering logic
        # TODO: filter stuff that is the same object/ is a duplicate detection
        # - Non-maximum suppression (NMS) could be applied here to remove overlapping detections
        # - based on IoU (Intersection over Union) thresholding
        # - or use IoA / containment checks to remove detections that are fully contained within others
        # - add parameter for supression sensitivity
        return detections

    def _preprocess_labels(self, raw_labels: dict) -> dict:
        label_data = dict()
        for pred in raw_labels["predictions"]:
            filepath = Path(pred["filepath"])

            key = filepath.name  # Use the filename as the key for label_data
            label = filepath.parent.name  # Parent directory name is the species name

            if key not in label_data:
                label_data[key] = dict()

            filtered_detections = self._filter_labels(pred["detections"])

            for detection in filtered_detections:
                if detection is None:
                    # TODO: how to classify things in which there is nothing?
                    print(f"No detection found for {key}")
                    continue

                confidence = detection.get("conf", detection.get("confidence", 0.0))
                if (
                    confidence < self.confidence_threshold
                    or detection["label"] not in self.allowed_classes
                ):
                    continue

                bbox = detection["bbox"]

                # TODO: filter predictions if they overlap or likely represent the same object
                # if that happens it can confuse the model during training.

                # SpeciesNet stores normalized (x_min, y_min, width, height),
                # while YOLO labels need normalized (x_center, y_center, width, height).
                width = bbox[2]
                height = bbox[3]
                x_min = bbox[0]
                y_min = bbox[1]
                x_center = x_min + width / 2
                y_center = y_min + height / 2

                label_data[key]["bbox"] = [x_center, y_center, width, height]
                label_data[key]["label"] = label
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
                label_info = preprocessed_labels.get(img_name)

                if label_info is None or label_info == {}:
                    print(f"No label info found for {img_name}, skipping.")
                    continue
                bbox = label_info["bbox"]
                label = label_info["label"]

                # Convert to YOLO format: class_index x_center y_center width height
                class_index = self.class_names.index(label)
                yolo_bbox = [
                    class_index,
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                ]

                label_file_path = (
                    Path(self.dataset_output_path)
                    / "labels"
                    / split_name
                    / f"{img_path.stem}.txt"
                )

                with open(label_file_path, "w") as f:
                    f.write(" ".join(map(str, yolo_bbox)))
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
