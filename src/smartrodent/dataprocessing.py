"""Image preprocessing and CLIP-based filtering utilities.

The helpers in this module support the exploratory BioTrove workflow in
``notebooks/process_biotrove.ipynb``. They load an OpenAI CLIP model, preprocess
image batches, compare images against short text prompts, separate confident from
ambiguous prompt matches, and visualize the results for manual review.
"""

from itertools import product
from math import sqrt

import clip
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


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


class CocoDatasetCreator:
    def __init__(self, path_to_data: str, dataset_output_path: str):
        pass

    # TODO:
