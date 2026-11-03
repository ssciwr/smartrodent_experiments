import ultralytics
import torch
import yaml
import mlflow

# TODO: supcon loss
# TODO: hard negative sampling
# TODO: add mlflow logging


if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available. Using GPU for training.")
    else:
        print("GPU is not available. Using CPU for training.")

    # TODO
