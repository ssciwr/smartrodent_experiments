# Preparing an Image Dataset for YOLO26 Fine-Tuning

For YOLO26/Ultralytics fine-tuning, the key thing to understand is the **Ultralytics YOLO object detection dataset format**. The model-specific YOLO26 documentation is useful, but the dataset layout follows the same standard YOLO detection format used across Ultralytics models.

## Useful Resources

- [Ultralytics detection dataset format](https://docs.ultralytics.com/datasets/detect/)  
  The most relevant page for organizing images, bounding boxes, labels, and `data.yaml`.

- [Ultralytics fine-tuning guide](https://docs.ultralytics.com/guides/finetuning-guide/)  
  Useful once the dataset is already organized.

- [Ultralytics YOLO26 model page](https://docs.ultralytics.com/models/yolo26/)  
  Model-specific details, though dataset preparation still follows the standard Ultralytics detection format.

- [Roboflow YOLO format explanation](https://roboflow.com/formats/yolo)  
  Often clearer than the Ultralytics docs for understanding YOLO annotation files and coordinate conversion.

## Expected Dataset Layout

A typical YOLO object detection dataset should look like this:

```text
my_dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   ├── val/
│   │   └── img101.jpg
│   └── test/          # optional
│       └── img201.jpg
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   └── img002.txt
│   ├── val/
│   │   └── img101.txt
│   └── test/          # optional
│       └── img201.txt
└── data.yaml
```

For every image file:

```text
images/train/img001.jpg
```

there should be a matching label file:

```text
labels/train/img001.txt
```

The image and label filenames must have the same basename.

## YOLO Label File Format

Each image gets one `.txt` label file. Each line in that file represents one object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Example:

```text
0 0.5123 0.4388 0.1200 0.2100
2 0.3000 0.7000 0.0800 0.1500
```

Where:

```text
class_id = integer index into your class list
x_center = box center x / image width
y_center = box center y / image height
width    = box width / image width
height   = box height / image height
```

All bounding box values must be **normalized between 0 and 1**.

## `data.yaml` Example

Example `data.yaml`:

```yaml
path: /absolute/path/to/my_dataset

train: images/train
val: images/val
test: images/test  # optional

names:
  0: cat
  1: dog
  2: person
```

The `names` mapping defines the class IDs used in the `.txt` label files.

## Bounding Box Conversion

If your bounding boxes are currently stored in pixel coordinates as:

```text
xmin, ymin, xmax, ymax
```

convert them to YOLO format like this:

```python
x_center = ((xmin + xmax) / 2) / image_width
y_center = ((ymin + ymax) / 2) / image_height
width = (xmax - xmin) / image_width
height = (ymax - ymin) / image_height
```

If your bounding boxes are instead stored as:

```text
xmin, ymin, width, height
```

convert them like this:

```python
x_center = (xmin + width / 2) / image_width
y_center = (ymin + height / 2) / image_height
width = width / image_width
height = height / image_height
```

## Training Command Example

After the dataset is organized, fine-tuning can be started with a command like:

```bash
yolo detect train model=yolo26n.pt data=/absolute/path/to/my_dataset/data.yaml epochs=100 imgsz=640
```

Adjust the model checkpoint, number of epochs, image size, and dataset path as needed.

## Key Takeaway

The unusual part of YOLO dataset organization is that labels are usually **not stored in one CSV or JSON file**. Instead, YOLO expects:

- one `.txt` label file per image
- the same basename for image and label file
- mirrored `images/` and `labels/` directory trees
- normalized bounding boxes in `x_center y_center width height` format
- class IDs that correspond to the `names` section in `data.yaml`
