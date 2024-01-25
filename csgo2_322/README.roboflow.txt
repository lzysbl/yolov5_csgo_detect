
csgo2 - v1 2024-01-24 9:36pm
==============================

This dataset was exported via roboflow.com on January 24, 2024 at 1:36 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 334 images.
Person2 are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 41 percent of the image
* Random rotation of between -6 and +6 degrees
* Random brigthness adjustment of between -14 and +14 percent
* Random exposure adjustment of between -11 and +11 percent
* Salt and pepper noise was applied to 0.73 percent of pixels


