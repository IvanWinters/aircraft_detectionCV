# aircraft_detectionCV

for Computer Vision project 3

# instructions

Due Monday October 21
Idea: Pick a project that is an _end-to-end_ computer vision task, and complete it using calls to current Vision APIs.
What is an end-to-end computer vision task?
Examples of things that are end-to-end tasks:
Detect images that have current wildfires (e.g. with data from here: https://alertcalifornia.org/)
Count people in surveillance video or time-lapse of a mall: https://www.youtube.com/watch?v=AgVLVqzdrbg
Read license plates from video sequences
for the purpose of this assigning: complete a kaggle contest (or similar)
Things that are not end to end tasks:

Running image segmentation on a number of images (why? segmentation isn't the answer to a question. It's often a useful intermediate step, but not an answer.
Doing license plate detection on a test dataset of images of license plates (why? for a real problem domain, often getting the images into a useful format is the most important and challenging part)
Running YOLO to label the objects in a scene (why? This might be _almost_ all you have to do, but if you want to count how many people are in a scene, then you have to process the YOLO output and produce a count per image).
•There are 3 options:
–The end-to-end task of your choice, with the constraint that results must be based on >1000 images.
–Any Kaggle contest that relates to images, for example, this one:
•https://www.kaggle.com/competitions/planttraits2024/leaderboard, or
•https://www.kaggle.com/competitions/tpu-getting-started
https://www.kaggle.com/competitions/isic-2024-challenge/data
–Other contests:
•https://www.aicitychallenge.org/2024-challenge-tracks/
• Propose your own project on data you care about
Good Luck!

---

# Military Aircraft Detection Project

This project aims to detect and classify military aircraft in images using object detection techniques with PyTorch.

## Dataset

The dataset contains images of military aircraft and corresponding annotations in CSV format, stored in the `data/dataset` folder.

## Project Structure

- `dataset.py`: Custom dataset class for loading images and annotations.
- `model.py`: Functions to load and modify the pre-trained model.
- `utils.py`: Utility functions and class mappings.
- `train.py`: Script to train the model.
- `evaluate.py`: Script to evaluate the model.
- `predict.py`: Script to perform inference on new images.
- `requirements.txt`: Python dependencies.
- `data/`: Contains the dataset.

## Setup Instructions

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

`python train.py`

`python evaluate.py`

`python predict.py`
