import os
import shutil
import json
import random


def index_dataset(path: str, start_index: int):
    files = os.listdir(path)
    for i, file in enumerate(files):
        if file != "clean":
            shutil.copy(path + file, f"{path}/clean/{start_index + i}.png")
    print(f"Moved {len(files)} files")


def load_annotations(file_path: str) -> dict:
    """_summary_

    Args:
        file_path: path to coco annotation file.

    Returns:
        coco format annotation dict.
    """
    with open(file_path) as fp:
        annotations = json.load(fp)

    for image in annotations["images"]:
        image["file_name"] = image["file_name"].split("/")[-1]

    return annotations


def prepare_coco_dataset(folder_path: str, train_percentage: float):
    """Split a coco compatible dataset into train and val datasets.

    Args:
        path: path to main coco dataset folder
        train_percentage: percentage of images that will be allocated to `train` folder
    """
    valid_ext = ["jpeg", "png"]
    # Load the data
    print("📊 Loading the data")
    files = os.listdir(f"{folder_path}/images")
    files = [file for file in files if file.split(".")[1] in valid_ext]
    annotations = load_annotations(file_path=f"{folder_path}/result.json")
    print(f"📊 Loaded {len(files)} images with annotations")

    # Create the folders
    print("🗂️ Creating the folders")
    if os.path.exists(f"{folder_path}/train"):
        shutil.rmtree(f"{folder_path}/train")
    if os.path.exists(f"{folder_path}/val"):
        shutil.rmtree(f"{folder_path}/val")
    if os.path.exists(f"{folder_path}/annotations"):
        shutil.rmtree(f"{folder_path}/annotations")
    os.mkdir(f"{folder_path}/train")
    os.mkdir(f"{folder_path}/val")
    os.mkdir(f"{folder_path}/annotations")
    print("🗂️ Created the folders : train, val, annotations")

    # Randomize the data
    random.Random().shuffle(files)

    # Split the dataset
    print("🪢 Starting to split the images")
    image_mapping = {}
    split = int(train_percentage * len(files))
    for i in range(split):
        file = files[i]
        src = f"{folder_path}/images/{file}"
        dest = f"{folder_path}/train"
        image_mapping[file] = "train"
        shutil.copy(src=src, dst=dest)

    for i in range(split, len(files)):
        file = files[i]
        src = f"{folder_path}/images/{file}"
        dest = f"{folder_path}/val"
        image_mapping[file] = "val"
        shutil.copy(src=src, dst=dest)
    print(
        f"🪢 Finished to split the images, train = {split} val = {len(files) - split}"
    )

    # Split the annotations$
    print("📝 Splitting the annotations")
    split_annotations = {
        "train": {"images": [], "categories": [], "annotations": [], "info": []},
        "val": {"images": [], "categories": [], "annotations": [], "info": []},
    }
    id_image_mapping = {}
    for image in annotations["images"]:
        split_annotations[image_mapping[image["file_name"]]]["images"].append(image)
        id_image_mapping[image["id"]] = image_mapping[image["file_name"]]

    for annotation in annotations["annotations"]:
        image_id = annotation["image_id"]
        split_annotations[id_image_mapping[image_id]]["annotations"].append(annotation)

    split_annotations["train"]["categories"] = annotations["categories"]
    split_annotations["val"]["categories"] = annotations["categories"]
    split_annotations["train"]["info"] = annotations["info"]
    split_annotations["val"]["info"] = annotations["info"]

    # Save annotations
    with open(f"{folder_path}/annotations/instances_train.json", "w+") as fp:
        json.dump(split_annotations["train"], fp)
    with open(f"{folder_path}/annotations/instances_val.json", "w+") as fp:
        json.dump(split_annotations["val"], fp)
    print("📝 Finished to split the annotations")


if __name__ == "__main__":
    prepare_coco_dataset(folder_path="../../dataset/mk-coco", train_percentage=0.75)
    # index_dataset(path="../../dataset/", start_index=53)
