import logging
import os

from typing import Optional, List, Tuple

import itertools

from tqdm import tqdm


def from_kopernikus(dir_with_datasets: str,
                    save_train_to: str,
                    save_val_to: str,
                    images_labels_subdirs: Tuple[str] = ("images", "labels"),
                    train_val_subdirs: Tuple[str] = ("train", "val"),
                    image_extensions: Tuple[str] = (".jpg", ".jpeg", ".png")) -> bool:
    """Converts kopernikus labels to the format needed for this repo

    dir_with_datasets: Absolute path to a directory with datasets.
        It's assumed the following structure of directories:
        ```
        dir_with_datasets
            ├── dataset_1
            │   ├── dataset.yaml
            │   ├── images
            │   │   ├── train
            │   │   │   ├── image1.jpg
            │   │   │   ├── image2.jpeg
            │   │   │   ├── image3.png
            │   │   │   └── ...
            │   │   ├── val
            │   │   │   ├── image1.jpg
            │   │   │   ├── image2.jpeg
            │   │   │   ├── image3.png
            │   │   │   └── ...
            │   ├── labels
            │   │   ├── train
            │   │   │    ├── image1.txt
            │   │   │    ├── image2.txt
            │   │   │    ├── image3.txt
            │   │   │    └── ...
            │   │   ├── val
            │   │   │   ├── image1.txt
            │   │   │   ├── image2.txt
            │   │   │   ├── image3.txt
            │   │   │   └── ...
            │   │   └── ...
            │   └── ...
            ├── dataset_2
            │   ├── dataset.yaml
            │   ├── images
            │   │   ├── train
            │   │   │   └── ...
            │   │   └── ...
            │   ├── labels
            │   │   ├── val
            │   │   │   └── ...
            │   │   └── ...
            │   └── ...
            └── ...
        ```
        So, you download the directories 'dataset_*' from AWS server,
        run the scripts 'download_script.sh' which creates the directories
        'images' and 'labels' and feed the directory with datasets to this
        function.

    save_to: An output file in the below format:
        ```
        image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
        image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
        ...
        ```
    """
    logging.info("Converting labels...")

    datasets = get_complete_datasets(
        dir_with_datasets, images_labels_subdirs, train_val_subdirs
    )
    if not datasets:
        logging.error(
            f"No complete datasets are found in the directory: "
            f"'{dir_with_datasets}'. Exit."
        )
        return False
    msg = "\n".join(datasets)
    logging.info(f"Datasets found: {msg}")

    train_images_subdir = os.path.join(
        images_labels_subdirs[0], train_val_subdirs[0],
    )
    train_labels_subdir = os.path.join(
        images_labels_subdirs[1], train_val_subdirs[0],
    )
    train_labels_fname = os.path.join(save_train_to, "train.txt")
    are_train_labels_converted = write_labels(
        dir_with_datasets,
        datasets,
        train_labels_fname,
        train_images_subdir,
        train_labels_subdir,
        image_extensions
    )

    val_images_subdir = os.path.join(
        images_labels_subdirs[0], train_val_subdirs[1],
    )
    val_labels_fname = os.path.join(save_train_to, "val.txt")
    val_labels_subdir = os.path.join(
        images_labels_subdirs[1], train_val_subdirs[1],
    )
    are_val_labels_converted = write_labels(
        dir_with_datasets,
        datasets,
        val_labels_fname,
        val_images_subdir,
        val_labels_subdir,
        image_extensions
    )

    return are_train_labels_converted and are_val_labels_converted


def get_complete_datasets(dir_with_datasets: str,
                          images_labels_subdirs: Tuple[str],
                          train_val_subdirs: Tuple[str]) -> List[str]:
    """Checks subdirectories of 'dir_with_datasets' and returns a list
    of those of them which contain the following subdirectories:
        * images/train
        * images/val
        * labels/train
        * labels/val
    """
    if not exists_and_non_empty(dir_with_datasets):
        logging.error(
            f"The directory '{dir_with_datasets}' does NOT exist or is EMPTY."
        )
        return []

    datasets = []
    for ds in os.listdir(dir_with_datasets):
        ds_path = os.path.join(dir_with_datasets, ds)
        if os.path.isdir(ds_path):
            datasets.append(ds)

    if not datasets:
        logging.error(
            f"The directory {dir_with_datasets} does NOT contain "
            f"subdirectories."
        )
        return []

    complete_datasets = []

    for ds in datasets:

        abs_ds_path = os.path.join(dir_with_datasets, ds)

        is_dataset_complete = True

        for subdirs in itertools.product(images_labels_subdirs, train_val_subdirs):

            abs_subdir_path = os.path.join(abs_ds_path, *subdirs)
            if not exists_and_non_empty(abs_subdir_path):
                logging.warning(
                    f"The directory '{ds}/{os.path.join(*subdirs)}'"
                    f"does NOT exist or is EMPTY. "
                    f"The dataset '{ds}' will NOT be processed. "
                    f"Continue..."
                )
                is_dataset_complete = False

        if is_dataset_complete:
            complete_datasets.append(ds)

    msg = "\n".join(complete_datasets)
    logging.info(f"Complete datasets:\n{msg}")

    return complete_datasets


def exists_and_non_empty(abs_dir_path: str,
                         contains_subdirs: Optional[List[str]] = None) -> bool:

    if not os.path.isdir(abs_dir_path):
        return False

    if not os.path.exists(abs_dir_path):
        return False

    subdirs = os.listdir(abs_dir_path)
    if not subdirs:
        return False

    if contains_subdirs and not set(contains_subdirs).issubset(subdirs):
        return False

    return True


def write_labels(dir_with_datasets: str,
                 datasets: List[str],
                 save_to: str,
                 images_subdir: str,
                 labels_subdir: str,
                 image_extensions: Tuple[str]) -> bool:

    with open(save_to, "w") as output_file:

        for ds in datasets:

            ds_label_subdir = os.path.join(ds, labels_subdir)

            abs_label_dir = os.path.join(dir_with_datasets, ds_label_subdir)
            label_fnames = [f for f in os.listdir(abs_label_dir) if f.endswith(".txt")]

            ds_images_subdir = os.path.join(ds, images_subdir)
            image_fnames = os.listdir(os.path.join(dir_with_datasets, ds_images_subdir))

            logging.info(
                f"Writing labels for the dataset '{ds}' ({abs_label_dir}). "
                f"Images subdirectory: '{ds_images_subdir}'...\n"
            )

            for label_fname in tqdm(label_fnames):

                image_fname = find_image(label_fname, image_fnames, image_extensions)
                if image_fname is None:
                    logging.info(
                        f"No images for label '{label_fname}'. Continue..."
                    )
                    continue

                new_line = ""

                abs_kopernikus_fname = os.path.join(abs_label_dir, label_fname)
                with open(abs_kopernikus_fname, "r") as label_f:

                    while True:

                        line = label_f.readline().replace("\n", "")
                        if not line:
                            break

                        obj_id, x1, y1, x2, y2 = line.split(" ")
                        new_line += ",".join([x1, y1, x2, y2, obj_id]) + " "

                if new_line:

                    output_file.write(
                        os.path.join(ds_images_subdir, image_fname) + " " +
                        new_line[:-1] + "\n"
                    )

    is_output_file_not_empty = os.stat(save_to).st_size > 0

    return is_output_file_not_empty


def find_image(label_fname: str,
               image_file_list: List[str],
               image_extensions: Tuple[str]) -> Optional[str]:

    basename = os.path.splitext(label_fname)[0]

    for ext in image_extensions:

        image_fname = basename + ext

        if image_fname in image_file_list:
            return image_fname

    return None


if __name__ == "__main__":

    import argparse

    cli = argparse.ArgumentParser()

    cli.add_argument(
        "--dir-with-datasets", "-d", type=str, required=True,
        help="Absolute path of a directory with datasets."
    )
    cli.add_argument(
        "--save-train-to", "-t", type=str, required=True,
        help="Absolute directory path to save 'train.txt' to."
    )
    cli.add_argument(
        "--save-val-to", "-v", type=str, required=True,
        help="Absolute directory path to save 'val.txt' to."
    )

    args = cli.parse_args()

    is_ok = from_kopernikus(
        args.dir_with_datasets, args.save_train_to, args.save_val_to
    )

    logging.info(f"All done?: [{is_ok}]. Exit.")
