# -*- coding: utf-8 -*-
import glob
import logging
import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.data.mean_std_dataset import calculate_mean_std, load_mean_std
from src.data.raw_to_interim import RawToInterim


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True), default="data/raw/")
@click.argument(
    "interim_filepath", type=click.Path(exists=True), default="data/interim/"
)
@click.argument("output_filepath", type=click.Path(), default="data/processed/")
@click.argument(
    "annotation_filepath",
    type=click.Path(exists=True),
    required=False,
    default="data/raw/root_annotations/",
)
def main(input_filepath, interim_filepath, output_filepath, annotation_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    if not glob.glob(f"{interim_filepath}*"):

        logger.info("from raw to interim")
        r2i = RawToInterim(input_filepath, interim_filepath, annotation_filepath)

        r2i._image_collector()
        r2i._annotation_collector()
        logger.info(f"{r2i.length} datapoints with annotation")
        r2i._write_to_npy()

    else:

        logger.info("interim already exists")


    if not glob.glob(f"{interim_filepath}*.pickle"):
        logger.info("calculating mean and std values for the dataset")
        mean_std_dict = calculate_mean_std(interim_filepath)
    else:
        logger.info("loading mean and std values for the dataset")
        mean_std_dict = load_mean_std(interim_filepath)

    print(mean_std_dict)

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
