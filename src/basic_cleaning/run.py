#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    with wandb.init(job_type="basic_cleaning") as run:
        run.config.update(args)

        logger.info("Downloading artifact")
        local_path = wandb.use_artifact(args.input_artifact).file()
        df = pd.read_csv(local_path)

        # handling nan
        df['last_review'] = pd.to_datetime(df['last_review']) #firstly change data type
        df['last_review'] = df['last_review'].fillna(df['last_review'].min())
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
        df = df.dropna(subset=['name', 'host_name'])
        logger.info("NaN removed")

        # Drop outliers
        idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
        df = df[idx].copy()

        min_price = args.min_price
        max_price = args.max_price
        index_mask = df['price'].between(min_price, max_price)

        df = df[index_mask].copy()
        logger.info("Outliers removed")

        df['price'] = np.log1p(df['price'])
        logger.info("Target log transformation done")

        df.to_csv(args.output_artifact, index=False)
        logger.info("Saving file")

        artifact = wandb.Artifact(args.output_artifact,
                                  type=args.output_type,
                                  description=args.output_description)

        artifact.add_file(args.output_artifact)
        run.log_artifact(artifact)
        logger.info("Artifact logged")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fill with Wandb input artifact name",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Fill with Wandb output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Fill with the type. Ex.: 'Data with outliers and null values removed' ",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Fill with Wandb output artifact description",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=int,
        help="Outlier removal botton cap",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=int,
        help="Outlier removal top cap",
        required=True
    )


    args = parser.parse_args()

    go(args)
