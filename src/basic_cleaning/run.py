#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    with wandb.init(job_type="basic_cleaning") as run:
        run.config.update(args)

        logger.info("Downloading artifact")
        local_path = wandb.use_artifact(args.input_artifact).file()
        df = pd.read_csv(local_path)

        # Drop outliers
        min_price = args.min_price
        max_price = args.max_price
        index_mask = df['price'].between(min_price, max_price)
        df = df[index_mask].copy()
        logger.info("Outliers removed")

        #Converting date columns to datetime
        df['last_review'] = pd.to_datetime(df['last_review'])
        logger.info("Converting data types")

        df.to_csv(args.output_artifact, index=False)
        logger.info("Saving file")

        artifact = wandb.Artifact(args.output_artifact,
                                  type=args.output_type,
                                  description=args.output_description)

        artifact.add_file(args.output_artifact)
        run.log_artifact(artifact)
        logger.info("Artifact logged")


    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################


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
