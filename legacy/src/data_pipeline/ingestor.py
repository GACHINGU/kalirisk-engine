# =========================================================================
# Project: Kali Risk Engine
# Module: Data Pipeline - Chronological Ingestor
# DESCRPTION: Streams massive loan data in chunks to prevent memory crashes
# ==========================================================================

import os

import pandas as pd


def stream_and_filter_loans(file_path, output_path, chunk_size=50000):
    """
    Reads a massive csv file in small wheelbarrows (chunks),
    keeps only the gold columns needed for risk modelling,
    and saves the clean data out to a new file.
    """
    print(f" KaliRisk: Starting the stream pipeline for {file_path}")

    # The gold columns needed for building our credit risk model
    # Also extremely define what data type each column must be
    # This silences the mixed type warning completely
    column_types = {
        "id": str,
        "loan_amnt": float,
        "term": str,
        "int_rate": float,
        "installment": float,
        "grade": str,
        "sub_grade": str,
        "home_ownership": str,
        "annual_inc": float,
        "verification_status": str,
        "issue_d": str,
        "loan_status": str,
        "purpose": str,
        # DATA SYSTEM UPGRADE: Open the gates for raw risk signals
        "dti": float,  # The Total Structural DTI
        "fico_range_low": float,
        "fico_range_high": float,
    }
    gold_columns = list(column_types.keys())

    # Counter to track our progress
    chunk_count = 0
    total_rows_processed = 0

    # The chunksize parameter tells pandas to return an iterator (the wheelbarrow stream)
    data_stream = pd.read_csv(file_path, chunksize=chunk_size, usecols=gold_columns)

    for chunk in data_stream:
        chunk_count += 1
        total_rows_processed += len(chunk)

        print(
            f" Processing chunk #{chunk_count} ({len(chunk)} rows)... Total rows hit: {total_rows_processed}"
        )

        # If it's the very first chunk, create the file and write the header.
        # If its chunk 2, 3, 4, ..., just append the data to the bottom of the file!
        if chunk_count == 1:
            # Make sure to write/overwrite, incase we run the script multiple times
            # Every time we run the script, it erases whatever old file was sitting there, and starts building brand new
            # we achieve these using mode="w"
            chunk.to_csv(output_path, index=False, mode="w")
        else:
            chunk.to_csv(output_path, index=False, mode="a", header=False)

    print()
    print("=" * 65)
    print(" KaliRisk SUCCESS: Streaming ingestion complete!")
    print(f" Total Chunks Processed: {chunk_count}")
    print(f"Total Bank Records Filtered: {total_rows_processed}")
    print(f" Clead dataset saved to: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    input_file = "./data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
    output_file = "./data/processed/cleaned_accepted_loans.csv"

    # Make sure our output processed folder exists before saving
    os.makedirs("./data/processed", exist_ok=True)

    stream_and_filter_loans(input_file, output_file)
