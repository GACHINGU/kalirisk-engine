# =====================================================================
# PROJECT: KaliRisk Engine
# MODULE: Data Ingestion Tool
# DESCRIPTION: Environment token injection script to bypass Windows paths
# =====================================================================


# 1. CRITICAL: Inject the keys BEFORE importing the kaggle library
# This makes sure the program has permission to download the files
# os.environ["KAGGLE_USERNAME"] = "legend"  # We use a dummy name since token handles auth
# os.environ["KAGGLE_KEY"] = (
# "KGAT_caad7d8848e71e8e350b41fa40b27070"  # the secret password
# )

# Now it is safe to bring the Kaggle agent into the script, since the keys are ready
# from kaggle.api.kaggle_api_extended import KaggleApi


def run_dataset_download() -> None:
    """
    This function connects into the internet, logs into kaggle,
    and downloads banking data files into a local folder

    Args:
        None

    Returns:
            None
    Side Effects:
            Downloads large files from the internet and saves them
            into the './data/raw' folder on your computer.
    """
    print(" KaliRisk: Waking up the Kaggle API agent...")
    api = KaggleApi()

    print(" KaliRisk: Verifying security credentials...")
    # log in using the secret keys we set up at the top
    api.authenticate()

    # These is where the online data lives and where we want to save it
    dataset_address = "wordsforthewise/lending-club"
    target_folder = "./data/raw"

    print(f" KaliRisk: Connecting to Kaggle to fetch '{dataset_address}'...")
    print(" Please wait, downloading massive banking files...")

    # Go get the files, bring them to our computer, and unzip them so we can read them
    api.dataset_download_files(dataset=dataset_address, path=target_folder, unzip=True)

    print(f"✅ KaliRisk SUCCESS: Raw CSV files saved inside '{target_folder}'!")


if __name__ == "__main__":
    # these starts the whole download process
    run_dataset_download()
