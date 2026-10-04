"""Download NASA CMAPSS dataset."""

import os
import zipfile
import urllib.request

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
DATA_DIR = "data"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "cmapss.zip")

    if not os.path.exists(os.path.join(DATA_DIR, "train_FD001.txt")):
        print("Downloading CMAPSS dataset...")
        urllib.request.urlretrieve(URL, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)

        # Move files from nested folder if needed
        nested = os.path.join(DATA_DIR, "CMAPSSData")
        if os.path.isdir(nested):
            for f in os.listdir(nested):
                os.rename(os.path.join(nested, f), os.path.join(DATA_DIR, f))
            os.rmdir(nested)

        os.remove(zip_path)
        print("Done.")
    else:
        print("Dataset already exists.")


if __name__ == "__main__":
    download()
