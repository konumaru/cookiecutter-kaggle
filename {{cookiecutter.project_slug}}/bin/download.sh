#!/bin/bash

set -exo pipefail

download_competition() {
    DIR_NAME=$1

    mkdir -p data/
    mkdir -p data/input

    # Download competition data
    kaggle competitions download -c "$DIR_NAME"

    # Unzip
    unzip -o "$DIR_NAME.zip" -d "./data/input"

    # Remove zip
    rm "$DIR_NAME.zip" 
}

download_dataset() {
    DATASET_NAME=$1
    ARR=${DATASET_NAME/\// }
    DIR_NAME=${ARR[1]}

    mkdir -p data/
    mkdir -p data/external

    # Download dataset data
    kaggle datasets download -d "$DATASET_NAME"

    # Unzip
    unzip "$DIR_NAME.zip" -d "data/external/$DIR_NAME"

    # Remove zip
    rm "$DIR_NAME.zip"
}

download_kernels_dataset() {
    DATASET_NAME=$1
    IFS='/' read -ra ARR <<< "$DATASET_NAME"
    DIR_NAME="${ARR[1]}"

    mkdir -p data/
    mkdir -p data/external

    # Download dataset data
    kaggle kernels output $DATASET_NAME -p data/external/$DIR_NAME
}


# Download competitoin data.
download_competition {{cookiecutter.competition_url_name}}

# Download datasets.
# download_dataset cdeotte/census-data-for-godaddy
