# Face-Verification

## Installation process

This repo can be installed easily using [docker](#using-docker).

It should also be possible to install it on an Ubuntu 20.04 machine using [pip](#using-pip) (not tested yet).

### Using docker

First, alter the docker compose file to point the volumes toward the correct location on your system :
- `/dataset` = Where to find the YouTubeFaces dataset and external metadata.
- `/output` = Where to save the results of this repo

The image can be built using :
```bash
docker compose build
```

Then, run the container using :
```bash
docker compose run -p 8888:8888 test_technique
```

### Using pip

To mitigate potential conflicts with other libraries/projet, first create a new python environment (eg. using conda), launch it then install the following packages using pip :
```bash
pip install torch facenet-pytorch jupyterlab scipy pandas skin-tone-classifier matplotlib torch-summary scikit-learn
```

Warning: The installation with pip has not been test. Some packages might be missing !

## Downloading the data

This section describes how to download the dataset(s) and external metadata needed to run this repo. The final expected format of the `dataset` folder is the following:
```
dataset/
|-- Additional_Labels
|   |-- females.txt
|   `-- males.txt
`-- YouTubeFaces
    |-- README.txt
    |-- WolfHassnerMaoz_CVPR11.pdf
    |-- aligned_images_DB
    |-- descriptors_DB
    |-- frame_images_DB
    |-- headpose_DB
    `-- meta_data
```

### Downloading YouTubeFaces

The YouTubeFaces dataset can be downloaded on its [homepage](https://www.cs.tau.ac.il/~wolf/ytfaces/) after answering a GoogleForm. The dataset is made available immediately after answering the form.

### Downloading additional external metadata

Since the `YouTubeFaces` dataset and the `Labeled Faces in the Wild (LFW)` [dataset](http://vis-www.cs.umass.edu/lfw/) share their subjects, additional metadatadata created for `LFW` can be used. In particular we use gender manual [annotations](https://www.dropbox.com/sh/l3ezp9qyy5hid80/AAAjK6HdDScd_1rXASlsmELla?dl=0).

## Running the different processes

### Extract metadata 
To extract metadata from the YouTubeFaces dataset and save them in a `.csv` format, run the `preprocess_data.py` script.
```bash
python preprocess_data.py {path_to_dataset_folder} {path_to_output_folder}

# eg. in the docker container
/opt/conda/bin/python preprocess_data.py /dataset/ /output/
```
The metadata will be saved in the file `{path_to_dataset_folder}/metadata.csv`. An example result is given in `container_output/metadata.csv`

### Data analysis and selection

The data analysis is performed using jupyter notebook/lab. The data analysis also select train/test/dev data from the whole dataset.

First run the jupyter server:
```
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

Then run the `data_analysis.ipynb` notebook. If needed, change the following variables in the first cell :
- `input_file` : Path to the metadata generated in [the previous section](#extract-metadata)
- `output_folderpath` : Path of the folder to save the selected data

The script generate the following files :
- `train_metadata.csv` : data to be used for training
- `test_metadata.csv` : data to be used for testing/evaluation
- `dev_metadata.csv` : data to be used for development
- `selected_metadata.csv` : whole of the three previous set

Example of the resulting files are given in the `container_output` folder.

### Face verification

To perform face verification between two images, this repo first extract face embeddings for those two images. Then a distance between those two embeddings is computed (eg. cosine distance). Finally, the label `same_face`/`different_face` is predicted by comparing this distance to a threshold. If the distance is lower than the threshold, the `same_face` label is predicted, otherwise the `different_face` is predicted.

In this repo, we use [FaceNet](https://github.com/timesler/facenet-pytorch) to perform face embeddings extraction. Then, to implement the optimal threshold search, we use a [Linear Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC). This allows for code modularity (eg. if switching from cosine distance to cosine similarity, etc.)

To perform face verification, run the following command:
```bash
python train_sklearn.py {train_metadata} {test_metadata} {classifier_save_path}

# eg. in the docker container
/opt/conda/bin/python train_sklearn.py /output/train_metadata.csv /output/test_metadata.csv /output/svm_classifier.pkl
```

At the end of the script, the performance of the face verification pipeline is computed on the train and test set. Furthermore, the resulting classifier is saved.

Example of the resulting files are given in the `container_output` folder.

### Face tracking

In this repo, face tracking highlights the faces in a video by drawing a box around them. Furthermore, the identity of each face is tracked during the video and printed as a number inside the bounding boxes.

The detection of faces is done using MTCNN, and the identity tracking is done by using the face verification pipeline described in [the previous section](#face-verification).

To perform face tracking, use the following command :
```bash 
python demonstrator.py {input_video.mp4} {output_video.mp4} {classifier_path}

# eg. on the docker container
/opt/conda/bin/python demonstrator.py /dataset/Web_Videos/pexels-anthony-shkraba-7509468\ \(720p\).mp4 /output/tracked.mp4 /output/svm_classifier.pkl 
```

Alternatively, to perform face tracking interactively in a jupyter notebook, run the following command :
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```
Then, run the `demonstrator.ipynb` notebook.
