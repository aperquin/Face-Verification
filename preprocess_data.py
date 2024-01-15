# %%
from scipy.io import loadmat
from pathlib import Path
import random
import pandas as pd
import stone
from tqdm import tqdm
import argparse

def read_triplets_from_file(mat_filepath:Path|str) -> list[tuple[str, str, int]]:
    """Read the triplets describing whether two videos feature the same face or not from the corresponding MatLab file.
    According to the dataset documentation:
    'The Splits is a data structure dividing the data set to 10 independent splits.
    Each triplet in the Splits is in the format of (index1, index2, is_same_person), where index1 and index2 are the indices in the mat_names structure.
    All together 5000 pairs divided equaly to 10 independent splits, with 2500 same pairs and 2500 not-same pairs.'

    Args:
        mat_filepath (Path|str): Path to the file describing the dataset

    Returns:
        list[tuple[str, str, int]]: A list that describes the dataset in the format `[name_video1, name_video2, label]`, where `label` indicates whether the two videos feature the same person.
    """
    # Read the MatLab file containing the list of splits
    data_dict = loadmat(mat_filepath)

    result = []
    for split_list in data_dict['Splits']:
        for video_index_1, video_index_2, is_same_person in zip(*split_list):
            # The file was generated in MatLab where indexes start at 1, whereas they begin at 0 in Python
            video_index_1 -= 1
            video_index_2 -= 1
            
            result.append((data_dict['video_names'][video_index_1][0][0], data_dict['video_names'][video_index_2][0][0], is_same_person))

    return result

def choose_image_and_find_face_location(video_name:str, nb_images:int=1) -> list[dict]:
    """For a given video, choose randomly one or more images (default=1) from that video.
    Also, extracts the position of the face in the chosen images.
    In the YouTubeFaces dataset, this position does not need to be predicted as it is given in the "{celeb_name}.labeled_faces.txt" file.
    Store those information as a dictionary along with the name of the subject.

    Args:
        video_name (str): Name of the video to extract data from.
        nb_images (int), optional: Number of images to select in the video. Default to 1.

    Returns:
        list[dict]: A list that describes that describe each image selected. Each image is represented by a dictionary with the following keys:
            - 'x': horizontal position of the center of the face in the original image
            - 'y': vertical position of the center of the face in the original image
            - 'width': width of the face in the original image
            - 'height': height of the face in the original image
            - 'filepath': path to the original image
            - 'subject_name': Name of the person featured around which the bound box is centered
    """    
    # Chose an image in the video
    video_folderpath = videos_folderpath / video_name
    all_images = list(video_folderpath.iterdir())
    chosen_images = random.sample(all_images, nb_images)
    
    # Find the location of the face in each of the image
    celeb_name = video_name.split('/')[0]
    face_location_filepath = videos_folderpath / f"{celeb_name}.labeled_faces.txt"
    df = pd.read_csv(face_location_filepath, sep=',', header=None, index_col=0, names=["key", "x", "y", "width", "height"], usecols=[0,2,3,4,5])
    face_positions = []
    for img_filepath in chosen_images:
        # Convert the image filepath to the format used as a key in the "{celeb_name}.labeled_faces.txt" file
        img_key = str(img_filepath).replace(str(videos_folderpath), '')[1:].replace("/", "\\")
        face_positions.append(df.loc[img_key].to_dict())

    # Format the result as a list of dictionary
    result = []
    for img_filepath, face_position in zip(chosen_images, face_positions):
        face_position["filepath"] = img_filepath
        face_position["subject_name"] = celeb_name
        result.append(face_position)

    return result

def expand_triplets(triplet_list:list[tuple[str, str, int]]) -> list[tuple[dict, dict, int]]:
    """For each triplet:
    - Select a single image to represent each of the two videos and find its filepath
    - Find the position of the face of the subject in each image.
    - Keep track of the name of the subject

    Args:
        triplet_list (list[tuple]): List of triplet videos. Each triplet is described by the name of the two videos and the label `same`/`different`

    Returns:
        list[tuple[dict, dict, int]]: A list describing each triplet where videos have been represented by a single image each. The info on each image is encoded in a dictionary, cf. `choose_image_and_find_face_location`
    """
    new_triplet_list = []
    for video_name_1, video_name_2, label in tqdm(triplet_list):
        if video_name_1 == video_name_2:
            # To make sure the image selected is not the same, we sample two images of the video at once
            info = choose_image_and_find_face_location(video_name_1, 2)
            new_triplet_list.append((info[0], info[1], label))
        else:
            info1 = choose_image_and_find_face_location(video_name_1, 1)[0]
            info2 = choose_image_and_find_face_location(video_name_2, 1)[0]
            new_triplet_list.append((info1, info2, label))

    return new_triplet_list

def parse_gender_file(gender_filepath:Path) -> set:
    """Parse a file listing all the samples featuring a person of a given gender.

    Args:
        gender_filepath (Path): File listing all the samples in the dataset featuring a person of a given gender

    Returns:
        set: Set of people of a given gender
    """
    lines = gender_filepath.read_text().splitlines()

    name_set = set()
    for line in lines:
        name = "_".join(line.split("_")[:-1]) # Remove the suffix and extension of the filename to get the name of the subject
        name_set.add(name)

    return name_set

def find_gender(name:str) -> str:
    """Given the set of people belonging to each gender, return the gender of a given name

    Args:
        name (str): Name of the person of which to retrieve the gender.

    Returns:
        str: Gender of the person. Can be 'F' (Female), 'M' (Male), 'N/A' (gender not present in the metadata)
    """
    if name in female_set:
        gender = "F"
    elif name in male_set:
        gender = "M"
    else:
        gender = "N/A"
    return gender

def predict_skin_tone(img_filepath:Path) -> str:
    """Attempt to predict the skin tone of the person featured in a given image.

    Args:
        img_filepath (Path): Image featuring a person of which to predict the skin tone.

    Returns:
        str: Skin tone predicted
    """
    # From original filepath, find related cropped/aligned image
    img_filepath = str(img_filepath)
    img_filepath = img_filepath.replace("frame_images_DB", "aligned_images_DB")
    img_filepath = img_filepath.split("/")
    img_filepath[-1] = "aligned_detect_" + img_filepath[-1]
    img_filepath = Path("/".join(img_filepath))

    # Predict skin tone from cropped/aligned image
    info_dict = stone.process(img_filepath, image_type="color")
    skin_tone = info_dict["faces"][0]["skin_tone"]

    return skin_tone

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract metadata from the YouTubeFaces dataset and save them as a `.csv` file.')
    parser.add_argument('dataset_folderpath', type=str, help='Path to the folder containing the dataset and external gender metadata.')
    parser.add_argument('output_folderpath', type=str, help='Path to the folder where to save the generated metadata.')

    args = parser.parse_args()
    dataset_folderpath = Path(args.dataset_folderpath)
    output_folderpath = Path(args.output_folderpath)

    split_filepath = dataset_folderpath / "YouTubeFaces" / "meta_data"/ "meta_and_splits.mat"
    videos_folderpath = dataset_folderpath / "YouTubeFaces" / "frame_images_DB"
    metadata_output_filepath = output_folderpath / "metadata.csv"
    gender_metadata_folder = dataset_folderpath / "Additional_Labels" 

    # %% Load the description of the dataset as triplets (video1, video2, label)
    triplets = read_triplets_from_file(split_filepath)

    # %% Convert each triplet (video1, video2, label) into (img1, img2, label)
    new_triplets = expand_triplets(triplets)

    # %% Format the data as a Pandas Dataframe so it can be exported easily to a '.csv' file and for ease of analysis
    series_list = []
    for i, triplet in enumerate(new_triplets):
        series_1 = pd.Series(triplet[0]).add_suffix("_1")
        series_2 = pd.Series(triplet[1]).add_suffix("_2")
        series_3 = pd.Series({"label": triplet[2]})
        row = pd.concat([series_1, series_2, series_3])
        series_list.append(row)
    metadata_df = pd.DataFrame(series_list)

    # %%  Add gender information (if available) to each sample in the selected dataset
    female_set, male_set = parse_gender_file(gender_metadata_folder / "females.txt"), parse_gender_file(gender_metadata_folder / "males.txt")
    metadata_df["gender_1"] = metadata_df["subject_name_1"].apply(find_gender)
    metadata_df["gender_2"] = metadata_df["subject_name_2"].apply(find_gender)

    # %% Try to predict the skin tone of each person featured in the selected dataset
    tqdm.pandas()
    metadata_df["skin_tone_1"] = metadata_df["filepath_1"].progress_apply(predict_skin_tone)
    metadata_df["skin_tone_2"] = metadata_df["filepath_2"].progress_apply(predict_skin_tone)

    # %% Save the metadata into a `.csv` file
    metadata_df.to_csv(metadata_output_filepath, index=False)
# %%
