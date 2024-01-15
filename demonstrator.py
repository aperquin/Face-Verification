# %%
from src.FaceTracker import FaceTracker
import numpy as np
from tqdm import tqdm
import argparse

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform face tracking on a video.')
    parser.add_argument('input_video_filepath', type=str, help='Path to the video to perform face tracking on.')
    parser.add_argument('output_video_filepath', type=str, help='Path to the file to save the video after face tracking.')
    parser.add_argument('svm_classifier_filepath', type=str, help='Path to SVM classifier performing the label prediction "same"/"different" from the cosine distance between two face embeddings.')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of the batch to split the video into during processing.')

    args = parser.parse_args()
    input_video_filepath = args.input_video_filepath
    output_video_filepath = args.output_video_filepath
    svm_classifier_filepath = args.svm_classifier_filepath
    batch_size = args.batch_size

    # %% Load the face tracker
    face_tracker = FaceTracker(svm_classifier_filepath)
    face_tracker.reinit_face_db()

    # %% Read the video
    video_frames = face_tracker.read_video(input_video_filepath)

    # %% Find the number of batches needed to process the whole video
    if len(video_frames) % batch_size == 0:
        nb_batches = int(len(video_frames) / batch_size)
    else:
        nb_batches = int(len(video_frames) / batch_size) + 1

    # %% Perform face tracking + identification
    new_video = []
    for batch_index in tqdm(range(nb_batches)):
        video_batch = video_frames[batch_index*batch_size:(batch_index+1)*batch_size]
        
        # Detect the faces
        boxes_found = face_tracker.detect_faces(video_batch)

        # Assign an identity to each of the detected faces
        found_ids = face_tracker.identify(video_batch, boxes_found)

        # Put a square around the detected faces and write their ID
        new_video.extend(face_tracker.rectangle_detected_faces(video_batch, boxes_found, found_ids))
    
    face_tracker.write_video(np.array(new_video), output_video_filepath)

