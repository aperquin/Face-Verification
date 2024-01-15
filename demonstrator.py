# %%
from src.FaceTracker import FaceTracker
import numpy as np
from tqdm import tqdm

# %%
if __name__ == "__main__":
    batch_size = 32

    # %% Load the face tracker
    face_tracker = FaceTracker("/output/svm_classifier.pkl")
    face_tracker.reinit_face_db()

    # %% Read the video
    video_frames = face_tracker.read_video("/dataset/Web_Videos/pexels_videos_1721303_(720p).mp4")

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
        print("==========")
        print(len(boxes_found))
        for box in boxes_found:
            if box is None:
                print(box)
            else:
                print(box.shape)

        # Assign an identity to each of the detected faces
        found_ids = face_tracker.identify(video_batch, boxes_found)
        print("~~~~~~~~")
        print(len(boxes_found))
        for box in boxes_found:
            if box is None:
                print(box)
            else:
                print(box.shape)

        # Put a square around the detected faces and write their id
        new_video.extend(face_tracker.rectangle_detected_faces(video_batch, boxes_found, found_ids))
    
    face_tracker.write_video(np.array(new_video), "/output/tracked.mp4")

