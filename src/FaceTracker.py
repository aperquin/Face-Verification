# %%
from pathlib import Path
from facenet_pytorch import MTCNN
import cv2
from PIL import Image, ImageDraw
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1
import torch
from src.ModelsDNN import fixed_image_standardization
from scipy.spatial.distance import cosine as cosine_distance


def crop_and_resize(full_img:np.ndarray, box:np.ndarray, size:int) -> np.ndarray:
    """Crop an image according to given coordinates and resize the resulting image as a square.

    Args:
        full_img (np.ndarray): Numpy array representing an image. Expected shape `(height, width, channel)`
        box (np.ndarray): Coordinates to crop the image around. Expected format [`left`, `upper`, `right`, `lower`]
        size (int): Size to resize the cropped image.

    Returns:
        cropped_resized_img (np.ndarray): The cropped and resized image. Shape `(height, width, channel)`
    """
    point_1_x = int(max(box[0], 0))
    point_1_y = int(max(box[1], 0))
    point_2_x = int(min(box[2], full_img.shape[1]))
    point_2_y = int(min(box[3], full_img.shape[0]))

    cropped_img = full_img[point_1_y:point_2_y, point_1_x:point_2_x].copy()
    cropped_resized_img = cv2.resize(cropped_img, (size, size))

    return cropped_resized_img


class FaceTracker():
    """Class defining a face tracker combining face detection (MTCNN) and face verification (FaceNet + Linear SVM)
    """
    def __init__(self, classifier_filepath:Path|str) -> None:
        """Constructor

        Args:
            classifier_filepath (Path|str): Path to the sklearn classifier trained to predict the 'same'/'different' label from cosine distances between face embeddings
        """
        # Load the model used for face detection (MTCNN)
        self.mtcnn = MTCNN(keep_all=True) # keep_all=True : We want to get all detected faces, not just the largest
        
        # Load the model used for face embeddings extraction (FaceNet)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
        # Load the model used to predict the 'same'/'different' label from cosine distances between face embeddings (Linear SVM)
        with Path(classifier_filepath).open('rb') as opened_file:
            self.classifier = pickle.load(opened_file)

        # Keeps track of the faces seen during face tracking to allow verification
        self.known_faces_embeddings = None # At the beginning, no faces are know, this array will be filled during identification

    def read_video(self, video_filepath:Path|str) -> np.ndarray:
        """ Load a video into memory, frame to frame, using openCV.

        Args:
            video_filepath (Path|str): Path to the video to load

        Returns:
            frames (np.ndarray): Loaded video. Shape `(nb_frame, height, width, channel)`
        """
        video_capture = cv2.VideoCapture(video_filepath)

        # Read the video frame by frame
        frames = []
        ret, frame = video_capture.read()
        while ret:
            frames.append(frame)
            ret, frame = video_capture.read()

        # BGR to RGB conversion
        frames = [cv2.cvtColor(frame , cv2.COLOR_BGR2RGB) for frame in frames] 

        return np.array(frames)

    def detect_faces(self, frame_batch:np.ndarray) -> list[None|np.ndarray]:
        """Detect faces present in a batch of frames using MTCNN.

        Args:
            frame_batch (np.ndarray): Batch of frames to perform face detection on. Expected shape `(nb_frame, height, width, channel)`

        Returns:
            list[None|np.ndarray]: List of size `nb_frame`. For each frame :
            - contains None if no face was dected.
            - contains an array of shape `(nb_faces, 4)`, where the four last dimensions are a bounding box arround the detected face in the format [`left`, `upper`, `right`, `lower`]
        """
        # Detect faces using MTCNN
        boxes, probas = self.mtcnn.detect(frame_batch)

        return boxes
    
    def write_video(self, frame_batch:np.ndarray, output_filepath:Path|str) -> None:
        """Write a sequence of frame as an `.mp4` video using OpenCV.

        Args:
            frame_batch (np.ndarray): Sequence of frames to write as a video. Expected format `(nb_frame, height, width, channel)`
            output_filepath (Path|str): Path to the file where the video will be saved
        """
        # Define the codec and format to save the video to.
        fps = 30
        fc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_filepath, fc, fps, (frame_batch.shape[2], frame_batch.shape[1]))
        
        # Write the video frame by frame
        for frame in frame_batch:
            frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR) # RGB to BGR conversion
            writer.write(frame)
        
        # Clean-up
        cv2.destroyAllWindows()
        writer.release()

    def compute_embeddings(self, imgs:np.ndarray) -> torch.Tensor:
        """Compute face embeddings for a batch of images

        Args:
            imgs (np.ndarray): Batch of images to compute face embeddings from. Expected shape `(batch_size, height, width, channel)` where `height==width`.

        Returns:
            torch.Tensor: Batch of face embeddings. Shape `(batch_size, 512)`
        """
        # Formatting the data as expecte by FaceNet
        imgs = fixed_image_standardization(torch.Tensor(imgs)) # FaceNet was trained using standardized images
        imgs = imgs.permute(0, 3, 1, 2) # Need to be of shape (B, C, size, size) instead of (B, size, size, C)

        # Embeddings prediction
        embeddings = self.resnet(imgs).detach().numpy()

        return embeddings
    
    def compare_face_embeddings(self, embedding_request:torch.Tensor, embeddings_db:torch.Tensor) -> np.ndarray[int]:
        """Compare a face embedding (request) to each face embedding in a list (db) and return a predicted label 'different'/'same' as 0/1 for each comparison.

        Args:
            embedding_request (torch.Tensor): The embedding of a face to use as a request. Expected shape (1, 512)
            embeddings_db (torch.Tensor): "List" of face embeddings to compare the request to. Expected shape (nb_faces, 512).

        Returns:
            np.ndarray[int]: Array containing the label predicted for each face in `embeddings_db` : 0 means 'different', 1 means 'same'. Shape (nb_faces, 1)
        """
        # Compute the cosine distance
        cosine_dists = np.array([cosine_distance(np.squeeze(embedding_request), embeddings_db_i) for embeddings_db_i in embeddings_db]).reshape(-1, 1)

        # Predict the label 'Same' / 'Different'
        predictions = self.classifier.predict(cosine_dists)

        return predictions
    
    def identify(self, video_frames:np.ndarray, boxes:list[None|np.ndarray]) -> list[None | list[int]]:
        """In a video, identify the faces highlighted in bounding boxes by comparing them to a 'database' of known faces (`self.known_faces_embeddings`).

        Args:
            video_frames (np.ndarray): The video to identify faces in. Expected shape `(nb_frame, height, width, channel)`
            boxes (list[None | np.ndarray]): List of size `nb_frame` containg bounding boxes marking the position of faces in the video. 

        Returns:
            list[None | list[int]]: List of size `nb_frame`. For each frame :
            - contains None if no face was dected.
            - contains a list of size `nb_frame` with the ID of the detected faces in the same order as in `boxes`
        """
        found_ids = []
        for frame_index, frame in enumerate(video_frames):
            # If no face was detected, no need to identitify faces
            if boxes[frame_index] is None:
                found_ids.append(None)
            
            # Else, for each detected face in the current frame, we need to find its ID if the face is in the 'database'; or add it in otherwise
            else:
                tmp_found_ids = []
                for box in boxes[frame_index]:
                    # Select the part of the image containing the current detected face
                    current_face = crop_and_resize(frame, box, 160)
                    current_face = np.expand_dims(current_face, 0)

                    # Compute the face embedding
                    face_embedding = self.compute_embeddings(current_face)

                    # If the current face is the first one detected, add it to the database directly
                    if self.known_faces_embeddings is None:
                        self.known_faces_embeddings = face_embedding
                        found_id = 0
                    else:
                        # Compare the current detected face to each known face
                        predictions = self.compare_face_embeddings(face_embedding, self.known_faces_embeddings)      

                        # If the current detected face is already known, assign it its ID
                        if sum(predictions) >= 1: # `predictions` is an array of 0 and 1 where 0 means 'different' and 1 means 'same'
                            found_id = np.where(predictions==1)[0][0] # Find the first face with the label 'same'

                        # If the current dected face is not already known, add it to the known faces and assign it an ID
                        else:
                            self.known_faces_embeddings = np.concatenate((self.known_faces_embeddings, face_embedding))
                            found_id = len(self.known_faces_embeddings) - 1
                    
                    tmp_found_ids.append(found_id)
                
                found_ids.append(tmp_found_ids)
            
        return found_ids
    
    def rectangle_detected_faces(self, video_frames:np.ndarray, boxes:list[None|np.ndarray], face_ids:list[None|list[int]]=None) -> None:
        """Draw a rectangle around the detected faces and, if provided, write the face ID in the rectangle.

        Args:
            video_frames (np.ndarray): The video to draw rectangle on. Expected shape `(nb_frame, height, width, channel)`
            boxes (list[None | np.ndarray]): List of size `nb_frame` containing bounding boxes of the faces.
            face_ids (list[None | list[int]], optional): List of size `nb_frame` containing the ID of the faces.

        Returns:
            np.ndarray: Video with rectangles drawn over (and IDs if provided). Shape `(nb_frame, height, width, channel)`
        """
        new_video = []
        # For each frame in the video
        for frame_index, frame in enumerate(video_frames):
            # Prepare the frame to be drawn over, using PIL
            frame_pil_img = Image.fromarray(frame)
            draw_img = ImageDraw.Draw(frame_pil_img)

            # If a least one face was detected for the current frame
            if boxes[frame_index] is not None: 
                # For each face detected in the current frame
                for box_index, box in enumerate(boxes[frame_index]):
                    # Write a rectangle on the image around the detected face
                    try:
                        draw_img.rectangle(box, outline='red')
                    except Exception as e:
                        print(f"Current : {box}")
                        print(f"Previous : {boxes[frame_index][box_index-1]}")
                        print(f"Next : {boxes[frame_index][box_index+1]}")
                        raise(e)

                    # If face IDs were provided, write the corresponding ID in the rectangle
                    if face_ids is not None:
                        found_id = face_ids[frame_index][box_index]
                        draw_img.text(box, str(found_id))

            new_video.append(frame_pil_img)

        return np.array(new_video)
    
    def reinit_face_db(self):
        """Re-initialize the known faces 'database'. Useful when switching from a video to another
        """
        self.known_faces_embeddings = None
    

# %% Test on two batches of 32 frames of a video
if __name__ == "__main__":
    batch_size = 32

    # %% Load the face tracker
    face_tracker = FaceTracker("/output/svm_classifier.pkl")
    face_tracker.reinit_face_db()

    # %% Read the video
    video_frames = face_tracker.read_video("/dataset/Web_Videos/pexels_videos_1721303_(720p).mp4")

    # %% Perform face tracking + identification
    new_video = []
    for batch_index in range(2):
        video_batch = video_frames[batch_index*batch_size:(batch_index+1)*batch_size]
        # Detect the faces
        boxes_found = face_tracker.detect_faces(video_batch)

        # Assign an identity to each of the detected faces
        found_ids = face_tracker.identify(video_batch, boxes_found)

        # Put a square around the detected faces and write their id
        new_video.extend(face_tracker.rectangle_detected_faces(video_batch, boxes_found, found_ids))
    
    face_tracker.write_video(np.array(new_video), "/output/tracked.mp4")

