# %%
from src.Dataset import YouTubeFacesDatasetTorch
from src.ModelsDNN import fixed_image_standardization
from pathlib import Path
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
from tqdm import tqdm
import torch
import numpy as np
import pickle
import argparse


def compute_embeddings(dataset:YouTubeFacesDatasetTorch, resnet:InceptionResnetV1) -> tuple[np.ndarray, np.ndarray]:
    """Compute face embeddings for a given dataset using FaceNet. Also extracts the correspounding ground truth labels.

    Args:
        dataset (YouTubeFacesDatasetTorch): The dataset to extract embeddings from.
        resnet (InceptionResnetV1): The FaceNet neural network to use for embedding extraction.

    Returns:
        np.ndarray: Array of embeddings.
        np.ndarray: Array of ground truth labels.
    """
    labels = []
    embeddings_list = []

    for i in tqdm(range(len(dataset))):
        img_1, img_2, label = dataset[i]
        
        # ResNet expects normalized images
        stacked_tensors = torch.stack([fixed_image_standardization(img) for img in [img_1, img_2]])
        embeddings = resnet(stacked_tensors).detach().numpy()

        embeddings_list.append(embeddings)
        labels.append(label)
    
    return np.array(embeddings_list), np.array(labels).reshape(-1, 1)


def print_performance(model:LinearSVC, cosine_dists:np.ndarray, gt_labels:np.ndarray) -> None:
    """Print the classification performance of a Scikit-Learn model.

    Args:
        model (LinearSVC): Model to assess the performances of.
        cosine_dists (np.ndarray): The cosine distances to use as input features of the model.
        gt_labels (np.ndarray): The ground truth labels to compare the predictions to.
    """
    pred_labels = model.predict(cosine_dists)

    print("Confusion matrix :")
    print(confusion_matrix(gt_labels, pred_labels))
    
    accuracy = accuracy_score(gt_labels, pred_labels)
    print(f"Accuracy = {accuracy}")
    
    precision, recall, _, _ = precision_recall_fscore_support(gt_labels, pred_labels, average='binary')
    print(f"Precision = {precision}, Recall = {recall}")


if __name__ == "__main__":
    # %% Parameters of the script
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--metadata_train_filepath', type=str, help='Path to the metadata listing the training data.', default="/output/train_metadata.csv")
    parser.add_argument('--metadata_test_filepath', type=str, help='Path to the metadata listing the test data.', default="/output/test_metadata.csv")
    parser.add_argument('--save_model_filepath', type=str, help='Path to save the resulting classifier.', default="/output/svm_classifier.pkl")
    parser.add_argument('--image_size_resnet', type=int, help='The size images will be resized to before being fed to ResNet.', default=160)

    args = parser.parse_args()
    metadata_train_filepath = Path(args.metadata_train_filepath)
    metadata_test_filepath = Path(args.metadata_test_filepath)
    image_size_resnet = args.image_size_resnet
    save_model_filepath = Path(args.save_model_filepath)

    # %% Load the data and the FaceNet model
    train_dataset = YouTubeFacesDatasetTorch(metadata_train_filepath, resize_size=image_size_resnet)
    test_dataset = YouTubeFacesDatasetTorch(metadata_test_filepath, resize_size=image_size_resnet)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # %% Compute the face embeddings for the train and test set
    print("Extracting embeddings for the train set")
    embeddings_train, gt_labels_train = compute_embeddings(train_dataset, resnet)
    print("Extracting embeddings for the test set")
    embeddings_test, gt_labels_test = compute_embeddings(test_dataset, resnet)

    # %% For each triplet, compute the cosine distance between the embeddings of the two faces
    cosine_dists_train = np.array([cosine_distance(embeddings_train[i, 0], embeddings_train[i, 1]) for i in range(len(embeddings_train))]).reshape(-1, 1)
    cosine_dists_test = np.array([cosine_distance(embeddings_test[i, 0], embeddings_test[i, 1]) for i in range(len(embeddings_test))]).reshape(-1, 1)

    # %% Train a classifier to predict the label 'Same'/'Different' from the cosine distances on the train set
    # NB. Since the model is a Linear Support Vector Machine (Linear SVM), and the data in the input is 1D,
    # it's equivalent to finding out the best threshold value where :
    # - 'cosine distance < threshold' --> 'Same'
    # - 'cosine distance > threshold' --> 'Different'
    classifier = LinearSVC().fit(cosine_dists_train, gt_labels_train)

    # %% Print the performance on the train set and the test set
    print("Performances on the train set:")
    print_performance(classifier, cosine_dists_train, gt_labels_train)
    print("===============")
    print("Performances on the test set:")
    print_performance(classifier, cosine_dists_test, gt_labels_test)

    # %% Save the classifier
    with save_model_filepath.open('wb') as opened_file:
        pickle.dump(classifier, opened_file)