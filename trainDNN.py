# %%
from src.Dataset import YouTubeFacesDatasetTorch
from src.ModelsDNN import FaceVerificationModel, fixed_image_standardization
from torch.utils.data import DataLoader
from torchsummary import summary
from pathlib import Path
from tqdm import tqdm
import torch

# %%
metadata_train_filepath = Path("/output/metadata_train.csv")
metadata_test_filepath = Path("/output/metadata_test.csv")
batch_size = 32
nb_epoch = 20

# %% Load the data
train_dataset = YouTubeFacesDatasetTorch(metadata_train_filepath, resize_size=160)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = YouTubeFacesDatasetTorch(metadata_test_filepath, resize_size=160)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# %% Load the model (pre-trained embedding model + from-scratch classifier)
model = FaceVerificationModel()
model.freeze_embedding_model()
model.eval() # To deactivate the Dropout layers in ResNet

# Print a summary of the model
summary(model, input_data=[torch.zeros(batch_size, 3, 256, 256), torch.zeros(batch_size, 3, 256, 256)])

# %% Define the loss and the optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# %% Training loop
for epoch in range(nb_epoch):
    # Train the model for an epoch
    for index_train, batch_train in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch_img_1, batch_img_2, batch_labels = batch_train

        # Clean stored gradients if there were any        
        optimizer.zero_grad()

        # Forward pass : Predict the logits (since the sigmoid is computed by the loss, these are not probabilities yet.)
        batch_logits = model(fixed_image_standardization(batch_img_1), fixed_image_standardization(batch_img_2))

        # Compute the loss function
        loss = criterion(input=batch_logits, target=batch_labels.float().unsqueeze(-1))
        print(loss)

        # Perform the backward pass and update the optimizer if needed (eg. changing the learning rate)
        loss.backward()
        optimizer.step()

        # Compute human-readable metrics, here the accuracy
        predicted_labels = torch.nn.functional.sigmoid(batch_logits) >= 0.5
        accuracy = torch.sum(predicted_labels == batch_labels.unsqueeze(-1)) / len(predicted_labels)
        print(accuracy)

    # Evaluate the performance of the model at the end of each epoch
    for index_valid, batch_valid in enumerate(test_dataloader):
        break
        pass

    break
