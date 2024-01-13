# %%
from facenet_pytorch import InceptionResnetV1
import torch

# Standardization expected by ResNet (old range = [0, 255]; new range = [-1, 1])
def fixed_image_standardization(image_tensor:torch.Tensor) -> torch.Tensor:
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

# %% Model definition
class ClassifierNeuralNetwork(torch.nn.Module):
    def __init__(self, dim_embedding:int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2*dim_embedding, 1)

    def forward(self, embeddings_img_1:torch.Tensor, embeddings_img_2:torch.Tensor) -> torch.Tensor:
        mixed_embeddings = torch.concat([embeddings_img_1, embeddings_img_2], dim=-1)
        return self.linear(mixed_embeddings)
    
class FaceVerificationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_model = InceptionResnetV1(pretrained='vggface2', classify=False)
        self.classifier = ClassifierNeuralNetwork(dim_embedding=512)

    def forward(self, batch_img_1: torch.Tensor, batch_img_2:torch.Tensor) -> torch.Tensor:
        # Combines the two batches of images to run them as a single batch by the ResNet model
        batch_img = torch.concat([batch_img_1, batch_img_2], dim=0)
        embeddings = self.embedding_model(batch_img)

        # Structure the stack of embeddings to match the original triplet structure
        embeddings_img_1 = embeddings[:len(batch_img_1)]
        embeddings_img_2 = embeddings[len(batch_img_1):]

        # Predict the label for each triplet
        x = self.classifier(embeddings_img_1, embeddings_img_2)

        return x
    
    def freeze_embedding_model(self):
        for param in list(self.embedding_model.parameters()):
            param.requires_grad = False

# %%
if __name__ == "__main__":
    # %%
    from pathlib import Path
    from Dataset import YouTubeFacesDatasetTorch

    input_folderpath = Path("/output/metadata.csv")
    dataset = YouTubeFacesDatasetTorch(input_folderpath)
    img_1, img_2, label = dataset[-10]
    print(img_1.unsqueeze(0).shape)
    
    model = FaceVerificationModel()

    # %%
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=2)
    for i, batch in enumerate(dataloader):
        batch_img_1, batch_img_2, batch_labels = batch
        blob = model(fixed_image_standardization(batch_img_1), fixed_image_standardization(batch_img_2))
        print(blob)
        break

    # %%
    from torchsummary import summary
    
    model = FaceVerificationModel()
    summary(model, input_data=[fixed_image_standardization(batch_img_1), fixed_image_standardization(batch_img_2)])

    # %%
    model = FaceVerificationModel()
    model.freeze_embedding_model()
    summary(model, input_data=[fixed_image_standardization(batch_img_1), fixed_image_standardization(batch_img_2)])
