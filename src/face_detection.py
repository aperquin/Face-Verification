# %%
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import ToPILImage
from PIL import Image
from torchvision.io import read_image
from Dataset import YouTubeFacesDatasetTorch, show_tensor_image
import torch

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

# %%
input_img = "/dataset/YouTubeFaces/frame_images_DB/Alicia_Witt/5/5.2247.jpg"
output_img = "/output/Alicia_Witt_5_2247"
image_size = 160
margin = 32

input_folderpath = "/output/metadata.csv"

# %%
mtcnn = MTCNN(image_size=image_size, margin=margin)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# %%
# img = Image.open(input_img)
img = read_image(input_img)
img_cropped = mtcnn(img, save_path="/output/Alicia_Witt_5_2247_cropped.jpg")

# %%
from facenet_pytorch import MTCNN, InceptionResnetV1

image_size = 160
margin = 32
mtcnn = MTCNN(image_size=image_size, margin=margin, post_process=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# %%
dataset = YouTubeFacesDatasetTorch(input_folderpath)
img_1, img_2, label = dataset[-10]
show_tensor_image(img_1)

# %% Face detection
to_pil_image = ToPILImage()
img_1_cropped = mtcnn(ToPILImage()(img_1), save_path="/output/cropped_img_1.jpg")
img_2_cropped = mtcnn(ToPILImage()(img_2), save_path="/output/cropped_img_2.jpg")
show_tensor_image(img_1_cropped)

# %% Embedding extraction
aligned = torch.stack([img_1_cropped, img_2_cropped])
embeddings = resnet(aligned).detach().cpu()
embeddings.shape

# %%
(embeddings[0]-embeddings[1]).norm(p=2)

# %% Embedding extraction
aligned = torch.stack([fixed_image_standardization(img_1), fixed_image_standardization(img_2)])
embeddings = resnet(aligned).detach().cpu()
embeddings.shape

# %%
(embeddings[0]-embeddings[1]).norm(p=2)

# %%
show_tensor_image(fixed_image_standardization(img_1))
