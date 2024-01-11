# %%
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import InterpolationMode, functional as F
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch

# %%
class YouTubeFacesDataset(Dataset):
    def __init__(self, metadata_filepath:Path, crop:bool=True, resize:bool=True, resize_size:int=256) -> None:
        self.metadata = pd.read_csv(metadata_filepath)
        self.crop = crop
        self.resize = resize
        self.resize_size = resize_size

    def __getitem__(self, index:int) -> None:
        filepath_1, x_1, y_1, width_1, height_1 = list(self.metadata.iloc[index][["filepath_1", 'x_1', 'y_1', 'width_1', 'height_1']])
        filepath_2, x_2, y_2, width_2, height_2 = list(self.metadata.iloc[index][["filepath_2", 'x_2', 'y_2', 'width_2', 'height_2']])
        label = self.metadata.iloc[index]["label"]

        if self.crop:
            img_1 = self.open_and_crop(filepath_1, x_1, y_1, width_1, height_1)
            img_2 = self.open_and_crop(filepath_2, x_2, y_2, width_2, height_2)
        else:
            img_1 = read_image(filepath_1, ImageReadMode.RGB)
            img_2 = read_image(filepath_2, ImageReadMode.RGB)

        if self.resize:
            img_1 = self.resize_img(img_1)
            img_2 = self.resize_img(img_2)

        return img_1, img_2, label
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    
    def open_and_crop(self, img_filepath:Path, x_center:int, y_center:int, width:int, height:int) -> torch.Tensor:
        img = read_image(img_filepath, ImageReadMode.RGB)
        left = int(x_center - width / 2)
        upper = int(y_center - height / 2)
        img_cropped = F.crop(img, upper, left, height, width)
        return img_cropped
    
    def resize_img(self, img:torch.Tensor) -> torch.Tensor:
        return F.resize(img, size=(self.resize_size, self.resize_size), interpolation=InterpolationMode.BILINEAR, antialias=None)
    
def show_tensor_image(tensor_img):
    plt.imshow(F.to_pil_image(tensor_img))

# %% Some tests :
if __name__ == "__main__":
    # %% Open the Dataset
    input_folderpath = Path("/output/metadata.csv")

    dataset = YouTubeFacesDataset(input_folderpath)
    img_1, img_2, label = dataset[-10]
    print(img_1.shape, img_2.shape)

    # %% Print the first image
    show_tensor_image(img_1)

    # %% Print the second image
    show_tensor_image(img_2)

# %%
