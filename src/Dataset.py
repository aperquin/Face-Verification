# %%
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import InterpolationMode, functional as F
from pathlib import Path
from abc import ABC, abstractmethod
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torch

# %%
class YouTubeFacesDatasetAbstract(Dataset, ABC):
    def __init__(self, metadata_filepath:Path, crop:bool=True, resize:bool=True, resize_size:int=256) -> None:
        self.metadata = pd.read_csv(metadata_filepath)
        self.crop = crop
        self.resize = resize
        self.resize_size = resize_size

    def __getitem__(self, index:int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        filepath_1, x_1, y_1, width_1, height_1 = list(self.metadata.iloc[index][["filepath_1", 'x_1', 'y_1', 'width_1', 'height_1']])
        filepath_2, x_2, y_2, width_2, height_2 = list(self.metadata.iloc[index][["filepath_2", 'x_2', 'y_2', 'width_2', 'height_2']])
        label = self.metadata.iloc[index]["label"]

        img_1 = self.open_img(filepath_1)
        img_2 = self.open_img(filepath_2)

        if self.crop:
            img_1 = self.crop_img(img_1, x_1, y_1, width_1, height_1)
            img_2 = self.crop_img(img_2, x_2, y_2, width_2, height_2)
            
        if self.resize:
            img_1 = self.resize_img(img_1)
            img_2 = self.resize_img(img_2)

        return img_1, img_2, label
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    @abstractmethod
    def open_img():
        raise NotImplementedError()

    @abstractmethod
    def crop_img():
        raise NotImplementedError()
    
    @abstractmethod
    def resize_img():
        raise NotImplementedError()


class YouTubeFacesDatasetTorch(YouTubeFacesDatasetAbstract):
    def open_img(self, img_filepath:Path) -> torch.Tensor:
        return read_image(img_filepath, ImageReadMode.RGB)

    def crop_img(self, img:torch.Tensor, x_center:int, y_center:int, width:int, height:int) -> torch.Tensor:
        left = int(x_center - width / 2)
        upper = int(y_center - height / 2)
        img_cropped = F.crop(img, upper, left, height, width)
        return img_cropped
    
    def resize_img(self, img:torch.Tensor) -> torch.Tensor:
        return F.resize(img, size=(self.resize_size, self.resize_size), interpolation=InterpolationMode.BILINEAR, antialias=None)
    

def show_tensor_image(tensor_img):
    plt.imshow(F.to_pil_image(tensor_img))


class YouTubeFacesDatasetPIL(YouTubeFacesDatasetAbstract):
    def open_img(self, img_filepath:Path) -> Image:
        return Image.open(img_filepath)

    def crop_img(self, img:Image, x_center:int, y_center:int, width:int, height:int) -> Image:
        left = int(x_center - width / 2)
        upper = int(y_center - height / 2)
        right = int(x_center + width / 2)
        lower = int(y_center + height / 2)
        img_cropped = img.crop([left, upper, right, lower])
        return img_cropped
    
    def resize_img(self, img:Image) -> Image:
        return img.resize(size=(self.resize_size, self.resize_size), resample=Image.Resampling.BILINEAR)

# %% Some tests :
if __name__ == "__main__":
    # %% Open the Dataset
    input_folderpath = Path("/output/metadata.csv")

    dataset = YouTubeFacesDatasetTorch(input_folderpath)
    img_1, img_2, label = dataset[-10]
    print(img_1.shape, img_2.shape)

    # %% Print the first image
    show_tensor_image(img_1)

    # %% Print the second image
    show_tensor_image(img_2)

    # %% 
    dataset = YouTubeFacesDatasetPIL(input_folderpath)
    img_1, img_2, label = dataset[-10]

    # %%
    img_1.show()

    # %%
    img_2.show()
