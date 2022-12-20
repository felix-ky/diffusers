from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw

DATA_PATH = "../../data/LIP/"
LIP_PATH = "../../data/LIP/LIP/"
CIHP_PATH = "../../data/LIP/CIHP/"

class CIHP_Dataset(Dataset):
    def __init__(self, 
                 usage: str,
                 shuffle: bool,
                 size = 512,
                 instance = True,
                 category = True,
                 human = True,
                 mask = True) -> None:
        self.usage = usage
        self.need_instance = instance
        self.need_category = category
        self.need_human = human
        self.need_mask = mask
        
        if self.usage == "train":
            id_file = os.path.join(CIHP_PATH, "instance-level_human_parsing/Training/train_id.txt")
            self.images_folder = os.path.join(CIHP_PATH, "instance-level_human_parsing/Training/Images/")
            self.categories_folder = os.path.join(CIHP_PATH, "instance-level_human_parsing/Training/Categories/")
            self.human_folder = os.path.join(CIHP_PATH, "instance-level_human_parsing/Training/Human/")
            self.instances_folder = os.path.join(CIHP_PATH, "instance-level_human_parsing/Training/Instances/")
        else :
        # self.usage == "valid": 
            id_file = os.path.join(CIHP_PATH, "instance-level_human_parsing/Validation/train_id.txt")
            self.images_folder = os.path.join(CIHP_PATH, "instance-level_human_parsing/Validation/Images/")
            self.categories_folder = os.path.join(CIHP_PATH, "instance-level_human_parsing/Validation/Categories/")
            self.human_folder = os.path.join(CIHP_PATH, "instance-level_human_parsing/Validation/Human/")
            self.instances_folder = os.path.join(CIHP_PATH, "instance-level_human_parsing/Validation/Instances/")
        
        with open(id_file, 'r') as file:
            self.names = file.read().splitlines()
        if (shuffle) :
            np.random.shuffle(self.names)
        
        self.image_transforms_resize_and_crop = transforms.Compose(
            transforms=[
                transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size=size),
            ]
        )
        self.image_transforms = transforms.Compose(
            transforms=[
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_transforms_to_tensor = transforms.ToTensor()
    
    def __len__(self) -> int :
        return len(self.names)
    
    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        name = self.names[index]
        record = {}

        image = Image.open(os.path.join(self.images_folder, name + '.jpg'))
        record["image"] = self.image_transforms_to_tensor(self.image_transforms_resize_and_crop(img=image))
        if self.need_instance :
            instance = Image.open(os.path.join(self.instances_folder, name + '.png'))
            record["instance"] = self.image_transforms_to_tensor(self.image_transforms_resize_and_crop(img=instance))
        if self.need_category:
            category = Image.open(os.path.join(self.categories_folder, name + '.png'))
            record["category"] = self.image_transforms_to_tensor(self.image_transforms_resize_and_crop(img=category))
        if self.need_human:
            human = Image.open(os.path.join(self.human_folder, name + '.png'))
            record["human"] = self.image_transforms_to_tensor(self.image_transforms_resize_and_crop(img=human))
        if self.need_mask:
            category = Image.open(os.path.join(self.categories_folder, name + '.png'))
            mask = self.image_transforms_to_tensor(self.image_transforms_resize_and_crop(img=category.convert('L')))
            mask[mask > 0] = 1
            record["mask"] = mask
        
        return record
    
if __name__ == "__main__":
    test = CIHP_Dataset("train", False)
    save_image(test[0]["category"], "./origin.jpg")
    save_image(test[0]["mask"], "./test.jpg")
    # print(test[0]["image"].shape)
    # print(test[0]["instance"].shape)
    # print(test[0]["category"].shape)
    # print(test[0]["human"].shape)