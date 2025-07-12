import os
import torch
import numpy as np
from PIL import Image as im
from skimage.draw import disk
from skimage.transform import swirl, resize
from torch.utils.data import Dataset

labels2int = {
    'daisy': 0,
    'dandelion': 1,
    'roses': 2,
    'sunflowers': 3,
    'tulips': 4
}

def load_flowers_dataset(data_path):
    """
    Loads the flowers dataset from the specified path.

    Args:
        data_path (str): Path to the dataset file.

    Returns:
        tuple: (data, labels) where data is a list of data points and labels is a list of corresponding labels.
    """
    # Placeholder for actual loading logic
    data = []  # Load your data here
    labels = []  # Load your labels here
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            #read image file
            img_path = os.path.join(root, filename)
            img = im.open(img_path)
            img = np.array(img)
            data.append(img)
            
            # Assuming the label is the folder name
            label = os.path.basename(root)  
            labels.append(label)

    return data, labels


def preprocess_img(image: np.ndarray, IMG_SIZE=(128, 128, 3), normalize=True):
    """
    Preprocesses the image for model input.

    Args:
        image (numpy.ndarray): The input image as a numpy array.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    # resize the image to the specified size
    # @NOTE: this normalizes the image to [0, 1] range automatically
    img = resize(
        image,
        IMG_SIZE, 
    )
    # Change from HWC to CHW format
    img = img.transpose((2, 0, 1))  
    return img

def inv_preprocess_img(img_tensor: torch.Tensor, denormalize=False):
    img_np = img_tensor.detach().numpy()  # Convert tensor to numpy array
    img_np = img_np.transpose((1, 2, 0))  # Change from CHW to HWC format  
    if denormalize:
        img_np = img_np * 255.0  # Rescale to [0, 255]
    return img_np

def generate_swirl_defect(img_np, DEFECT_RADIUS=32, STRENGTH=4):
    """
    Generates a swirl defect in the image.

    Args:
        img_np (numpy.ndarray): The input image as a numpy array with shape (C, H, W).
        DEFECT_RADIUS (int): Radius of the swirl defect.
        STRENGTH (int): Strength of the swirl effect.
    Returns:
        tuple: (swirled_image, mask) where swirled_image is the image with the swirl defect and mask is the mask of the defect.
    """
    img_np = img_np.transpose((1, 2, 0))
    h, w = img_np.shape[:2]
    radius = int(DEFECT_RADIUS * np.random.uniform(0.5, 1.))
    strength = STRENGTH * np.random.uniform(0.5, 1.)
    center = np.random.randint(radius, h - radius, size=2)
    swirled = swirl(
        img_np, 
        center=center[::-1], 
        strength=strength, 
        radius=radius
    )
    mask = np.zeros((h, w, 1), dtype=np.uint8)
    rr, cc = disk(center, radius, shape=(h, w))
    mask[rr, cc] = 1
    return swirled.transpose((2,0,1)), mask.transpose((2,0,1))


class FlowersDataset(Dataset):

    def __init__(self, data: list[im.Image]):
        """
        Initializes the FlowersDataset.

        Args:
            data (list): List of PIL Images.
            labels (list): List of corresponding labels.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(preprocess_img(self.data[idx])).float()
    

class SwirledDataset(Dataset):

    def __init__(self, 
            data: list[im.Image], 
            DEFECT_RADIUS=32, 
            STRENGTH=4
        ):
        """
        Initializes the SwirledDataset.

        Args:
            data (list): List of PIL Images.
            DEFECT_RADIUS (int): Radius of the swirl defect.
            STRENGTH (int): Strength of the swirl effect.
        """
        self.data = data
        self.DEFECT_RADIUS = DEFECT_RADIUS
        self.STRENGTH = STRENGTH

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_np = preprocess_img(self.data[idx])
        swirled_img, mask = generate_swirl_defect(
            img_np, 
            self.DEFECT_RADIUS, 
            self.STRENGTH
        )
        return torch.tensor(swirled_img).float(), torch.tensor(mask).float(), torch.tensor(img_np).float()
