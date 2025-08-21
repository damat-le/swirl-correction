import os
import torch
import torch.nn.functional as F
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
            img = torch.tensor(img, dtype=torch.uint8)  # Convert to tensor
            data.append(img)
            
            # Assuming the label is the folder name
            label = os.path.basename(root)  
            labels.append(label)

    return data, labels


def preprocess_img(image: torch.Tensor, IMG_SIZE=(128, 128, 3), normalize=True):
    """
    Preprocesses the image for model input using PyTorch.

    Args:
        image (torch.Tensor): Input image tensor in HWC format, dtype uint8 or float, range [0,255] or [0,1].
        IMG_SIZE (tuple): Target size as (H, W, C).
        normalize (bool): If True, normalize to [0, 1].

    Returns:
        torch.Tensor: Preprocessed image in CHW format, float32, range [0,1] if normalize=True.
    """
    target_h, target_w, target_c = IMG_SIZE
    assert image.ndim == 3, "image must be HWC"
    assert image.shape[2] == target_c, f"Expected {target_c} channels, got {image.shape[2]}"

    # Convert to float and normalize if needed
    if normalize:
        image = image.float() / 255.0
    else:
        image = image.float()

    # Convert HWC → CHW for interpolate
    img_chw = image.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # Resize
    img_resized = F.interpolate(img_chw, size=(target_h, target_w), mode='bilinear', align_corners=False)

    return img_resized.squeeze(0)  # (C, H, W)


def inv_preprocess_img(img_tensor: torch.Tensor, denormalize=False):
    img_np = img_tensor.detach().numpy()  # Convert tensor to numpy array
    img_np = img_np.transpose((1, 2, 0))  # Change from CHW to HWC format  
    if denormalize:
        img_np = img_np * 255.0  # Rescale to [0, 255]
    return img_np


def generate_swirl_defect(img_tensor, DEFECT_RADIUS=32, STRENGTH=4, ROTATION=0.0):
    """
    Generates a swirl defect in the image using the same mapping as skimage's swirl.
    """
    assert img_tensor.ndim == 3, "img_tensor must be (C, H, W)"
    C, H, W = img_tensor.shape
    device = img_tensor.device
    dtype = img_tensor.dtype

    # Random radius, strength, and center
    radius = int(DEFECT_RADIUS * torch.empty(1, dtype=dtype, device=device).uniform_(0.5, 1.).item())
    strength = STRENGTH * torch.empty(1, dtype=dtype, device=device).uniform_(0.5, 1.).item()
    cy = torch.randint(radius, H - radius, (1,), device=device).item()
    cx = torch.randint(radius, W - radius, (1,), device=device).item()

    # Coordinate grid in same dtype as image
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )

    # Polar coordinates relative to center
    rho = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Decay adjustment for radius
    adj_radius = radius / 5 * torch.log(torch.tensor(2.0, dtype=dtype, device=device))

    theta = (torch.tensor(ROTATION, dtype=dtype, device=device)
             + strength * torch.exp(-rho / adj_radius)
             + torch.atan2(yy - cy, xx - cx))

    # Back to Cartesian
    xs = cx + rho * torch.cos(theta)
    ys = cy + rho * torch.sin(theta)

    # Normalize to [-1, 1] for grid_sample
    xs_norm = (xs / (W - 1)) * 2 - 1
    ys_norm = (ys / (H - 1)) * 2 - 1
    grid = torch.stack((xs_norm, ys_norm), dim=-1).unsqueeze(0)

    # Apply grid_sample
    img_batch = img_tensor.unsqueeze(0)
    swirled = F.grid_sample(img_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # Binary mask for defect
    mask = (rho <= radius).unsqueeze(0)

    return swirled.squeeze(0), mask

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
        img = preprocess_img(self.data[idx])
        swirled_img, mask = generate_swirl_defect(
            img, 
            self.DEFECT_RADIUS, 
            self.STRENGTH
        )
        return swirled_img.float(), mask.float(), img.float()
