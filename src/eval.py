import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.dataset import inv_preprocess_img


def eval_swirl_mask(dataloader, model):

    # set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = next(model.parameters()).device

    pbar_val = tqdm(
        total=len(dataloader), 
        desc=f"Validation (mse_ratio)", 
        leave=False
    )

    loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            simg, mask, oimg = batch
            simg = simg.to(device)
            mask = mask.to(device)
            oimg = oimg.to(device)

            output = model(simg)

            #only compute loss for the masked area
            output = output * mask
            oimg = oimg * mask
            simg = simg * mask
            # Compute the loss
            baseline_loss = torch.nn.functional.mse_loss(simg, oimg)
            model_loss = torch.nn.functional.mse_loss(output, oimg)
            loss += model_loss / baseline_loss

            pbar_val.update(1)

    # Average the loss over the number of batches
    loss /= len(dataloader)

    pbar_val.set_postfix({
        "mse_ratio": loss.item(),
    })
    pbar_val.close()

    return {'mse_ratio:' : loss.item()}

def generate_samples(log_dir, dataset, model, num_samples=5):
    """
    Generate samples from the dataset using the model.
    """
    device = next(model.parameters()).device
    model.eval()
    
    for i in range(num_samples):
        sample = dataset[i]
        simg, mask, oimg = sample

        with torch.no_grad():
            output = model(simg.unsqueeze(0).to(device))
            output = output.squeeze(0).cpu()

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 4, 1)
        plt.imshow(
            inv_preprocess_img(simg)
        )
        plt.title("Swirled Image")

        plt.subplot(1, 4, 2)
        plt.imshow(
            inv_preprocess_img(mask),
            cmap='gray'
        )
        plt.title("Mask")

        plt.subplot(1, 4, 3)
        plt.imshow(
            inv_preprocess_img(oimg)
        )
        plt.title("Original Image")

        plt.subplot(1, 4, 4)
        plt.imshow(
            inv_preprocess_img(output)
        )
        plt.title("Model Output")

        plt.savefig(
            f"{log_dir}/sample_{i}.png",
            bbox_inches='tight'
        )
        plt.close()
                