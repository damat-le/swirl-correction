import random
import matplotlib.pyplot as plt 
from src.eval import inv_preprocess_img

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
            pred_mask, output = model(simg.unsqueeze(0).to(device))
            pred_mask = pred_mask.squeeze(0).cpu()
            output = output.squeeze(0).cpu()

        plt.figure(figsize=(20, 6))

        plt.subplot(1, 5, 1)
        plt.imshow(
            inv_preprocess_img(simg)
        )
        plt.title("Swirled Image")

        plt.subplot(1, 5, 2)
        plt.imshow(
            inv_preprocess_img(mask),
            cmap='gray'
        )
        plt.title("Mask")

        plt.subplot(1, 5, 3)
        plt.imshow(
            inv_preprocess_img(oimg)
        )
        plt.title("Original Image")

        plt.subplot(1, 5, 4)
        plt.imshow(
            inv_preprocess_img(output)
        )
        plt.title("Model Output")

        plt.subplot(1, 5, 5)
        plt.imshow(
            inv_preprocess_img(pred_mask),
            cmap='gray'
        )
        plt.title("Predicted Mask")

        plt.savefig(
            f"{log_dir}/sample_{i}.png",
            bbox_inches='tight'
        )
        plt.close()
                

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

            _, output = model(simg)

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


if __name__ == "__main__":

    import time
    import torch
    import numpy as np
    from src.dataset import load_flowers_dataset, SwirledDataset
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from tqdm import tqdm

    from argparse import ArgumentParser
    from src.utils.config import Config
    from src.utils.logger import MyLogger
    from src.models import MODEL_REGISTRY
    

    # -------------------------------------------
    # Load config
    # -------------------------------------------
    parser = ArgumentParser()
    parser.add_argument("-c", type=str, required=True)
    args = parser.parse_args()
    c = Config.from_yaml(args.c)


    # -------------------------------------------
    # Setup logger
    # -------------------------------------------
    logger = MyLogger(c.log_params["log_dir"])
    logger.log_config(c)


    # -------------------------------------------
    # Setup device
    # -------------------------------------------
    device = torch.device(c.trainer_params["device"])


    # -------------------------------------------
    # Setup datasets
    # -------------------------------------------
    data, labels = load_flowers_dataset(
        c.data_params["in_dir"]
    )

    val_idx = np.random.choice(
        len(data),
        c.val_params["val_size"]
    )

    val_data =[]
    val_labels = []
    for i in val_idx:
        val_data.append(data[i])
        val_labels.append(labels[i])
    
    data = [data[i] for i in range(len(data)) if i not in val_idx]
    labels = [labels[i] for i in range(len(labels)) if i not in val_idx]
    
    ds = SwirledDataset(data)

    dl = DataLoader(
        ds, 
        shuffle=True,
        batch_size=c.trainer_params["batch_size"], 
        num_workers=c.trainer_params["data_workers"],
    )

    val_ds = SwirledDataset(val_data)

    val_dl = DataLoader(
        val_ds, 
        shuffle=False,
        batch_size=c.val_params["batch_size"], 
        num_workers=c.val_params["data_workers"],
    )

    # -------------------------------------------
    # Initialize the model
    # -------------------------------------------
    architecture = MODEL_REGISTRY[c.model_params.pop("model_name")]
    kwargs_detector = c.detector_params
    kwargs_corrector = c.corrector_params
    model = architecture(kwargs_detector, kwargs_corrector)
    model = model.to(device)

    # -------------------------------------------
    # Initialize the optimizer
    # -------------------------------------------
    optimizer = Adam(
        model.parameters(),
        lr=c.opt_params["lr"]
    )

    # -------------------------------------------
    # Print the model summary
    # -------------------------------------------
    print('----------------------------------------')
    print(f'Model: {architecture.__name__.split(".")[-1]}')
    print(f"Num. of params: {sum(p.numel() for p in model.parameters())}")
    print(f'Device: {c.trainer_params["device"]}')
    print('----------------------------------------')

  
    # -------------------------------------------
    # Training cycle
    # -------------------------------------------

    pbar = tqdm(total=len(dl), desc="Training")
    it = 0
    for epoch in range(c.trainer_params["epochs"]):
        model.train()

        END_TIME = time.time()
        for batch in dl:

            data_time = time.time() - END_TIME
        
            swirled, mask, original = batch
            swirled = swirled.to(device)
            mask = mask.to(device)
            original = original.to(device)

            optimizer.zero_grad()
            output = model(swirled)

            loss, other_info = model.loss_function(output, original, mask)

            logger.log_scalars(
                scalars={
                    'LR': optimizer.param_groups[0]['lr'],
                    'Epoch': epoch,
                    **other_info
                },
                step=it
            )

            if it % 1 == 0:
                pbar.set_postfix({
                    "Epoch": epoch,
                    "Loss": loss.item(),
                    "DataTime": f"{data_time:.2f}s",
                })

            loss.backward()
            optimizer.step()

            pbar.update(1)
            it += 1

            END_TIME = time.time()
        # -------------------------------------------
        # Validation
        # -------------------------------------------

        model.eval()

        val_results = eval_swirl_mask(
            dataloader=val_dl, 
            model=model
        )

        logger.log_scalars(
            scalars=val_results,
            step=it
        )

        generate_samples(
            log_dir=logger.samples_dir,
            dataset=val_ds,
            model=model,
            num_samples=5
        )
        # -------------------------------------------
        # Save checkpoint
        # -------------------------------------------
        logger.log_checkpoint(
            model=model,
            step=it,
            optimizer=None,
            scheduler=None
        )

        pbar.reset()
    pbar.close()