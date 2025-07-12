if __name__ == "__main__":

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
    from src.eval import eval_swirl_mask, generate_samples
    

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
    model = architecture(**c.model_params)
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

        if epoch < 5:
            top_k = 64
        elif epoch < 50:
            top_k = 16
        elif epoch < 100:
            top_k = 8
        else:
            top_k = 4

        for batch in dl:
            swirled, mask, original = batch
            swirled = swirled.to(device)
            mask = mask.to(device)
            original = original.to(device)

            optimizer.zero_grad()
            output = model(swirled)
            loss, other_info = model.loss_function_patch(output, original, top_k=top_k)
            
            logger.log_scalars(
                scalars={
                    'TopKLoss/train': loss.item(),
                    'Loss/train': other_info['loss_full'].item(),
                    'LR': optimizer.param_groups[0]['lr'],
                    'Epoch': epoch,
                    'TopK': top_k
                },
                step=it
            )

            if it % 10 == 0:
                pbar.set_postfix({
                    "Epoch": epoch,
                    "Loss": loss.item(),
                })

            loss.backward()
            optimizer.step()

            pbar.update(1)
            it += 1

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