if __name__ == "__main__":

    import torch
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
    
    ds = SwirledDataset(data)

    dl = DataLoader(
        ds, 
        shuffle=True,
        batch_size=c.trainer_params["batch_size"], 
        num_workers=c.trainer_params["data_workers"],
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
        
        for batch in dl:
            swirled, mask, original = batch
            swirled = swirled.to(device)
            mask = mask.to(device)
            original = original.to(device)

            optimizer.zero_grad()
            output = model(swirled)
            loss = model.loss_function(output, original)
            
            logger.log_scalars(
                scalars={
                    'TopKLoss/train': 0,
                    'Loss/train': loss.item(),
                    'LR': optimizer.param_groups[0]['lr'],
                    'Epoch': epoch
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
            dataloader=dl, 
            model=model
        )

        logger.log_scalars(
            scalars=val_results,
            step=it
        )

        generate_samples(
            log_dir=logger.samples_dir,
            dataset=ds,
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