if __name__ == "__main__":

    import torch
    from src.dataset import load_flowers_dataset, FlowersDataset
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from models.ae.ae import Autoencoder
    from tqdm import tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    data, labels = load_flowers_dataset('./data/flower_photos')
    
    # Create the dataset object
    ds = FlowersDataset(data)

    # Create a DataLoader
    dl = DataLoader(
        ds, 
        batch_size=8, 
        shuffle=True,
    )

    # Initialize the Autoencoder model
    model = Autoencoder(in_channels=3, hidden_dim=512)
    model = model.to(device)
    model.train()

    # Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=0.0005)

    # Print the model summary
    print(f"Num. of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Iterate through the DataLoader and print the shape of the first batch
    for epoch in range(10):
        pbar = tqdm(dl, desc="Training", unit="batch")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            output, _ = model(batch)
            loss = torch.nn.functional.mse_loss(output, batch)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"epoch": epoch, "loss": loss.item()})

        # save the model
        torch.save(model.state_dict(), './logs/autoencoder.pth')