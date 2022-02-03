def train_autorec():
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()
    trainer = pl.Trainer()
    trainer.fit(autoencoder, DataLoader(train), DataLoader(val))