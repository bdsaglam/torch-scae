import gc

import numpy as np
import torch
from torchvision import datasets, transforms

from torch_scae import factory
from torch_scae.configs import mnist_config


def train(scae, optimizer, data_loader, epoch, device=torch.device("cpu")):
    scae.to(device)
    scae.train()

    batch_size = int(data_loader.batch_size)
    n_batch = int(np.ceil(len(data_loader.dataset) / batch_size))
    for i, (image, label) in enumerate(data_loader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        res = scae(image=image, label=label)
        loss = scae.loss(res)
        loss.backward()
        optimizer.step()

        loss_value = loss.detach().cpu().item()
        accuracy = res.best_cls_acc.detach().cpu().item()

        if i % 100 == 0:
            print(
                f"Epoch: [{epoch}], Batch: [{i + 1}/{n_batch}] | Train | "
                f"Loss: {loss_value:.1f} "
                f"Accuracy: {accuracy:.3f} "
            )

        del res, loss
        torch.cuda.empty_cache()
        gc.collect()
        break


def evaluate(scae, data_loader, epoch, device=torch.device("cpu")):
    scae.to(device)
    scae.eval()

    batch_size = int(data_loader.batch_size)
    n_batch = int(np.ceil(len(data_loader.dataset) / batch_size))
    losses = []
    accuracies = []
    for i, (image, label) in enumerate(data_loader):
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            res = scae(image=image, label=label)
            loss = scae.loss(res)

        loss_value = loss.detach().cpu().item()
        losses.append(loss_value)

        accuracy = res.best_cls_acc.detach().cpu().item()
        accuracies.append(accuracy)

        del res, loss
        torch.cuda.empty_cache()
        gc.collect()
        break

    print(
        f"Epoch: [{epoch}] Batch: [{n_batch}/{n_batch}] | Valid | "
        f"Loss(μ:{np.mean(losses):.1f}, σ: {np.std(losses):.1f} "
        f"Accuracy(μ:{np.mean(accuracies):.3f}, σ: {np.std(accuracies):.3f} "
    )


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('/Users/bdsaglam/torch-datasets',
                                   train=True,
                                   download=True,
                                   transform=dataset_transform)
    val_dataset = datasets.MNIST('/Users/bdsaglam/torch-datasets',
                                 train=False,
                                 download=True,
                                 transform=dataset_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    scae = factory.make_scae(mnist_config)
    scae.to(DEVICE)

    optimizer = torch.optim.Adam(scae.parameters(), lr=LEARNING_RATE)

    for e in range(1, 1 + EPOCHS):
        train(scae, optimizer, train_loader, e, device=DEVICE)
        evaluate(scae, val_loader, e, device=DEVICE)
