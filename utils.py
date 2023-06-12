import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

train_losses = []
test_losses = []
train_acc = []
test_acc = []
test_incorrect_pred = {
    "images": list(),
    "ground_truths": list(),
    "predicted_vals": list(),
}


def show_examples(train_loader, figsize=None):
    batch_data, batch_label = next(iter(train_loader))

    fig = plt.figure(figsize=figsize)

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap="gray")
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(
            desc=f"Train: Batch Loss={loss.item():0.4f} Batch_id={batch_idx} Average Loss={train_loss / (batch_idx + 1):0.4f} Accuracy={100 * correct / processed:0.2f}"
        )

    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))


def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)

            test_loss += criterion(
                pred, target, reduction="sum"
            ).item()  # sum up batch loss

            correct += GetCorrectPredCount(pred, target)

            ind, pred, truth = GetInCorrectPreds(pred, target)
            test_incorrect_pred["images"] += data[ind]
            test_incorrect_pred["ground_truths"] += truth
            test_incorrect_pred["predicted_vals"] += pred

    test_loss /= len(test_loader.dataset)
    test_acc.append(100.0 * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    return test_loss


def plot_stats():
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def GetInCorrectPreds(pPrediction, pLabels):
    pPrediction = pPrediction.argmax(dim=1)
    indices = pPrediction.ne(pLabels).nonzero().reshape(-1).tolist()
    return indices, pPrediction[indices].tolist(), pLabels[indices].tolist()


def show_incorrect(figsize=None):
    fig = plt.figure(figsize=figsize)

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.tight_layout()
        plt.imshow(test_incorrect_pred["images"][i].cpu().squeeze(0), cmap="gray")
        plt.title(
            repr(test_incorrect_pred["predicted_vals"][i])
            + " vs "
            + repr(test_incorrect_pred["ground_truths"][i])
        )
        plt.xticks([])
        plt.yticks([])
