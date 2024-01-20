import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim), 
                norm(hidden_dim), 
                nn.ReLU(), 
                nn.Dropout(drop_prob), 
                nn.Linear(hidden_dim, dim), 
                norm(dim)
            )
        ), 
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules = [
        nn.Linear(dim, hidden_dim), 
        nn.ReLU()
    ]
    modules += [ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)]
    modules += [nn.Linear(hidden_dim, num_classes)]
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.train()
    if opt is None:
        model.eval()
    total_loss = 0
    loss_fn = ndl.nn.SoftmaxLoss()
    num_correct = 0
    
    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.reshape((-1, 28*28))
        # print(inputs.shape)
        if opt is not None:
            opt.reset_grad()
        logits = model(inputs)
        num_correct += (logits.numpy().argmax(-1) == labels.numpy()).sum()
        loss = loss_fn(logits, labels)
        if opt is not None:
            loss.backward()
            opt.step()
        total_loss = total_loss + loss.detach()
    return 
    return num_correct / dataloader.num_batch / dataloader.batch_size, total_loss.detach().numpy()[0] / dataloader.num_batch
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model = MLPResNet(28*28, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_mnist_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_mnist_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    
    for _ in range(epochs):
        train_data_loader = ndl.data.DataLoader(train_mnist_dataset, batch_size, shuffle=True)
        acc, loss = epoch(train_data_loader, model, opt)
        print(f"Epoch: {_}, training loss {loss}")
        if _ == epochs - 1:
            test_data_loader = ndl.data.DataLoader(test_mnist_dataset, batch_size, shuffle=False)
            eval_acc, eval_loss = epoch(test_data_loader, model, None)
            print(f"Epoch: {_}, evaluation loss {eval_loss}")
            return acc, loss, eval_acc, eval_loss
    
    
        
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
