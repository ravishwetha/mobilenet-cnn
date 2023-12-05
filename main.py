import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from mobilenet import MobileNet
from utils import plot_loss_acc

import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

def compute_mean_std(dataset, indices):
    # Use dataloader to efficiently extract images
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(indices), sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images, _ in loader:
        # Shape of batch: [B, C, H, W]
        # Here we compute the mean and std for the whole dataset (all batches)
        mean += images.mean([0, 2, 3])
        std += images.std([0, 2, 3])
    mean /= len(loader)
    std /= len(loader)

    return mean, std

def get_class_proportions(dataset, indices):
    # Extract all labels from the dataset assuming labels are integers from 0 to 99
    all_labels = [label for _, label in dataset]
    
    # Extract labels for the specified indices
    subset_labels = [all_labels[i] for i in indices]
    
    # Count occurrences for each class
    class_counts = {}
    for label in subset_labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    # Calculate proportions
    total_samples = len(subset_labels)
    class_proportions = {label: count / total_samples for label, count in class_counts.items()}

    return class_proportions

def get_train_valid_loader(data_dir, batch_size, shuffle=True, random_seed=0, save_images=False):

    # Placeholder transforms for initial dataset loading
    placeholder_transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset_placeholder = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=placeholder_transform)
    valid_dataset_placeholder = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=placeholder_transform)

    num_train = len(train_dataset_placeholder)
    indices = list(range(num_train))
    split = 10000  # 10,000 for validation

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    mean, std = compute_mean_std(train_dataset_placeholder, train_idx)
    print("Mean:", mean)
    print("Std:", std)

    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # No augmentation for validation but still normalize
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Redefine datasets with proper transforms
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    valid_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_val)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    # Get proportions for the training set
    train_proportions = get_class_proportions(train_dataset, train_idx)
    print(train_proportions)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def get_test_loader(data_dir, batch_size):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

def plot_loss_acc(training_loss, val_loss, training_acc, val_acc, fig_name):
    """
    Plot training and validation losses and accuracies against epochs.

    Parameters:
        - training_loss (list): Training loss values for each epoch
        - val_loss (list): Validation loss values for each epoch
        - training_acc (list): Training accuracy values for each epoch
        - val_acc (list): Validation accuracy values for each epoch
        - fig_name (str): Name to save the figure
    """

    epochs = range(1, len(training_loss) + 1)

    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # train val test
    # AI6103 students: You need to create the dataloaders youself
    train_loader, valid_loader = get_train_valid_loader(args.dataset_dir, args.batch_size, True, args.seed, save_images=args.save_images) 
    if args.test:
        test_loader = get_test_loader(args.dataset_dir, args.batch_size)

    # model
    model = MobileNet(100)
    print(model)
    model.cuda()

    # criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs)

    stat_training_loss = []
    stat_val_loss = []
    stat_training_acc = []
    stat_val_acc = []
    for epoch in range(args.epochs):
        training_loss = 0
        training_acc = 0
        training_samples = 0
        val_loss = 0
        val_acc = 0
        val_samples = 0
        # training
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()
           
            batch_size = imgs.shape[0]
            optimizer.zero_grad()
            logits = model.forward(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            _, top_class = logits.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)
            training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            training_loss += batch_size * loss.item()
            training_samples += batch_size
        # validation
        model.eval()
        for val_imgs, val_labels in valid_loader:
            batch_size = val_imgs.shape[0]
            val_logits = model.forward(val_imgs.cuda())
            loss = criterion(val_logits, val_labels.cuda())
            _, top_class = val_logits.topk(1, dim=1)

            equals = top_class == val_labels.cuda().view(*top_class.shape)
            val_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            val_loss += batch_size * loss.item()
            val_samples += batch_size
        assert val_samples == 10000
        # update stats
        stat_training_loss.append(training_loss/training_samples)
        stat_val_loss.append(val_loss/val_samples)
        stat_training_acc.append(training_acc/training_samples)
        stat_val_acc.append(val_acc/val_samples)
        # print
        print(f"Epoch {(epoch+1):d}/{args.epochs:d}.. Learning rate: {scheduler.get_lr()[0]:.4f}.. Train loss: {(training_loss/training_samples):.4f}.. Train acc: {(training_acc/training_samples):.4f}.. Val loss: {(val_loss/val_samples):.4f}.. Val acc: {(val_acc/val_samples):.4f}")
        # lr scheduler
        scheduler.step()

    # test
    if args.test:
        test_loss = 0
        test_acc = 0
        test_samples = 0
        for test_imgs, test_labels in test_loader:
            batch_size = test_imgs.shape[0]
            test_logits = model.forward(test_imgs.cuda())
            test_loss = criterion(test_logits, test_labels.cuda())
            _, top_class = test_logits.topk(1, dim=1)
            equals = top_class == test_labels.cuda().view(*top_class.shape)
            test_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            test_loss += batch_size * test_loss.item()
            test_samples += batch_size
        assert test_samples == 10000
        print('Test loss: ', test_loss/test_samples)
        print('Test acc: ', test_acc/test_samples)

    # plot
    plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, args.fig_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset_dir',type=str, help='', required=True)
    parser.add_argument('--batch_size',type=int, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--lr',type=float, help='')
    parser.add_argument('--wd',type=float, help='')
    parser.add_argument('--fig_name',type=str, help='')
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.set_defaults(lr_scheduler=False)
    parser.add_argument('--mixup', action='store_true')
    parser.set_defaults(mixup=False)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--save_images', action='store_true')
    parser.set_defaults(save_images=False)
    parser.add_argument('--seed', type=int, default=0, help='')
    args = parser.parse_args()
    # print(args)
    # main(args)

    # Ran the script using python instead of the run.sh shell script because I switched to Google Colab
    # Because failed to have Cuda enabled pytorch on MacOS

    # learning rate 0.5

    args1 = argparse.Namespace(
      dataset_dir='.',
      batch_size=128,
      epochs=15,
      lr=0.5,
      wd=0.0,  # No weight decay
      fig_name='lr=0.5-no_lr_sche-no_wd.png',
      lr_scheduler=False,  # No learning rate schedule
      mixup=False,  # Assuming don't need mixup
      test=True,
      save_images=False,
      seed=0
    )
    print(args1)
    main(args1)

    # learning rate 0.05

    args2 = argparse.Namespace(
        dataset_dir='.',
        batch_size=128,
        epochs=15,
        lr=0.05,
        wd=0.0,  # No weight decay
        fig_name='lr=0.05-no_lr_sche-no_wd.png',
        lr_scheduler=False,  # No learning rate schedule
        mixup=False,  # Assuming don't need mixup
        test=True,
        save_images=False,
        seed=0
    )
    print(args2)
    main(args2)

    # learning rate 0.01

    args3 = argparse.Namespace(
      dataset_dir='.',
      batch_size=128,
      epochs=15,
      lr=0.01,
      wd=0.0,  # No weight decay
      fig_name='lr=0.01-no_lr_sche-no_wd.png',
      lr_scheduler=False,  # No learning rate schedule
      mixup=False,  # Assuming don't need mixup
      test=True,
      save_images=False,
      seed=0
    )
    print(args3)
    main(args3)

    # learning rate 0.05, 300 epochs, no schedule

    args_constant_lr = argparse.Namespace(
        dataset_dir='.',
        batch_size=128,
        epochs=300,
        lr=0.05,
        wd=0.0,
        fig_name='lr=0.05-constant_lr-no_wd.png',
        lr_scheduler=False,
        mixup=False,
        test=True,  
        save_images=False,
        seed=0
    )
    print(args_constant_lr)
    main(args_constant_lr)

    # cosine annealing schedule

    args_cosine_annealing = argparse.Namespace(
        dataset_dir='.',
        batch_size=128,
        epochs=300,
        lr=0.05,
        wd=0.0,
        fig_name='lr=0.05-cosine_annealing-no_wd.png',
        lr_scheduler=True,  # Enabling learning rate scheduler
        mixup=False,
        test=True,  
        save_images=False,
        seed=0
    )
    print(args_cosine_annealing)
    main(args_cosine_annealing)

    # Weight Decay of 5e-4

    args_wd_5e4 = argparse.Namespace(
        dataset_dir='.',
        batch_size=128,
        epochs=300,
        lr=0.05,
        wd=5e-4,  # Weight decay of 5e-4
        fig_name='lr=0.05-cosine_annealing-wd=5e-4.png',
        lr_scheduler=True,  # Keep scheduler on
        mixup=False,
        test=True,  
        save_images=False,
        seed=0
    )
    print(args_wd_5e4)
    main(args_wd_5e4)

    # Weight Decay of 1e-4

    args_wd_1e4 = argparse.Namespace(
        dataset_dir='.',
        batch_size=128,
        epochs=300,
        lr=0.05,
        wd=1e-4,  # Weight decay of 1e-4
        fig_name='lr=0.05-cosine_annealing-wd=1e-4.png',
        lr_scheduler=True,  # Keep scheduler on
        mixup=False,
        test=True,  
        save_images=False,
        seed=0
    )
    print(args_wd_1e4)
    main(args_wd_1e4)

    # Mixup augmentation

    args_mixup = argparse.Namespace(
        dataset_dir='.',
        batch_size=128,
        epochs=300,
        lr=0.05,
        wd=5e-4,
        fig_name='lr=0.05-cosine_annealing-wd=5e-4-mixup.png',
        lr_scheduler=True,
        mixup=True,  # Enabling mixup augmentation
        alpha=0.2,
        test=True,  
        save_images=False,
        seed=0
    )
    print(args_mixup)
    main(args_mixup)