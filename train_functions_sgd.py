from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from tqdm.auto import tqdm
import torch
import pickle


def imshow(inp, title=None):
    """
    Imshow for Tensor.
    plots the tensor as image

    inp: input tensor
    title: title to the plot
    """

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    """
      train_model:
                 model: defined model (Resnet18)
                 criterion: cross entropy loss function
                 optimizer: optimization algorithm Stochastic Gradient Descent (SGD)
                 scheduler: decay LR by a factor
                 dataloaders: batch of images, 16 images per batch
                 dataset_sizes: size of train and val samples
                 device: GPU or CPU
                 num_epochs: number of epochs to train in the data
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    losses = []
    accs = []

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses.append((phase,epoch_loss))
            accs.append((phase,epoch_acc.data.item()))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    pickle.dump(losses, open("model/losses.pkl", 'wb'), protocol=4)
    pickle.dump(accs, open("model/accs.pkl", 'wb'), protocol=4)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, device, dataloaders, num_images=6, ds='val'):
    """
    Plots to plot as image n number of tensors

    model: trained model
    device: GPU or CPU
    dataloaders: batch of images
    num_images: images to plot
    ds: type of dataset (train, val or test)
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[ds]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'pred: {preds[j]} / gt: {labels[j]} / ds: {ds}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def test_model(model, criterion, device, dataloaders, dataset_sizes, phase):
    """
    Function to evaluate the model with a test dataset.

	model: trained model
    criterion: cross entropy loss function
    device: GPU or CPU
    dataloaders: batch of images
    dataset_sizes: size of val or test samples
    phase: type of dataset (val or test)
    """

    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    y_true = torch.ones(1, dtype=torch.long).to(device)  # to generate numpy arrays for return
    y_pred = torch.ones(1, dtype=torch.long).to(device)  # --//--

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        y_true = torch.cat((y_true, labels))  # add batch of label to tensor

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            y_pred = torch.cat((y_pred, preds))  # add batch of label to tensor

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / dataset_sizes[phase]
    test_acc = running_corrects.double() / dataset_sizes[phase]

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Acc: {test_acc:.4f}')

    y_true = y_true[1:].to('cpu').detach().numpy()
    y_pred = y_pred[1:].to('cpu').detach().numpy()

    return y_true, y_pred


