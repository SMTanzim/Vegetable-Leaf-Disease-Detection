import os
import time
import copy
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch.nn as nn


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = 'dataset'
    # train_dir = root_dir + '/train'
    # valid_dir = root_dir + '/valid'
    batch_size = 32

    # Define your transforms for the training and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(root_dir, x), data_transforms[x])
                      for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    num_class = len(class_names)
    print(dataset_sizes)

    # # Label mapping
    # with open('categories.json', 'r') as f:
    #     categories: object = json.load(f)

    # # Run this to test the data loader
    # images, labels = next(iter(dataloaders['train']))
    # rand_idx = np.random.randint(len(images))
    # # Print(rand_idx)
    # print("label: {}, class: {}, name: {}".format(labels[rand_idx].item(),
    #                                               class_names[labels[rand_idx].item()],
    #                                               categories[class_names[labels[rand_idx].item()]]))

    def build_classifier(input_features, hidden_layers, output_features):
        classifier = nn.conv2d()
        if hidden_layers is None:
            classifier.add_module('fc0', nn.conv2d(input_features, output_features))
        else:
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            classifier.add_module('fc0', nn.conv2d(input_features, hidden_layers[0]))
            classifier.add_module('relu0', nn.ReLU())
            classifier.add_module('drop0', nn.Dropout(.6))
            classifier.add_module('relu1', nn.ReLU())
            classifier.add_module('drop1', nn.Dropout(.5))
            for i, (h1, h2) in enumerate(layer_sizes):
                classifier.add_module('fc' + str(i + 1), nn.Linear(h1, h2))
                classifier.add_module('relu' + str(i + 1), nn.ReLU())
                classifier.add_module('drop' + str(i + 1), nn.Dropout(.5))
            classifier.add_module('output', nn.Linear(hidden_layers[-1], output_features))

        return classifier

    def train_model(model, criterion, optimizer, sched, num_epochs):

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    print("in training mode:")
                    model.train()  # Set model to training mode
                else:
                    print("in evaluation mode:")
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    print('iterating over data')
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    print('forward propagation')
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            print('backward propagation')
                            sched.step()
                            loss.backward()

                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model

    # Create classifier
    model = models.vgg19(pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    num_in_features = 25088
    num_hidden_layers = None
    num_out_features = 9
    new_classifier = build_classifier(num_in_features, num_hidden_layers, num_out_features)
    # model.classifier = new_classifier
    # train model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    epochs = 1
    model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, epochs)
    # Evaluation
    model.eval()
    accuracy = 0

    for inputs, labels in dataloaders['valid']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Class with the highest probability is our predicted class
        equality = (labels.data == outputs.max(1)[1])

        # Accuracy is number of correct predictions divided by all predictions
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Test accuracy: {:.3f}".format(accuracy / len(dataloaders['valid'])))

    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'input_size': dataset_sizes['train'],
                  'output_size': num_class,
                  'epochs': epochs,
                  'batch_size': batch_size,
                  'model': models.vgg19(pretrained=True),
                  'classifier': new_classifier,
                  'scheduler': scheduler,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                  }

    torch.save(checkpoint, 'trained_model.pth')


if __name__ == '__main__':
    main()