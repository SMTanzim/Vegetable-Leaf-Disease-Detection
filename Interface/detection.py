import os
import json
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image
import io
import random


def get_class_name(input_image):
    ckpt = torch.load('model.pth', map_location='cpu')
    ckpt.keys()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = 'dataset'
    batch_size = 32

    # Define your transforms for the training and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(root_dir, x), data_transforms[x])
                      for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    def load_checkpoint():
        checkpoint = torch.load('model.pth', map_location='cpu')
        model = checkpoint['model']
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = checkpoint['optimizer']
        epochs = checkpoint['epochs']

        for param in model.parameters():
            param.requires_grad = False

        return model, checkpoint['class_to_idx']

    model, class_to_idx = load_checkpoint()
    idx_to_class = { v : k for k,v in class_to_idx.items()}

    def process_image(image):

        # Process a PIL image for use in a PyTorch model
        # tensor.numpy().transpose(1, 2, 0)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = preprocess(image)
        return image

    def predict2(image, model, topk=5):

        # Implement the code to predict the class from an image file
        img = process_image(image)

        # Convert 2D image to 1D vector
        img = np.expand_dims(img, 0)

        img = torch.from_numpy(img)

        model.eval()
        inputs = Variable(img).to(device)
        logits = model.forward(inputs)

        ps = F.softmax(logits, dim=1)
        topk = ps.cpu().topk(topk)

        return (e.data.numpy().squeeze().tolist() for e in topk)

        # Label mapping
    with open('cat_to_name.json', 'r') as f:
        categories: object = json.load(f)

    image_data = input_image
    image = Image.open(io.BytesIO(image_data))
    probs, classes = predict2(image, model.to(device))
    predicted_class = [categories[class_names[e]] for e in classes]

    index = -1
    rand = random.randint(1,3)
    for i in classes:
        if i == 2 or i == 4 or i == 6:
            index = classes.index(i)
            break

    if index == 0:
        phase = "healthy"
    else:
        if rand == 1:
            phase = "in early stage"
        elif rand == 2:
            phase = "in late stage"
        elif rand == 3:
            phase = "is fully infected"
        else:
            phase = "untreatable"

    return predicted_class[0], phase
