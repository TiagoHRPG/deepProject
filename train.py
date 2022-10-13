import torch 
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# preprocessing according to MobileNetV2 recomendations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# import data 
data_dir = "dataset_cancer"
sets = ['Train', 'Test']

train_dataset = train_dataset = torchvision.datasets.ImageFolder(root='dataset_cancer/Train', transform=preprocess)

test_dataset =  torchvision.datasets.ImageFolder(root='dataset_cancer/Test', transform=preprocess)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=8)


class_names = train_dataset.classes
print(class_names)

# training

def training(model, loss_f, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            #forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_f(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # statistics
            running_loss += loss.item() * input.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_loss = running_loss / da

        model.eval()


        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            #forward
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_f(outputs, labels)

            # statistics
            running_loss += loss.item() * input.size(0)
            running_corrects += torch.sum(preds == labels.data)

