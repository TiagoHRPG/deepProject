import pytorch_lightning as pl
from transfer_pl import ShuffleNetModel, MobileNetModel
from torchvision import transforms
import torchvision
import torch

# preprocessing according to MobileNetV2 recomendations
preprocess_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandAugment(3),
    transforms.ToTensor(),
    #transforms.RandomHorizontalFlip(0.5), # 53% com augmentation
    #transforms.RandomRotation(45),
    #transforms.GaussianBlur(3),
    #transforms.ColorJitter(),
    #transforms.RandomPerspective(),
    #transforms.RandomEqualize(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

preprocess_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(root='dataset_cancer/Train', transform=preprocess_train)

# ImageFolder
test_dataset =  torchvision.datasets.ImageFolder(root='dataset_cancer/Test', transform=preprocess_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=8)

# DataLoader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=24, shuffle=True, num_workers=8)

mobile_model = MobileNetModel((3,256, 256), 9, transfer=False)
shuffle_model = ShuffleNetModel((3,256, 256), 9, transfer=False)

mobile_model.load_state_dict('mobile_model_state.pt')
shuffle_model.load_state_dict('shuffle_model_state.pt')

# Instanciando os Trainers
trainer_mobile = pl.Trainer(max_epochs=20)
trainer_shuffle = pl.Trainer(max_epochs=20)

# Treinando, testando  e salvando os modelos 
trainer_mobile.fit(mobile_model, train_loader)
trainer_mobile.test(mobile_model, test_loader)

torch.save(mobile_model.state_dict(), "mobile_model_state.pt")


trainer_shuffle.fit(shuffle_model, train_loader)
trainer_shuffle.test(shuffle_model, test_loader)

torch.save(shuffle_model.state_dict(), "shuffle_model_state.pt")