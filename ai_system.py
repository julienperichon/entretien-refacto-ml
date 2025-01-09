import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision
from PIL import Image


class BaseModel:
    def __init__(self):
        self.default_value = 42


class PreprocessingClass:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def basic_resize(self, img):
        return img.resize(self.target_size)


class PostprocessingClass:
    def __init__(self):
        pass

    def simple_output_interpretation(self, output):
        _, predicted_idx = torch.max(output, 1)
        return predicted_idx.item()


class ImageClassifier(BaseModel, PreprocessingClass, PostprocessingClass):
    def __init__(self, target_size=(224, 224)):
        BaseModel.__init__(self)
        PreprocessingClass.__init__(self, target_size=target_size)
        PostprocessingClass.__init__(self)

        self.train_csv = "train_data.csv"
        self.test_csv = "test_data.csv"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 111 * 111, 10),
        ).to(self.device)

        self.transform_pipeline = T.Compose(
            [
                T.ToTensor(),
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.train_images = []
        self.train_labels = []

        self.test_images = []
        self.test_labels = []

    def load_data(self):
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)

        # Loading train images
        for i in range(len(train_df)):
            img_path = train_df.loc[i, "image_path"]
            label = train_df.loc[i, "label"]
            img = Image.open(img_path).convert("RGB")
            img = self.basic_resize(img)
            tensor_img = self.transform_pipeline(img)
            self.train_images.append(tensor_img)
            self.train_labels.append(label)

        # Loading test images
        for i in range(len(test_df)):
            img_path = test_df.loc[i, "image_path"]
            label = test_df.loc[i, "label"]
            img = Image.open(img_path).convert("RGB")
            img = self.basic_resize(img)
            tensor_img = self.transform_pipeline(img)
            self.test_images.append(tensor_img)
            self.test_labels.append(label)

        self.train_images = torch.stack(self.train_images).to(self.device)
        self.train_labels = torch.tensor(self.train_labels).long().to(self.device)
        self.test_images = torch.stack(self.test_images).to(self.device)
        self.test_labels = torch.tensor(self.test_labels).long().to(self.device)

    def train_model(self, epochs=2):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            outputs = self.model(self.train_images)
            loss = criterion(outputs, self.train_labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} - Loss: {loss.item()}")

    def evaluate_model(self):
        with torch.no_grad():
            outputs = self.model(self.test_images)
            predicted = torch.argmax(outputs, dim=1)
            accuracy = (predicted == self.test_labels).sum().item() / len(
                self.test_labels
            )
            print(f"Test Accuracy: {accuracy}")
        return accuracy

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.basic_resize(img)
        tensor_img = self.transform_pipeline(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor_img)
        result = self.simple_output_interpretation(output)
        return result
