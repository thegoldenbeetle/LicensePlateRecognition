import re
from abc import ABC, abstractmethod
from typing import List

import cv2
import torch
from paddleocr import PaddleOCR
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


class CarNumberRecognizer(ABC):
    @abstractmethod
    def recognize_number(self, license_plate_image):
        pass

class ClassicCarNumberRecognizer(CarNumberRecognizer):
    def __init__(self):
        self.ocr = PaddleOCR(lang="en", rec_algorithm="CRNN")
        self.reg_exp = re.compile(r"([a-zA-Z]\d{3}[a-zA-Z]{2})[^\d]*(\d{2,3})")

    def recognize_number(self, license_plate_image):
        ocr_result = self.ocr.ocr(license_plate_image, cls=False, det=False)
        car_number_str = max(ocr_result, key=lambda x: x[1])[0]
        car_number_str = self.reg_exp.match(car_number_str)
        if car_number_str:
            car_number_str = car_number_str.group(1) + car_number_str.group(2)
        else:
            car_number_str = ""
        return car_number_str


class MobileNetCarNumberRecognizer(nn.Module, CarNumberRecognizer):

    SYMBOLS: List[str] = [
        "A",
        "2",
        "U",
        "K",
        "H",
        "V",
        "C",
        "P",
        "6",
        "1",
        "T",
        "0",
        "3",
        "9",
        "h",
        "Y",
        "8",
        " ",
        "N",
        "E",
        "O",
        "X",
        "4",
        "5",
        "S",
        "M",
        "7",
        "R",
        "B",
    ]

    def __init__(self, weights_path=None):
        super().__init__()

        self.symbols_encoder = LabelEncoder()
        self.symbols_encoder.fit(self.SYMBOLS)
        self.transform = transforms.Compose(
            [
                transforms.Resize((96, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.features = mobilenet_v3_small().features
        self.conv = nn.Conv2d(576, 256, kernel_size=1)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.linear = nn.Linear(256, len(self.SYMBOLS))

        self.eval()
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = x.reshape((x.shape[0], x.shape[2] * x.shape[3], x.shape[1]))
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def recognize_number(self, license_plate_image):
        img = self.transform(Image.fromarray(license_plate_image[:,:,::-1]))
        with torch.no_grad():
            number_code = self.forward(img.unsqueeze(0))[0]
        car_number_str = "".join(
            self.symbols_encoder.inverse_transform(
                number_code.argmax(axis=1).detach().cpu().numpy().tolist()
            )
        ).strip()
        return car_number_str


if __name__ == "__main__":

    file_path = "./test_images/test.jpg"
    img = cv2.imread(file_path)

    number_recognizer = ClassicCarNumberRecognizer()
    number = number_recognizer.recognize_number(img)
    print(number)

    number_recognizer = MobileNetCarNumberRecognizer("./model.pt")
    number = number_recognizer.recognize_number(img)
    print(number)
