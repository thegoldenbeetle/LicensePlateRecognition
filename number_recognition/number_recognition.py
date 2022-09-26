import re
from abc import ABC, abstractmethod

import cv2
from paddleocr import PaddleOCR


class CarNumberRecognizer(ABC):
    @abstractmethod
    def recognize_number(self, license_plate_image):
        pass


class ClassicCarNumberRecognizer(CarNumberRecognizer):
    def __init__(self):
        self.ocr = PaddleOCR(lang="en", rec_algorithm="CRNN")
        self.reg_exp = re.compile(r"([a-zA-Z]\d{3}[a-zA-Z]{2}\d{2,3})")

    def recognize_number(self, license_plate_image):
        ocr_result = self.ocr.ocr(license_plate_image, cls=False, det=False)
        car_number_str = max(ocr_result, key=lambda x: x[1])[0]
        car_number_str = self.reg_exp.match(car_number_str).group(1)
        return car_number_str


if __name__ == "__main__":

    file_path = "./test_images/test.jpg"
    img = cv2.imread(file_path)

    number_recognizer = ClassicCarNumberRecognizer()
    number = number_recognizer.recognize_number(img)

    print(number)
