import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ocr_dataset import OCRDataset, get_data_info
from tqdm import tqdm

from number_recognition import (
    ClassicCarNumberRecognizer,
    MobileNetCarNumberRecognizer,
)


def get_sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def experiment(
    classic_number_recognizer,
    mobilenet_number_recognizer,
    dataset,
    dataloader,
):
    classic_results = np.empty(len(dataset), dtype=bool)
    mobilenet_results = np.empty(len(dataset), dtype=bool)
    classic_time = 0.0
    mobilenet_time = 0.0
    for i, (img, real_number) in tqdm(
        enumerate(dataloader), total=len(dataset)
    ):
        if i % 100 == 0:
            print(f"Acc classic: {classic_results[:i].mean()}")
            print(f"Acc mobilenet: {mobilenet_results[:i].mean()}")
        time_1 = get_sync_time()
        classic_results[i] = (
            classic_number_recognizer.recognize_number(img) == real_number
        )
        time_2 = get_sync_time()
        mobilenet_results[i] = (
            mobilenet_number_recognizer.recognize_number(img) == real_number
        )
        time_3 = get_sync_time()
        classic_time += time_2 - time_1
        mobilenet_time += time_3 - time_2

    print(
        f"Acc classic: {classic_results.mean()}, {classic_time / len(dataset):.6f} s"
    )
    print(
        f"Acc mobilenet: {mobilenet_results.mean()}, {mobilenet_time / len(dataset):.6f} s"
    )


if __name__ == "__main__":

    file_path = "./test_images/test.jpg"
    img = cv2.imread(file_path)

    ROOT_DIR = Path("./data/data/")
    data_info = get_data_info(ROOT_DIR / "train.json")

    dataset = OCRDataset(ROOT_DIR, data_info)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        num_workers=80,
        batch_size=1,
        collate_fn=lambda batch: batch[0],
    )

    classic_number_recognizer = ClassicCarNumberRecognizer()
    mobilenet_number_recognizer = MobileNetCarNumberRecognizer("./model.pt")

    experiment(
        classic_number_recognizer,
        mobilenet_number_recognizer,
        dataset,
        dataloader,
    )

# Results
# Acc classic: 0.34646521171423245, 0.000290 s
# Acc mobilenet: 0.9201268420070883, 0.000040 s
