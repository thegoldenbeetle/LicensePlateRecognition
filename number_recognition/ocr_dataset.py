import json
from pathlib import Path

import cv2
import numpy as np
import torch
import unidecode
from PIL import Image


class OCRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        data_info,
        transform=None,
        symbols_encoder=None,
    ):
        self.root_dir = Path(root_dir)
        self.nums = []
        for item in data_info:
            for num in item["nums"]:
                self.nums.append(
                    {
                        "file": item["file"],
                        **num,
                    }
                )
        self.transform = transform
        self.symbols_encoder = symbols_encoder
        self.MAX_LEN = 3 * 8

    def __len__(self):
        return len(self.nums)

    def __getitem__(self, i):
        item = self.nums[i]
        img = Image.open(self.root_dir / item["file"]).convert("RGB")
        M = cv2.getPerspectiveTransform(
            np.array(item["box"]).astype(np.float32),
            np.array(
                [
                    [0, 0],
                    [260, 0],
                    [260, 96],
                    [0, 96],
                ],
                dtype=np.float32,
            ),
        )
        img = cv2.warpPerspective(np.array(img), M, (260, 96))
        if self.transform:
            try:
                img = self.transform(img)
            except:
                print(item)
                raise
            number = unidecode.unidecode(item["text"].upper())
            if len(number) < self.MAX_LEN:
                number += "".join([" "] * (self.MAX_LEN - len(number)))
            number = np.array(
                [self.symbols_encoder.transform([x]) for x in number]
            ).flatten()
        else:
            img = img[:, :, ::-1]
            number = unidecode.unidecode(item["text"].upper())
        return img, number


def get_data_info(data_json):
    data_info = json.load((data_json).open("r"))
    ignore_list = ["train/25632.bmp"]  # сломанный файл
    data_info = [x for x in data_info if x["file"] not in ignore_list]
    return data_info
