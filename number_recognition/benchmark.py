from number_recognition import ClassicCarNumberRecognizer, MobileNetCarNumberRecognizer
import unidecode
import torch
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import time

def get_sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class OCRDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, data_info):
        self.root_dir = Path(root_dir)
        self.nums = []
        for item in data_info:
            for num in item["nums"]:
                self.nums.append({
                    "file": item["file"],
                    **num,
                })
        
    def __len__(self):
        return len(self.nums)
    
    def __getitem__(self, i):
        item = self.nums[i]
        img = Image.open(self.root_dir / item["file"]).convert("RGB")
        M = cv2.getPerspectiveTransform(
            np.array(item["box"]).astype(np.float32),
            np.array([
                [0, 0],
                [260, 0],
                [260, 96],
                [0, 96], 
            ], dtype=np.float32),
        )
        img = cv2.warpPerspective(
            np.array(img), M, (260, 96)
        )
        img = img[:, :, ::-1]
        number = unidecode.unidecode(item["text"].upper())
        return img, number

if __name__ == "__main__":
    
    file_path = "./test_images/test.jpg"
    img = cv2.imread(file_path)

    ROOT_DIR = Path("./data/data/")
    data_info = json.load((ROOT_DIR / "train.json").open("r"))
    ignore_list = ["train/25632.bmp"]
    data_info = [x for x in data_info if x["file"] not in ignore_list]

    dataset = OCRDataset(ROOT_DIR, data_info)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=80, batch_size=1, collate_fn=lambda batch: batch[0])

    classic_number_recognizer = ClassicCarNumberRecognizer()
    mobilenet_number_recognizer = MobileNetCarNumberRecognizer("./model.pt")

    classic_results = np.empty(len(dataset), dtype=bool)
    mobilenet_results = np.empty(len(dataset), dtype=bool)
    classic_time = 0.0
    mobilenet_time = 0.0
    for i, (img, real_number) in tqdm(enumerate(dataloader), total=len(dataset)):
        if i % 1000 == 0:
            print(f"Acc classic: {classic_results[:i].mean()}")
            print(f"Acc mobilenet: {mobilenet_results[:i].mean()}")
        time_1 = get_sync_time()
        classic_results[i] = classic_number_recognizer.recognize_number(img) == real_number
        time_2 = get_sync_time()
        mobilenet_results[i] = mobilenet_number_recognizer.recognize_number(img) == real_number
        time_3 = get_sync_time()
        classic_time += (time_2 - time_1)
        mobilenet_time += (time_3 - time_2)

    print(f"Acc classic: {classic_results.mean()}, {classic_time / len(dataset):.6f} s")
    print(f"Acc mobilenet: {mobilenet_results.mean()}, {mobilenet_time / len(dataset):.6f} s")

# Results
# Acc classic: 0.34646521171423245, 0.000290 s
# Acc mobilenet: 0.9201268420070883, 0.000040 s