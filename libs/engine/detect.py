import os
from tkinter import Image
import torch
import torch.nn as nn
from libs.utils.utils import setup_model
import torchvision.transforms as T

class Detector:
    def __init__(self, model, args, device='cuda'):
        _, self.model = setup_model(model, args, device)
        self.support_img = ['jpg', 'jpeg', 'png', 'bmp', ]
        self.support_video = [] # TODO: support video

    def predict(self):
        if self.args.compile and hasattr(self.model, "_orig_mod"):
            self.model = self.model._orig_mod 
        self.model = self.model.half() if self.args.amp else self.model.float()
        self.model.eval()

        if not os.path.exists(self.args.detect_path):
            raise ValueError(f'detect_path not exists:{self.args.detect_path}')
        
        # TODO: support folder
        if self.args.detect_path.endswith(self.support_img):
            self.predict_img()
        # TODO: support video
        # elif self.args.detect_path.endswith(self.support_video):
        #     self.predict_video()
        else:
            raise ValueError(f'format not support')

    def predict_img(self):
        img_path = self.args.detect_path
        transform = T.Compose([
            T.Resize((self.args.imgsz, self.args.imgsz)),  # 调整图像大小
            T.ToTensor(),  # 转换为张量
            T.Normalize(mean=self.args.mean, std=self.args.std)  # 标准化
        ])
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            probabilities = torch.softmax(output, dim=1)
            outnum = min(self.args.nc, 5)
            top5_values, top5_indices = torch.topk(probabilities, k=outnum, dim=1)
            top5_out = [(f'{self.args.nc_name[top5_indices[i].item()]}: {top5_values[i].item():.4f}') for i in range(outnum)]
        for i in top5_out:
            print(i)
        # TODO: save result to img
        return top5_out