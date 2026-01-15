import torch
import torch.nn as nn
from libs.utils.utils import setup_model

class Exporter:
    def __init__(self, model, args, device='cuda'):
        _, self.model = setup_model(model, args, device)
        self.device = device
        self.args = args

    def export(self):
        if self.args.compile and hasattr(self.model, "_orig_mod"):
            self.model = self.model._orig_mod 
        self.model.eval()
        dummy_input = torch.randn(1, 3, self.args.imgsz, self.args.imgsz).to(self.device)
        
        torch.onnx.export(
            self.model,                          # 要导出的模型
            dummy_input,                    # 模型输入
            self.args.export_path + "/model.onnx",                   # 输出文件名
            export_params=True,             # 存储训练好的参数
            opset_version=18,               # ONNX 算子集版本
            do_constant_folding=True,       # 是否执行常量折叠优化
            input_names=['input'],          # 输入节点名称
            output_names=['output'],        # 输出节点名称
            dynamo=False
        )
        traced_model = torch.jit.trace(self.model, dummy_input)
        traced_model.save(self.args.export_path + "/model.torchscript")

'''
特性        Tracing (torch.jit.trace)   Scripting (torch.jit.script)
条件判断    只记录示例输入的分支    保留所有分支
控制流      静态记录               动态执行
适用场景    简单无分支模型          包含条件判断的模型
'''