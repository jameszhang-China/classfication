import os
import random

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import math
import warnings
from libs.engine.tasker import ClassificationModel

def setup_model(model_in, args, device='cuda'):
    ckpt = None
    if isinstance(model_in, dict):
        if model_in.get('ema', None) is not None and model_in.get('model', None) is not None:
            # check point
            ckpt = model_in.to(device)
        if args.resume:
            model = model_in.get('model', None) if 'model' in model_in else model_in.get('ema', None)
        else:
            model = model_in.get('ema', None) if 'ema' in model_in else model_in.get('model', None)
        if type(model) is ModelEMA:
            model = model.ema

        if isinstance(model_in, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in model_in.items()):
            # weights
            model = ClassificationModel(args.model, device=device)
            model.load_state_dict(model_in)
            model_in = model
        elif hasattr(model, 'forward') and hasattr(model, 'state_dict'):
            # full model
            model_in = model.to(device)
        else:
            raise ValueError('model must be a weight or a model checkpoint')
    elif isinstance(model_in, ClassificationModel):
        pass
    else:
        raise ValueError('model must be checkpoint or ClassificationModel instance')
    return ckpt, model_in

def init_seeds(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe

def attempt_compile(model, device='cuda', imgsz=640, warmup=False, mode=False):
    if not hasattr(torch, "compile") or not model:
        return model
    if mode is True:
        mode = "default"
    try:
        model = torch.compile(model, mode=mode, backend="inductor")
    except Exception as e:
        Warning(f"torch.compile failed, continuing uncompiled: {e}")
        return model
    if isinstance(device, str):
        device = torch.device(device)
    if warmup:
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        with torch.inference_mode():  # 进入推理模式，自动设置 torch.no_grad()，减少内存开销，比 torch.no_grad() 额外优化计算图
            if device == "cuda":
                with torch.autocast(device.type):
                    _ = model(dummy)  # 运行模型
            else:
                _ = model(dummy)  # 运行模型
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # 同步设备，确保所有 CUDA 操作完成
    return model

def unwrap_model(m: nn.Module) -> nn.Module:
    while True:
        if hasattr(m, "_orig_mod") and isinstance(m._orig_mod, nn.Module):
            m = m._orig_mod # 被编译后的模型
        # elif hasattr(m, "module") and isinstance(m.module, nn.Module):
        #     m = m.module
        else:
            return m
        
class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(unwrap_model(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)
            msd = unwrap_model(model).state_dict() 
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

class EarlyStopping:
    def __init__(self, patience=50):
        self.best_fitness = 0.0 
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness is None:  # check if fitness=None (happens when val=False)
            return False

        if fitness > self.best_fitness or self.best_fitness == 0:  # allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            warnings.warn(
                f"Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop
    