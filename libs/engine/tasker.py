import copy
import torch
import torch.nn as nn
import yaml
from libs.modules.block import Conv, ResNetBasicLayer, ResNetBottleneckLayer, Concat

def parse_model(d : dict, ch=3, device='cuda'):
    base_modules = frozenset({
        Conv,
        ResNetBasicLayer,
        ResNetBottleneckLayer,
    })
    repeat_models = frozenset({
        ResNetBasicLayer,
        ResNetBottleneckLayer
    })

    ch = [ch]
    layers = []
    c2 = ch[-1]

    for i, (f, n, m, args)in enumerate(d['backbone'] + d["head"]):
        m = (
            getattr(torch.nn, m[3:]) if "nn." in m
            else globals()[m] # get from globals() if not in torch.nn
        )
     
        if m in base_modules: # init param is (in,out,...), means input channel not same as output channel
            c1, c2 = ch[f], args[0] # channel in, channel out
            args = [c1, c2, *args[1:]]
            if m in repeat_models: # init param is (in,out,num,...), means repeat inside the layer
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is Concat: # param is (dimension)
            c2 = sum(ch[x] for x in f)
        elif m is nn.Flatten or m is nn.Linear: # output channel is 0
            c2 = 0
        else:
            c2 = ch[f] # input channel same as output channel
        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        m_.i, m_.f, m_.type = i, f, str(m)[8:-2].replace("__main__.", "") # jump <class ' and > from module name
        m_.np = sum(x.numel() for x in m_.parameters()) if hasattr(m_, 'parameters') else 0 # number of parameters
        layers.append(m_)
        ch.append(c2)

    return torch.nn.Sequential(*layers).to(device)

def yaml_model_load(cfg : str):
    with open(cfg, 'r') as file:
        d = yaml.load(file, Loader=yaml.SafeLoader)
    return d

class ClassificationModel(nn.Module):
    def __init__(self, cfg : str, device='cuda'):
        super(ClassificationModel, self).__init__()
        self.yaml = yaml_model_load(cfg)
        self.model = parse_model(self.yaml, device=device)
 
        self.fm = []
        self.device = device

    def forward(self, x):
        self.fm = []
        for m in self.model:
            if m.f != -1:
                x = self.fm[m.f] if isinstance(m.f, int) else [x if j == -1 else self.fm[j] for j in m.f]  # from earlier layers
            x = m(x)
            self.fm.append(x.clone().to(self.device))
        return x
    
    def get_fm(self):
        return self.fm.copy()
