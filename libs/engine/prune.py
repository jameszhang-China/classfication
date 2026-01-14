from copy import deepcopy
import torch
import torch.nn as nn
import torch_pruning as tp
from libs.utils.utils import setup_model, unwrap_model

class Pruner:
    def __init__(self, model, args:dict, device='cuda'):
        self.args = args
        self.device = device
        _, self.model = setup_model(model, args, device)

    def prune(self):
        if self.args.compile and hasattr(self.model, "_orig_mod"):
            self.model = self.model._orig_mod 
        self.model = self.model.half() if self.args.amp else self.model.float()
        self.model.eval()

        dummy_input = torch.randn(1, 3, self.args.img_size, self.args.img_size, device=self.device) # 运行，自动追踪并构建整个模型的「层间依赖关系图」
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=dummy_input)
        print(f"after pruning,\n parameters: {sum(p.numel() for p in self.model.parameters()) + sum(b.numel() for b in self.model.buffers()) / 1e6:.2f}M")
        print(f"grad: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}M")
        for i, (name, block) in enumerate(self.model.named_children()):
            if i in self.args.sparse_list:
                for module_name, module in block.named_modules():
                    if isinstance(module, nn.Conv2d):
                        if module.out_channels <= 16: # 保护极小通道层
                            continue
                        # TODO: 基于 BN 的准则更能反映通道在实际训练中的真实贡献，剪枝更精准，精度损失更小，这对嵌入式部署场景至关重要（基础精度越高，轻量化后的精度越有保障）
                        w = module.weight.data.clone().detach()
                        l1_norm = torch.norm(w, p=1, dim=(1, 2, 3)) # L1 norm of output_channels, weights: (output_channels, in_channels, kernel_size_h, kernel_size_w)
                        num_channels = module.out_channels
                        n_pruned = int(num_channels * self.args.prune_ratio)
                        if n_pruned > 0:
                            pruning_idxs = torch.argsort(l1_norm)[:n_pruned]
                            pruning_plan = DG.get_pruning_plan( # 生成剪枝策略
                                module, tp.prune_conv_out_channels, pruning_idxs)
                            if pruning_plan is not None:
                                pruning_plan.exec() # 开始剪枝
                                print(f"Pruned layer: {name}.{module_name}, channels: {num_channels} -> {num_channels - n_pruned}")
        print(f"after pruning,\n parameters: {sum(p.numel() for p in self.model.parameters()) + sum(b.numel() for b in self.model.buffers()) / 1e6:.2f}M")
        print(f"grad: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}M")
        self._save_prune(f"pruned")

    def _save_prune(self, name):
        pt_path = self.args.save_dir + "/" + name + ".pt"
        d = {
                "start": 0,
                "best_fitness": 0,
                "model": deepcopy(unwrap_model(self.model)).half(), # save as half precision
                "ema": None,
                "updates": None,
                "optimizer": None,
                "scaler": None,
                "args": self.args,  # save as dict
                "global_step": None, # SummaryWriter global step
            }
        torch.save(d, pt_path)
        print(f'***pt_path: {pt_path}')