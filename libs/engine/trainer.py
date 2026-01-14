from copy import deepcopy
import os
import time
import torch
import torch.nn as nn
from libs.utils.utils import init_seeds, attempt_compile, ModelEMA, EarlyStopping, unwrap_model, setup_model
from libs.utils.dataset import Dataset
from libs.utils.loss import DistillationLoss, ClassificationLoss, SPARSELoss
from torch.utils.tensorboard import SummaryWriter
from torch import optim, nn
import numpy as np
from tqdm import tqdm
import math

TORCH_2_4 = torch.__version__ >= "2.4"

class Trainer:
    def __init__(self, model, args, device='cuda'):
        self.model = model
        self.args = args
        self.device = device
        
        # seed
        init_seeds(self.args.seed)

        # save dir
        os.makedirs(self.args.save_dir, exist_ok=True)

        # setup
        self._setup_train()

    def _setup_model(self):
        self.ckpt, self.model = setup_model(self.model, self.args, self.device)
        
    def _setup_params(self):
        if self.args.resume: # resume
            if self.ckpt is not None:
                self.args = self.ckpt.get('args', self.args)
                self.best_fitness = self.ckpt.get('best_fitness', 0.0)
                self.start = self.ckpt.get('start', 0)
                self.global_step = self.ckpt.get('global_step', 0)
        else: # new training
            self.start = 0
            self.best_fitness = 0.0
            self.global_step = 0

    def _setup_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        g = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  

        if name == "auto":
            nc = self.args.nc

            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"AdamW", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name == 'AdamW':
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

        return optimizer

    def _setup_scheduler(self):
        def one_cycle(y1=0.0, y2=1.0, steps=100):
            return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.args.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.args.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
    
    def _load_checkpoint_state(self, ckpt):
        if ckpt is not None:
            if ckpt.get("optimizer") is not None:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scaler") is not None:
                self.scaler.load_state_dict(ckpt["scaler"])
            if self.ema and ckpt.get("ema"):
                self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
                self.ema.updates = ckpt["updates"]

    def _setup_train(self):
        self._setup_model()
        self._setup_params()

        # torch compile
        self.model = attempt_compile(self.model, self.device, self.args.imgsz, True, self.args.compile)

        # freeze layers
        self.freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        self.e_layer_names = [f"model.{x}." for x in self.freeze_list]
        for name, param in self.model.named_parameters():
            if any(x in name for x in self.e_layer_names):
                param.requires_grad = False
            elif not param.requires_grad and param.dtype.is_floating_point:
                param.requires_grad = True

        # dataloaders
        self.train_loader = Dataset(self.args).get_dataloader(self.args.train_path, self.args.batch_size, mode="train") 
        self.val_loader = Dataset(self.args).get_dataloader(self.args.val_path, self.args.batch_size, mode="val")
        self.ema = ModelEMA(self.model) if self.args.ema else None
        # amp
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.args.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.args.amp)
        )

        # optimizer
        self.accumulate = max(round(self.args.nbs / self.args.batch_size), 1)  # accumulate loss before optimizing 在 GPU 显存不足、无法设置大批次（batch_size） 时，通过「梯度累积」模拟大批次训练效果，保证训练稳定性和收敛性。
        weight_decay = self.args.weight_decay * self.args.batch_size * self.accumulate / self.args.nbs 
        iterations = math.ceil(len(self.train_loader.dataset) / self.args.batch_size) * self.args.epochs
        self.optimizer = self._setup_optimizer(self.model, self.args.optimizer, self.args.lr, self.args.momentum, weight_decay, iterations)

        # scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.scheduler.last_epoch = self.start - 1

        # tensor board writer
        self.write_path = self.args.save_dir + "/logs/"
        os.makedirs(self.write_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.write_path)

        # distillation loss
        self.distillation_loss = DistillationLoss(self.args, self.model, self.device)

        # sparse loss
        self.sparse_loss = SPARSELoss(self.args, self.device)

        # loss
        self.loss_cls = ClassificationLoss(self.args)

        # validate
        self.fitness = 0.0

        # load checkpoint for resume training
        self._load_checkpoint_state(self.ckpt)

    def _model_train(self):
        self.model.train() # set model to training mode before freezing layers
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.e_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()  # freeze batchnorms' mean and var

    def _do_val(self):
        model = self.ema.ema if self.ema else self.model
        if self.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod 
        model = model.half() if self.args.amp else model.float()
        model.eval()
        total_samples = 0
        loss_val = 0.0
        top1_correct = 0.0
        top5_correct = 0.0
        
        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            for i, batch in pbar:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                with torch.cuda.amp.autocast(self.args.amp):
                    preds = model(images)
                    loss = self.loss_cls(preds, labels)
                loss_val += loss.sum().item()
                outnum = min(self.args.nc, 5)
                top5_indices = torch.topk(preds, k=outnum, dim=1)[1]
                top1_correct += torch.topk(preds, k=1, dim=1)[1].squeeze(1).eq(labels).sum().item()
                labels_expanded = labels.unsqueeze(1).expand_as(top5_indices)
                top5_correct += top5_indices.eq(labels_expanded).any(dim=1).sum().item()

                total_samples += labels.size(0)

        avg_loss = loss_val / total_samples
        top1_accuracy = top1_correct / total_samples
        top5_accuracy = top5_correct / total_samples
        self.fitness = self.args.fitness_w[0] * avg_loss + self.args.fitness_w[1] * top1_accuracy + self.args.fitness_w[2] * top5_accuracy
        print(f"Validation Epoch {self.epoch} - Avg Loss: {avg_loss:.4f}, top1 acc: {top1_accuracy:.4f}, top5 acc: {top5_accuracy:.4f}")
        return self.fitness, avg_loss, top1_accuracy, top5_accuracy, preds, labels
    
    def _save_checkpoint(self, name):
        pt_path = self.args.save_dir + "/" + name + ".pt"
        # buffer = io.BytesIO() # Python 中用于创建 内存二进制流 的操作，它将数据临时存储在内存而非磁盘上,单次写入替代多次小文件操作
        d = {
                "start": self.epoch + 1,
                "best_fitness": self.best_fitness,
                # "model": None,
                "ema": deepcopy(unwrap_model(self.ema.ema.to(self.device))).half(),
                # "ema": (unwrap_model(self.ema.ema.to(self.device))),
                "updates": self.ema.updates,
                "optimizer": deepcopy(self.optimizer.state_dict()),
                "scaler": self.scaler.state_dict(),
                "args": self.args,  # save as dict
                "global_step": self.global_step, # SummaryWriter global step
            }
        if name == "last":
            d["model"] = {k: v.half().cpu() for k, v in unwrap_model(self.model).state_dict().items()}
        torch.save(d, pt_path)
        print(f'***pt_path: {pt_path}')
        
    def train(self):
        nb = len(self.train_loader)  # number of batches
        nw = max(round(nb * self.args.warmup_epochs), 100)  # number of warmup iterations
        last_opt_step = -1
        epoch = self.start
        while True: # epoch loop
            self.epoch = epoch
            self._model_train() # set model to training mode, freeze BN var and mean if needed
            print(f"\nEpoch: {self.epoch} - Training, lr: {self.optimizer.param_groups[0]['lr']:.6f}, momentum: {self.optimizer.param_groups[0].get('momentum', 0):.6f}")
            print(f"batch size: {self.args.batch_size}, accumulate: {self.accumulate}, total batch: {nb}")
            epoch_time_start = time.time()
            
            pbar = tqdm(enumerate(self.train_loader), total=nb)
            for i, batch in pbar:
                # warm up, set lr and momentum
                ni = i + nb * epoch  # number integrated batches (since train start)
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.args.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # forward
                with torch.cuda.amp.autocast(self.args.amp):
                    preds = self.model(batch[0].to((self.device)))
                    loss = self.loss_cls(preds, batch[1].to((self.device)))
                    loss += self.distillation_loss(self.model.get_fm(), batch[0])
                    loss += self.sparse_loss(self.model)
                    self.loss = loss.sum()

                # backward
                self.scaler.scale(self.loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.scaler.unscale_(self.optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update(self.model)
                    last_opt_step = ni
            
            self.scheduler.step()  # update lr schedule

            # validation
            self.fitness, self.v_loss, self.v_top1, self.v_top5, v_preds, v_labels = self._do_val()
            v_preds = v_preds.max(1)[1]
            # early stop
            if self.stopper(epoch, self.fitness):
                print(f"Early stopping at epoch {epoch} with fitness {self.fitness:.4f}")
                self.stop = True

            # SummaryWriter
            self.writer.add_scalars('Loss', {
                'train': self.loss,
                'val': self.v_loss
            }, self.global_step)
            self.writer.add_scalar('Acc/top1_val', self.v_top1, self.global_step)
            self.writer.add_scalar('Acc/top5_val', self.v_top5, self.global_step)
            self.writer.add_scalar('Misc/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.writer.add_scalar('Misc/momentum', self.optimizer.param_groups[0].get('momentum', 0), self.global_step)
            self.writer.add_pr_curve('PR_curve', v_preds, v_labels, self.global_step)
            
            # save
            self._save_checkpoint('last')
            if self.fitness > self.best_fitness: # save best model, only need to save ema if have
                self.best_fitness = self.fitness
                self._save_checkpoint('best')
            
            print(f"Time spend: {time.time() - epoch_time_start:.2f}s, best fitness: {self.best_fitness:.4f}, current fitness: {self.fitness:.4f}")
                
            # update epoch
            self.global_step += 1
            epoch += 1
            self.start = epoch
            if epoch >= self.args.epochs:
                break
            if self.stop:
                break