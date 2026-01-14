import os
from types import SimpleNamespace
import torch
import torch.nn as nn
import yaml

from libs.engine.tasker import ClassificationModel
from libs.engine.trainer import Trainer
from libs.engine.val import Validator
from libs.engine.detect import Detector
from libs.engine.exporter import Exporter
from libs.engine.prune import Pruner

class Classification:
    def __init__(self, model: str, device='cuda', hyper=None):
        if not hyper:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.yaml = os.path.join(current_dir, '../../cfg/default.yaml')
        else:
            self.yaml = hyper

        self.device = device
        with open(self.yaml, 'r') as f:
            hyper = yaml.safe_load(f)

        self._get_hyper(hyper)

        if model.endswith(('.pt', '.pth')):
            self.model = torch.load(model, map_location=device, weights_only=False)
        else:
            self.model = ClassificationModel(model, self.args, device)
    
    
    def _get_hyper(self, hyper):
        self.args = SimpleNamespace()

        # resume
        self.args.resume = hyper.get('resume', False)

        # model info
        self.args.model = hyper.get('model', '') # model yaml path, used when model is only weight
        self.args.nc = hyper.get('nc', 10) # model number of classes
        self.args.nc_name = hyper.get('nc_name', []) # model class name list

        # data info
        self.args.imgsz = hyper.get('imgsz', 224)
        self.args.batch_size = hyper.get('batch_size', 16)
        self.args.mean = hyper.get('mean', (0.485, 0.456, 0.406))
        self.args.std = hyper.get('std', (0.229, 0.224, 0.225))
        self.args.workers = hyper.get('workers', 4)

        # path
        self.args.train_path = hyper.get('train_path', '')
        self.args.val_path = hyper.get('val_path', '')
        self.args.detect_path = hyper.get('detect_path', '') # should be a folder, img or video
        self.args.export_path = hyper.get('export_path', '') # export model path

        # trainer params
        self.args.epochs = hyper.get('epochs', 10)
        self.args.lr = hyper.get('lr', 0.001)
        self.args.weight_decay = hyper.get('weight_decay', 0.0005)
        self.args.seed = hyper.get('seed', 42)
        self.args.lrf = hyper.get('lrf', 0.01)
        self.args.cos_lr = hyper.get('cos_lr', False)
        self.args.optimizer = hyper.get('optimizer', 'auto')
        self.args.momentum = hyper.get('momentum', 0.9)
        self.args.warmup_bias_lr = hyper.get('warmup_bias_lr', 0.1)
        self.args.warmup_epochs = hyper.get('warmup_epochs', 3)
        self.args.warmup_momentum = hyper.get('warmup_momentum', 0.8)
        self.args.nbs = hyper.get('nbs', 64)
        self.args.patience = hyper.get('patience', 50) # for early stop
        self.args.amp = hyper.get('amp', True)
        self.args.freeze = hyper.get('freeze', []) # freeze training layers
        self.args.compile = hyper.get('compile', False) # torch.compile
        self.args.ema = hyper.get('ema', True) # ema
        self.args.fitness_w = hyper.get('fitness_w', [0.0, 0.5, 0.5]) # weights for fitnesses
        self.args.sparse_train = hyper.get('sparse_train', False) # sparse training for prune
        self.args.sparse_weight = hyper.get('sparse_weight', 0.2) # sparse loss weight for prune
        self.args.sparse_list = hyper.get('sparse_list', []) # sparse layer list for prune

        # pruner params
        self.args.prune_ratio = hyper.get('prune_ratio', 0.2) # prune ratio for prune

        # log/result save
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_save_dir = os.path.join(current_dir, '../runs/train/default/')
        self.args.save_dir = hyper.get('save_dir', default_save_dir)
        while True:
            if os.path.exists(self.args.save_dir) and not self.args.resume:
                self.args.save_dir = self.args.save_dir[:-1] + '_t/'
            else:
                break

        # distillation
        self.args.distillation = hyper.get('distillation', False)
        self.args.study_list = hyper.get('study_list', []) # study layer list, used when distillation is True
        self.args.teacher = hyper.get('teacher', '') # teacher model path, used when distillation is True
        self.args.t_model = hyper.get('t_model', '') # teacher model yaml, used when teacher model is only weight
        self.args.feature_loss_weight = hyper.get('feature_loss_weight', 0.2) # feature loss weight for distillation
        self.args.output_loss_weight = hyper.get('output_loss_weight', 0.8) # output loss weight for distillation
        self.args.distillation_loss_weight = hyper.get('distillation_loss_weight', 0.5) # distillation loss weight
        self.args.distillation_loss_type = hyper.get('distillation_loss_type', 'MGD') # KD, MGD or CWD for middle layer feature map loss
        self.args.mgd_mask_ratio = hyper.get('mgd_mask_ratio', 0.5) # mask ratio for MGD distillation loss
        self.args.temperature = hyper.get('temperature', 1.0) # temperature for distillation loss

        # augment
        self.args.scale = hyper.get('scale', (0.08, 1.0))
        self.args.fliplr = hyper.get('fliplr', 0.5)
        self.args.flipup = hyper.get('flipup', 0.0)
        self.args.ratio = hyper.get('ratio', (3./4., 4./3.))
        self.args.hflip = hyper.get('hflip', 0.5)
        self.args.vflip = hyper.get('vflip', 0.0)
        self.args.color_jitter = hyper.get('color_jitter', 0.4)
        self.args.auto_augment = hyper.get('auto_augment', True)
        self.args.hsv_enabled = hyper.get('hsv_enabled', True)
        self.args.hsv_h = hyper.get('hsv_h', 0.015)
        self.args.hsv_s = hyper.get('hsv_s', 0.4)
        self.args.hsv_v = hyper.get('hsv_v', 0.4)
        self.args.interpolation = hyper.get('interpolation', 'bicubic')

    def train(self):
        self.trainer = Trainer(self.model, self.args, self.device)
        self.trainer.train()

    def predict(self):
        self.detect = Detector(self.model, self.args, self.device)
        self.detect.predict()

    def val(self):
        self.validator = Validator(self.model, self.args, self.device)
        self.validator.validate()

    def prune(self):
        self.pruner = Pruner(self.model, self.args, self.device)
        self.pruner.prune()

    def export(self):
        self.exporter = Exporter(self.model, self.args, self.device)
        self.exporter.export()