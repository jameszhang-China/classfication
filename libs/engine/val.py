import torch
import torch.nn as nn
from libs.utils.utils import setup_model
from libs.utils.dataset import Dataset
import tqdm as TQDM

class ClassifierValidator:
    def __init__(self, model, args:dict, device='cuda'):
        self.model = model
        self.args = args
        self.device = device
        self.val_loader = Dataset(self.args).get_dataloader(self.args.val_path, self.args.batch_size, mode="val")
    def __call__(self):
        if self.args.compile and hasattr(self.model, "_orig_mod"):
            self.model = self.model._orig_mod 
        self.model = self.model.half() if self.args.amp else self.model.float()
        self.model.eval()
        total_samples = 0
        top1_correct = 0.0
        top5_correct = 0.0
        
        with torch.no_grad():
            pbar = TQDM.tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            for i, batch in pbar:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                with torch.cuda.amp.autocast(self.args.amp):
                    preds = self.model(images)
                outnum = min(self.args.nc, 5)
                top5_indices = torch.topk(preds, k=outnum, dim=1)[1]
                top1_correct += torch.topk(preds, k=1, dim=1)[1].squeeze(1).eq(labels).sum().item()
                labels_expanded = labels.unsqueeze(1).expand_as(top5_indices)
                top5_correct += top5_indices.eq(labels_expanded).any(dim=1).sum().item()

                total_samples += labels.size(0)

        top1_accuracy = top1_correct / total_samples
        top5_accuracy = top5_correct / total_samples
        return top1_accuracy, top5_accuracy

class Validator:
    def __init__(self, model, args, device='cuda'):
        _, model = setup_model(model, args, device)
        self.valider = ClassifierValidator(model, args, device)

    def validate(self):
        # TODO: save result
        top1, top5 = self.valider()
        print(f"Top1 acc: {top1:.2f}, Top5 acc: {top5:.2f}")
        return top1, top5