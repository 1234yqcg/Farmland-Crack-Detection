import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.yolov10_crack import YOLOv10Crack
from evaluate import evaluate_map
from utils.roboflow_dataset import RoboflowFarmlandDataset
from utils.logger import setup_logger, AverageMeter

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

def collate_fn(batch):
    images = []
    targets = []
    for item in batch:
        images.append(item['image'])
        targets.append(item['labels'])
    images = torch.stack(images, dim=0)
    return {'images': images, 'targets': targets}

class Trainer:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = self._resolve_path(self.config['output']['dir'])
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'weights'), exist_ok=True)
        self.logger = setup_logger('Trainer', log_dir=self.output_dir)
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'logs'))
        
        self.reg_max = 16  # Fixed as per config/model
        self.strides = [8, 16, 32]
        
        self._init_model()
        self._init_data()
        self._init_training()

    def _resolve_path(self, path):
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(os.path.dirname(__file__), path))

    def _safe_load_checkpoint(self, path):
        last_error = None
        for weights_only in (True, False):
            try:
                return torch.load(path, map_location=self.device, weights_only=weights_only)
            except Exception as exc:
                last_error = exc
        raise last_error

    def _init_model(self):
        num_classes = self.config['model']['num_classes']
        self.model = YOLOv10Crack(num_classes=num_classes, reg_max=self.reg_max).to(self.device)
        
        pretrained = self._resolve_path(self.config['model'].get('pretrained'))
        print(f"\n{'='*50}")
        print(f"Using pretrained weights: {pretrained}")
        print(f"{'='*50}\n")
        
        if pretrained and os.path.exists(pretrained):
            try:
                ckpt = self._safe_load_checkpoint(pretrained)
                state_dict = ckpt.get('model', ckpt)
                if hasattr(state_dict, 'state_dict'):
                    state_dict = state_dict.state_dict()
                model_dict = self.model.state_dict()
                
                loaded = 0
                for k, v in state_dict.items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        model_dict[k] = v
                        loaded += 1
                
                self.model.load_state_dict(model_dict)
                self.logger.info(f"Loaded {loaded} layers from pretrained weights")
                print(f"Loaded {loaded} layers from pretrained weights")
            except Exception as e:
                self.logger.warning(f"Failed to load pretrained weights: {e}")
                import traceback
                traceback.print_exc()

    def _init_data(self):
        if 'dataset_yaml' in self.config.get('data', {}):
            dataset_yaml = self._resolve_path(self.config['data']['dataset_yaml'])
        else:
            dataset_yaml = os.path.join(os.path.dirname(__file__), 'data', 'dataset.yaml')
        self.dataset_yaml = dataset_yaml
        img_size = tuple(self.config['training']['image_size'])
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training'].get('num_workers', 4)
        
        train_dataset = RoboflowFarmlandDataset(dataset_yaml, 'train', img_size, augment=True)
        self.class_names = train_dataset.class_names
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=collate_fn, pin_memory=True, drop_last=True
        )
        
        data_cfg = train_dataset.config
        val_dir = data_cfg.get('val')
        if val_dir and not os.path.isabs(val_dir):
            val_dir = os.path.join(os.path.dirname(dataset_yaml), val_dir)
        if not val_dir:
            val_dir = os.path.join(os.path.dirname(dataset_yaml), 'val', 'images')
        val_split = 'val'
        if not os.path.exists(val_dir) or not os.listdir(val_dir):
            self.logger.warning(f"Validation directory {val_dir} is empty or missing. Using 'test' split for validation.")
            val_split = 'test'
        
        self.val_loader = DataLoader(
            RoboflowFarmlandDataset(dataset_yaml, val_split, img_size, augment=False),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=collate_fn, pin_memory=True
        )

    def _init_training(self):
        training_cfg = self.config['training']
        optimizer_cfg = training_cfg.get('optimizer', {})
        scheduler_cfg = training_cfg.get('scheduler', {})
        amp_cfg = training_cfg.get('amp', {})
        accumulation_cfg = training_cfg.get('gradient_accumulation', {})
        self.optimizer = AdamW(self.model.parameters(), 
                              lr=training_cfg['lr'],
                              betas=tuple(optimizer_cfg.get('betas', [0.9, 0.999])),
                              weight_decay=training_cfg['weight_decay'])
        self.scheduler = CosineAnnealingLR(self.optimizer, 
                                          T_max=training_cfg['epochs'],
                                          eta_min=training_cfg['min_lr'])
        self.use_amp = amp_cfg.get('enabled', True) and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        self.accumulation_steps = accumulation_cfg.get('steps', 1) if accumulation_cfg.get('enabled', False) else 1
        self.warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)
        self.warmup_lr = scheduler_cfg.get('warmup_lr', training_cfg['lr'])
        self.base_lr = training_cfg['lr']
        self.grad_clip_norm = training_cfg.get('grad_clip_norm')
        eval_cfg = training_cfg.get('evaluation', {})
        self.eval_conf = eval_cfg.get('conf_threshold', 0.25)
        self.eval_iou = eval_cfg.get('iou_threshold', 0.5)
        self.eval_period = eval_cfg.get('period', 5)
        self.best_loss = float('inf')
        self.best_map = float('-inf')
        # 训练概要信息
        self.logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        self.logger.info(f"Epochs: {self.config['training']['epochs']}")
        self.logger.info(f"Batch size: {self.config['training']['batch_size']}")
        try:
            self.logger.info(f"训练集样本数: {len(self.train_loader.dataset)}")
            self.logger.info(f"验证集样本数: {len(self.val_loader.dataset)}")
        except Exception:
            pass

    def _sigmoid_focal_loss(self, logits, targets, alpha=0.25, gamma=1.5):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if gamma <= 0:
            alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
            return (alpha_factor * bce_loss).sum()
        pred_prob = torch.sigmoid(logits)
        p_t = targets * pred_prob + (1 - targets) * (1 - pred_prob)
        alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
        modulating_factor = (1.0 - p_t).pow(gamma)
        return (alpha_factor * modulating_factor * bce_loss).sum()

    def _get_checkpoint(self, epoch):
        return {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'config_path': self.config_path,
            'num_classes': self.config['model']['num_classes'],
            'class_names': self.class_names
        }

    def _apply_warmup(self, epoch, step_idx, num_steps):
        if self.warmup_epochs <= 0 or epoch > self.warmup_epochs:
            return
        total_warmup_steps = max(1, self.warmup_epochs * num_steps)
        current_step = (epoch - 1) * num_steps + step_idx + 1
        progress = min(current_step / total_warmup_steps, 1.0)
        warmup_lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * progress
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def _compute_loss(self, outputs, targets):
        """
        Compute loss for anchor-free YOLO
        outputs: list of (cls_pred, reg_pred) for each stride
        targets: list of [class, x, y, w, h] normalized
        """
        loss_cfg = self.config['training']['loss']
        cls_loss_val = torch.tensor(0., device=self.device)
        box_loss_val = torch.tensor(0., device=self.device)
        dfl_loss_val = torch.tensor(0., device=self.device)
        
        total_samples = 0
        focal_gamma = loss_cfg.get('focal_gamma', 0.0)
        focal_alpha = loss_cfg.get('focal_alpha', 0.25)
        label_smoothing = loss_cfg.get('label_smoothing', 0.0)
        
        for i, (cls_pred, reg_pred) in enumerate(outputs):
            # cls_pred: [B, NC, H, W]
            # reg_pred: [B, 4*reg_max, H, W]
            stride = self.strides[i]
            b, nc, h, w = cls_pred.shape
            
            # Prepare targets
            target_cls = torch.zeros_like(cls_pred) # [B, NC, H, W]
            target_mask = torch.zeros((b, h, w), device=self.device, dtype=torch.bool)
            target_box = torch.zeros((b, h, w, 4), device=self.device) # [B, H, W, 4] (l,t,r,b) normalized by stride
            
            # Assign targets to grid cells
            for batch_idx, t_list in enumerate(targets):
                if len(t_list) == 0: continue
                
                gt_boxes = t_list[:, 2:6] # x1, y1, x2, y2
                gt_classes = t_list[:, 1].long()
                
                # Convert to grid coords
                gt_boxes_grid = gt_boxes / stride
                gt_centers = (gt_boxes_grid[:, :2] + gt_boxes_grid[:, 2:]) / 2
                
                # Find grid indices
                gx = gt_centers[:, 0]
                gy = gt_centers[:, 1]
                gx_i = gx.long().clamp(0, w-1)
                gy_i = gy.long().clamp(0, h-1)
                
                # Enhanced positive assignment: center + 8 neighbors (3x3 grid)
                # This provides more training signal for small datasets
                offsets = torch.tensor([
                    [0, 0],   # center
                    [1, 0], [-1, 0], [0, 1], [0, -1],  # 4 neighbors
                    [1, 1], [1, -1], [-1, 1], [-1, -1]  # 4 diagonal
                ], device=self.device)
                for idx in range(len(gt_boxes_grid)):
                    base_x = gx_i[idx].item()
                    base_y = gy_i[idx].item()
                    cls_id = gt_classes[idx].item()
                    for off in offsets:
                        cx = int(base_x + int(off[0]))
                        cy = int(base_y + int(off[1]))
                        if 0 <= cx < w and 0 <= cy < h:
                            center_xy = torch.tensor([cx + 0.5, cy + 0.5], device=self.device).unsqueeze(0)
                            box_xyxy = gt_boxes_grid[idx:idx+1, :]
                            ltrb = torch.cat([center_xy - box_xyxy[:, :2], box_xyxy[:, 2:] - center_xy], dim=1)
                            if ltrb.max() >= self.reg_max - 1.01:
                                continue
                            target_mask[batch_idx, cy, cx] = True
                            target_cls[batch_idx, cls_id, cy, cx] = 1.0
                            target_box[batch_idx, cy, cx] = ltrb.squeeze(0).clamp(min=0.01, max=self.reg_max - 1.01)
                
            num_pos = target_mask.sum()
            total_samples += num_pos
            
            if num_pos > 0:
                # Classification Loss
                if label_smoothing > 0:
                    target_cls = target_cls * (1 - label_smoothing) + label_smoothing / max(nc, 1)
                cls_loss_val += self._sigmoid_focal_loss(cls_pred, target_cls, alpha=focal_alpha, gamma=focal_gamma)
                
                # Regression Loss
                # Get predictions at positive locations
                # reg_pred: [B, 4*reg_max, H, W] -> permute -> [B, H, W, 4, reg_max]
                pred_dist = reg_pred.permute(0, 2, 3, 1).reshape(b, h, w, 4, self.reg_max)
                pred_dist_pos = pred_dist[target_mask] # [N_pos, 4, reg_max]
                target_box_pos = target_box[target_mask] # [N_pos, 4]
                
                # DFL Loss
                tl = target_box_pos.long() # left integer
                tr = tl + 1 # right integer
                wl = tr.float() - target_box_pos # weight left
                wr = 1 - wl # weight right
                
                loss_dfl = (F.cross_entropy(pred_dist_pos.view(-1, self.reg_max), tl.view(-1), reduction='none').view(-1, 4) * wl +
                           F.cross_entropy(pred_dist_pos.view(-1, self.reg_max), tr.view(-1), reduction='none').view(-1, 4) * wr).mean()
                
                dfl_loss_val += loss_dfl * num_pos
                
                # Box Loss (Smooth L1)
                pred_dist_pos_soft = pred_dist_pos.softmax(dim=-1)
                pred_val = (pred_dist_pos_soft * torch.arange(self.reg_max, device=self.device).float()).sum(dim=-1) # [N_pos, 4]
                
                box_loss_val += F.smooth_l1_loss(pred_val, target_box_pos, reduction='sum')

        # Normalize losses
        if total_samples == 0: total_samples = 1
        
        return (cls_loss_val * loss_cfg['cls_loss_weight'] + 
                box_loss_val * loss_cfg['box_loss_weight'] + 
                dfl_loss_val * loss_cfg['dfl_loss_weight']) / total_samples

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}', dynamic_ncols=True, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for batch in pbar:
            images = batch['images'].to(self.device)
            targets = [t.to(self.device) for t in batch['targets']]
            with autocast(enabled=self.use_amp):
                outputs = self.model(images, return_raw=True)
                loss = self._compute_loss(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate_metrics(self, epoch: int = 0):
        self.model.eval()
        ap, precision, recall = evaluate_map(
            self.model,
            self.val_loader,
            self.device,
            conf_threshold=self.eval_conf,
            iou_threshold=self.eval_iou,
            num_classes=self.config['model']['num_classes']
        )
        self.logger.info(
            f"Epoch {epoch}: mAP@0.5={ap:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, "
            f"conf={self.eval_conf}, iou={self.eval_iou}"
        )
        return ap, precision, recall

    def train(self):
        self.logger.info("Starting training...")
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.model.train()
            # 使用 dynamic_ncols 自动调整宽度，leave=False 跑完不保留进度条，保持界面整洁
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', dynamic_ncols=True, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            total_loss = 0
            self.optimizer.zero_grad(set_to_none=True)
            
            for step_idx, batch in enumerate(pbar):
                images = batch['images'].to(self.device)
                targets = [t.to(self.device) for t in batch['targets']]

                self._apply_warmup(epoch, step_idx, len(self.train_loader))
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images) # Returns list of tuples
                    loss = self._compute_loss(outputs, targets)
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                should_step = (step_idx + 1) % self.accumulation_steps == 0 or (step_idx + 1) == len(self.train_loader)
                if should_step:
                    if self.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                
                total_loss += loss.item() * self.accumulation_steps
                pbar.set_postfix({'loss': f'{loss.item() * self.accumulation_steps:.4f}'})
            
            avg_loss = total_loss / len(self.train_loader)
            val_loss = self.validate(epoch)
            self.writer.add_scalar('loss/train', avg_loss, epoch)
            self.writer.add_scalar('loss/val', val_loss, epoch)
            self.logger.info(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}")
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self._get_checkpoint(epoch), 
                           os.path.join(self.output_dir, 'weights', 'best_loss.pt'))
                self.logger.info(f"New best_loss model saved with loss: {val_loss:.4f}")

            should_eval_metrics = epoch == 1 or epoch % self.eval_period == 0 or epoch == self.config['training']['epochs']
            if should_eval_metrics:
                ap, precision, recall = self.validate_metrics(epoch)
                self.writer.add_scalar('metrics/mAP50', ap, epoch)
                self.writer.add_scalar('metrics/precision', precision, epoch)
                self.writer.add_scalar('metrics/recall', recall, epoch)
                if ap > self.best_map or (ap == self.best_map and val_loss < self.best_loss):
                    self.best_map = ap
                    torch.save(self._get_checkpoint(epoch),
                               os.path.join(self.output_dir, 'weights', 'best.pt'))
                    self.logger.info(f"New best model saved with mAP@0.5: {ap:.4f}")
            self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config['output']['save_period'] == 0:
                torch.save(self._get_checkpoint(epoch), 
                          os.path.join(self.output_dir, 'weights', f'epoch_{epoch}.pt'))
        
        torch.save(self._get_checkpoint(self.config['training']['epochs']),
                   os.path.join(self.output_dir, 'weights', 'last.pt'))
        self.logger.info("Training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='path to config file')
    parser.add_argument('--validate', action='store_true', help='only run validation using best weights')
    parser.add_argument('--weights', type=str, default=os.path.join('outputs', 'exp_anchor_loss', 'weights', 'best.pt'),
                        help='path to weights for validation')
    args = parser.parse_args()

    trainer = Trainer(args.config)
    if args.validate:
        # 加载最优权重并验证
        weights_path = trainer._resolve_path(args.weights)
        if os.path.exists(weights_path):
            ckpt = trainer._safe_load_checkpoint(weights_path)
            state_dict = ckpt.get('model', ckpt)
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            model_dict = trainer.model.state_dict()
            compatible_state = {
                key: value for key, value in state_dict.items()
                if key in model_dict and model_dict[key].shape == value.shape
            }
            model_dict.update(compatible_state)
            trainer.model.load_state_dict(model_dict, strict=False)
            trainer.logger.info(f"Loaded weights from {weights_path} for validation")
        else:
            trainer.logger.warning(f"Weights file not found at {weights_path}, validating current model state")
        val_loss = trainer.validate(epoch=0)
        trainer.logger.info(f"Validation complete. Avg Val Loss: {val_loss:.4f}")
    else:
        trainer.train()
