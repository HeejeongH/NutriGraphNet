"""
Training Utilities
논문의 최적화 기법 구현: Cosine Annealing, Early Stopping
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any


class CosineAnnealingWithWarmRestarts:
    """
    논문의 Cosine Annealing with Warm Restarts 구현
    
    η_t = η_min + 0.5(η_max - η_min)(1 + cos(T_cur/T_i × π))
    """
    
    def __init__(self, optimizer, T_0=10, T_mult=2, eta_min=1e-6, eta_max=0.001, last_epoch=-1):
        """
        Args:
            optimizer: PyTorch optimizer
            T_0: 첫 번째 재시작까지의 에포크 수
            T_mult: 재시작 주기 배율
            eta_min: 최소 학습률
            eta_max: 최대 학습률
            last_epoch: 마지막 에포크 번호
        """
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_cur = last_epoch + 1
        self.T_i = T_0
        self.epoch = last_epoch + 1
        
        # 초기 학습률 설정
        self._set_lr(eta_max)
    
    def _set_lr(self, lr):
        """학습률 설정"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def step(self):
        """한 스텝 진행"""
        self.T_cur += 1
        
        # Restart 체크
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
        
        # Cosine annealing 계산
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1 + np.cos(np.pi * self.T_cur / self.T_i)
        )
        
        self._set_lr(lr)
        self.epoch += 1
        
        return lr
    
    def get_last_lr(self):
        """현재 학습률 반환"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """상태 저장"""
        return {
            'T_0': self.T_0,
            'T_mult': self.T_mult,
            'eta_min': self.eta_min,
            'eta_max': self.eta_max,
            'T_cur': self.T_cur,
            'T_i': self.T_i,
            'epoch': self.epoch
        }
    
    def load_state_dict(self, state_dict):
        """상태 로드"""
        self.T_0 = state_dict['T_0']
        self.T_mult = state_dict['T_mult']
        self.eta_min = state_dict['eta_min']
        self.eta_max = state_dict['eta_max']
        self.T_cur = state_dict['T_cur']
        self.T_i = state_dict['T_i']
        self.epoch = state_dict['epoch']


class EarlyStopping:
    """
    논문의 Early Stopping with Adaptive Patience 구현
    """
    
    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True):
        """
        Args:
            patience: 개선이 없어도 기다릴 에포크 수
            min_delta: 개선으로 간주할 최소 변화량
            mode: 'max' (높을수록 좋음) or 'min' (낮을수록 좋음)
            verbose: 로그 출력 여부
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # Mode에 따른 비교 함수 설정
        if mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = -np.inf
        else:
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = np.inf
    
    def __call__(self, score, epoch, model=None):
        """
        Early stopping 체크
        
        Args:
            score: 현재 점수 (validation metric)
            epoch: 현재 에포크
            model: 저장할 모델 (optional)
            
        Returns:
            bool: True if should stop training
        """
        if self.is_better(score, self.best_score):
            # 개선됨
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
            if self.verbose:
                print(f"  ✓ Validation improved to {score:.4f} at epoch {epoch}")
            
            # 모델 저장 (optional)
            if model is not None:
                self.best_model_state = model.state_dict()
            
            return False
        else:
            # 개선 없음
            self.counter += 1
            
            if self.verbose and self.counter >= self.patience // 2:
                print(f"  ⚠ No improvement for {self.counter} epochs (patience: {self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  🛑 Early stopping triggered at epoch {epoch}")
                    print(f"     Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
            
            return False
    
    def reset(self):
        """상태 초기화"""
        self.counter = 0
        self.early_stop = False
        if self.mode == 'max':
            self.best_score = -np.inf
        else:
            self.best_score = np.inf
        self.best_epoch = 0


class TrainingMonitor:
    """
    훈련 과정 모니터링 및 메트릭 기록
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
    
    def log_epoch(self, epoch, train_loss, val_loss=None, train_metrics=None, 
                  val_metrics=None, lr=None):
        """에포크별 메트릭 기록"""
        self.history['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        
        if train_metrics is not None:
            self.history['train_metrics'].append(train_metrics)
        
        if val_metrics is not None:
            self.history['val_metrics'].append(val_metrics)
        
        if lr is not None:
            self.history['learning_rates'].append(lr)
    
    def get_best_epoch(self, metric='val_loss', mode='min'):
        """최고 성능 에포크 찾기"""
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0
        
        values = self.history[metric]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return best_idx
    
    def print_summary(self):
        """훈련 요약 출력"""
        print("\n" + "="*60)
        print("📊 Training Summary")
        print("="*60)
        
        if len(self.history['train_loss']) > 0:
            print(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}")
            print(f"Best Train Loss:  {min(self.history['train_loss']):.4f}")
        
        if len(self.history['val_loss']) > 0:
            best_val_idx = np.argmin(self.history['val_loss'])
            print(f"Final Val Loss:   {self.history['val_loss'][-1]:.4f}")
            print(f"Best Val Loss:    {self.history['val_loss'][best_val_idx]:.4f} (epoch {best_val_idx+1})")
        
        if len(self.history['learning_rates']) > 0:
            print(f"Final LR:         {self.history['learning_rates'][-1]:.2e}")
        
        print("="*60)


def compute_metrics(predictions, targets, threshold=0.5):
    """
    분류 메트릭 계산
    
    Args:
        predictions: 예측 확률 (0-1)
        targets: 실제 레이블 (0 or 1)
        threshold: 분류 임계값
        
    Returns:
        dict: 계산된 메트릭들
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score
    )
    
    # Numpy로 변환
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    # 이진 예측
    binary_preds = (predictions > threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(targets, binary_preds),
        'precision': precision_score(targets, binary_preds, zero_division=0),
        'recall': recall_score(targets, binary_preds, zero_division=0),
        'f1': f1_score(targets, binary_preds, zero_division=0),
        'auc': roc_auc_score(targets, predictions) if len(np.unique(targets)) > 1 else 0.5
    }
    
    return metrics


class GradientClipper:
    """
    논문의 Gradient Clipping 구현
    """
    
    def __init__(self, max_norm=1.0, norm_type=2):
        """
        Args:
            max_norm: 최대 그래디언트 norm
            norm_type: norm 타입 (2 = L2 norm)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip(self, model):
        """모델 그래디언트 클리핑"""
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=self.max_norm, 
            norm_type=self.norm_type
        )


def get_optimizer_with_scheduler(model, config):
    """
    논문의 최적화 설정 생성
    
    Args:
        model: PyTorch 모델
        config: dict with keys: lr, weight_decay, scheduler_type, etc.
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # AdamW optimizer (논문 권장)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Scheduler 설정
    scheduler_type = config.get('scheduler_type', 'cosine_warmup')
    
    if scheduler_type == 'cosine_warmup':
        scheduler = CosineAnnealingWithWarmRestarts(
            optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2),
            eta_min=config.get('eta_min', 1e-6),
            eta_max=config.get('lr', 0.001)
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 50),
            eta_min=config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 10),
            gamma=config.get('gamma', 0.5)
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """
    체크포인트 저장
    
    Args:
        model: PyTorch 모델
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: 현재 에포크
        metrics: 메트릭 dict
        filepath: 저장 경로
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath, device='cpu'):
    """
    체크포인트 로드
    
    Args:
        model: PyTorch 모델
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        filepath: 체크포인트 파일 경로
        device: 디바이스
        
    Returns:
        dict: {'epoch', 'metrics'}
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✅ Checkpoint loaded from {filepath}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Metrics: {checkpoint.get('metrics', {})}")
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {})
    }
