import json
import os
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, 'training_metrics.json')
        self.metrics = {
            'epochs': [],
            'mAP': [],
            'rank1': [],
            'rank5': [],
            'rank10': [],
            'total_loss': [],
            'id_loss': [],
            'triplet_loss': [],
            'supcon_loss': [],
            'lr': []
        }
        
    def log_epoch(self, epoch, lr, total_loss, id_loss, tri_loss, supcon_loss):
        """Log training metrics for an epoch"""
        self.metrics['epochs'].append(epoch)
        self.metrics['lr'].append(lr)
        self.metrics['total_loss'].append(total_loss)
        self.metrics['id_loss'].append(id_loss)
        self.metrics['triplet_loss'].append(tri_loss)
        self.metrics['supcon_loss'].append(supcon_loss)
        
    def log_validation(self, mAP, rank1, rank5, rank10):
        """Log validation metrics"""
        self.metrics['mAP'].append(mAP)
        self.metrics['rank1'].append(rank1)
        self.metrics['rank5'].append(rank5)
        self.metrics['rank10'].append(rank10)
        
    def save_metrics(self):
        """Save metrics to JSON file"""
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def load_metrics(self):
        """Load metrics from JSON file"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        return self.metrics
