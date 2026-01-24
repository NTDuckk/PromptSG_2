import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
from collections import defaultdict
import re

def parse_log_file(log_file):
    """Parse training log file to extract metrics"""
    data = {
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
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found!")
        return data
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    current_epoch = None
    current_lr = None
    
    for line in lines:
        # Extract epoch number
        epoch_match = re.search(r'Epoch (\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            if current_epoch not in data['epochs']:
                data['epochs'].append(current_epoch)
        
        # Extract learning rate
        lr_match = re.search(r'Lr: ([\d.e-]+)', line)
        if lr_match:
            current_lr = float(lr_match.group(1))
            if len(data['lr']) < len(data['epochs']):
                data['lr'].append(current_lr)
        
        # Extract loss values
        if 'Loss:' in line and 'ID' in line and 'TRI' in line and 'SupCon' in line:
            # Example: "Epoch[1] Iteration[50/100] Loss: 2.456 (ID 1.234 TRI 0.890 SupCon 0.332)"
            loss_match = re.search(r'Loss: ([\d.]+) \(ID ([\d.]+) TRI ([\d.]+) SupCon ([\d.]+)\)', line)
            if loss_match and current_epoch:
                data['total_loss'].append(float(loss_match.group(1)))
                data['id_loss'].append(float(loss_match.group(2)))
                data['triplet_loss'].append(float(loss_match.group(3)))
                data['supcon_loss'].append(float(loss_match.group(4)))
        
        # Extract validation results
        if 'Validation Results' in line and current_epoch:
            # Look for mAP in next few lines
            for i, next_line in enumerate(lines[lines.index(line)+1:lines.index(line)+5]):
                if 'mAP:' in next_line:
                    map_match = re.search(r'mAP: ([\d.]+)%', next_line)
                    if map_match:
                        data['mAP'].append(float(map_match.group(1)))
                elif 'Rank-1' in next_line:
                    rank1_match = re.search(r'Rank-1:{([\d.]+)%}', next_line)
                    if rank1_match:
                        data['rank1'].append(float(rank1_match.group(1)))
                elif 'Rank-5' in next_line:
                    rank5_match = re.search(r'Rank-5:{([\d.]+)%}', next_line)
                    if rank5_match:
                        data['rank5'].append(float(rank5_match.group(1)))
                elif 'Rank-10' in next_line:
                    rank10_match = re.search(r'Rank-10:{([\d.]+)%}', next_line)
                    if rank10_match:
                        data['rank10'].append(float(rank10_match.group(10)))
    
    return data

def plot_learning_curves(data, save_dir='plots'):
    """Plot learning curves for all metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    # 1. Validation Metrics (mAP, Rank-1, Rank-5, Rank-10)
    if data['epochs'] and data['mAP']:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Validation Metrics', fontsize=16, fontweight='bold')
        
        metrics = ['mAP', 'rank1', 'rank5', 'rank10']
        titles = ['mAP', 'Rank-1 Accuracy', 'Rank-5 Accuracy', 'Rank-10 Accuracy']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx//2, idx%2]
            if data[metric]:
                ax.plot(data['epochs'], data[metric], 
                       color=colors[idx], linewidth=2, marker='o', markersize=4)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.set_title(f'{title} vs Epoch')
                ax.grid(True, alpha=0.3)
                
                # Add best value annotation
                best_idx = np.argmax(data[metric])
                best_val = data[metric][best_idx]
                best_epoch = data['epochs'][best_idx]
                ax.annotate(f'Best: {best_val:.1f}% (Epoch {best_epoch})',
                           xy=(best_epoch, best_val), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'validation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Loss Components
    if data['epochs'] and data['total_loss']:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        loss_metrics = ['total_loss', 'id_loss', 'triplet_loss', 'supcon_loss']
        loss_labels = ['Total Loss', 'ID Loss', 'Triplet Loss', 'SupCon Loss']
        loss_colors = ['red', 'blue', 'green', 'orange']
        
        for metric, label, color in zip(loss_metrics, loss_labels, loss_colors):
            if data[metric]:
                ax.plot(data['epochs'][:len(data[metric])], data[metric], 
                       label=label, color=color, linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.set_title('Training Loss Components', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_components.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Learning Rate Schedule
    if data['epochs'] and data['lr']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(data['epochs'][:len(data['lr'])], data['lr'], 
               color='purple', linewidth=2, marker='d', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule', fontweight='bold')
        ax.set_yscale('log')  # Use log scale for LR
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Combined Overview
    if data['epochs'] and data['mAP'] and data['total_loss']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Validation metrics
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(data['epochs'], data['mAP'], 'b-', linewidth=2, marker='o', markersize=4, label='mAP')
        line2 = ax1.plot(data['epochs'], data['rank1'], 'r-', linewidth=2, marker='s', markersize=4, label='Rank-1')
        line3 = ax1_twin.plot(data['epochs'][:len(data['total_loss'])], data['total_loss'], 
                             'g--', linewidth=2, marker='^', markersize=4, label='Total Loss')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)', color='b')
        ax1_twin.set_ylabel('Loss', color='g')
        ax1.set_title('Training Progress Overview', fontweight='bold')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        ax1.grid(True, alpha=0.3)
        
        # Right: Learning rate
        ax2.plot(data['epochs'][:len(data['lr'])], data['lr'], 
                color='purple', linewidth=2, marker='d', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule', fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
        plt.show()

def create_summary_table(data, save_dir='plots'):
    """Create a summary table of key metrics"""
    if not data['epochs']:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    headers = ['Epoch', 'mAP (%)', 'Rank-1 (%)', 'Rank-5 (%)', 'Total Loss', 'LR']
    
    for i, epoch in enumerate(data['epochs']):
        row = [epoch]
        
        # Add metrics if available
        if i < len(data['mAP']):
            row.append(f"{data['mAP'][i]:.2f}")
        else:
            row.append("N/A")
            
        if i < len(data['rank1']):
            row.append(f"{data['rank1'][i]:.2f}")
        else:
            row.append("N/A")
            
        if i < len(data['rank5']):
            row.append(f"{data['rank5'][i]:.2f}")
        else:
            row.append("N/A")
            
        if i < len(data['total_loss']):
            row.append(f"{data['total_loss'][i]:.4f}")
        else:
            row.append("N/A")
            
        if i < len(data['lr']):
            row.append(f"{data['lr'][i]:.2e}")
        else:
            row.append("N/A")
        
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Training Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    plt.show()

def print_training_log(data):
    """Print training metrics to console in a formatted way"""
    if not data['epochs']:
        print("No training data to display!")
        return
    
    print("\n" + "="*80)
    print("TRAINING LOG SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Epoch':<6} {'mAP(%)':<8} {'Rank1(%)':<10} {'Rank5(%)':<10} {'Loss':<10} {'LR':<12}")
    print("-" * 70)
    
    # Data rows
    for i, epoch in enumerate(data['epochs']):
        map_val = f"{data['mAP'][i]:.2f}" if i < len(data['mAP']) else "N/A"
        rank1_val = f"{data['rank1'][i]:.2f}" if i < len(data['rank1']) else "N/A"
        rank5_val = f"{data['rank5'][i]:.2f}" if i < len(data['rank5']) else "N/A"
        loss_val = f"{data['total_loss'][i]:.4f}" if i < len(data['total_loss']) else "N/A"
        lr_val = f"{data['lr'][i]:.2e}" if i < len(data['lr']) else "N/A"
        
        print(f"{epoch:<6} {map_val:<8} {rank1_val:<10} {rank5_val:<10} {loss_val:<10} {lr_val:<12}")
    
    print("-" * 70)
    
    # Best metrics
    if data['mAP']:
        best_map_idx = np.argmax(data['mAP'])
        best_map = data['mAP'][best_map_idx]
        best_map_epoch = data['epochs'][best_map_idx]
        print(f"\nBest mAP: {best_map:.2f}% at Epoch {best_map_epoch}")
    
    if data['rank1']:
        best_rank1_idx = np.argmax(data['rank1'])
        best_rank1 = data['rank1'][best_rank1_idx]
        best_rank1_epoch = data['epochs'][best_rank1_idx]
        print(f"Best Rank-1: {best_rank1:.2f}% at Epoch {best_rank1_epoch}")
    
    print("="*80 + "\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate learning curves from training logs')
    parser.add_argument('--log_file', type=str, help='Path to specific log file')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory containing log files')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--save_json', action='store_true', help='Save parsed metrics to JSON file')
    
    args = parser.parse_args()
    
    # Ensure log directory exists
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Determine which log file to use
    if args.log_file:
        if not os.path.exists(args.log_file):
            print(f"Specified log file {args.log_file} not found!")
            return
        log_file = args.log_file
    else:
        # Find the latest log file in log_dir
        log_files = []
        for root, dirs, files in os.walk(args.log_dir):
            for file in files:
                if file.endswith('.log'):
                    log_files.append(os.path.join(root, file))
        
        if not log_files:
            print("No log files found!")
            return
        
        log_file = max(log_files, key=os.path.getctime)
    
    print(f"Processing log file: {log_file}")
    
    # Parse data and create plots
    data = parse_log_file(log_file)
    
    if data['epochs']:
        print(f"Found data for {len(data['epochs'])} epochs")
        print(f"Metrics found: mAP={len(data['mAP'])}, rank1={len(data['rank1'])}, loss={len(data['total_loss'])}")
        
        # Print training log to console
        print_training_log(data)
        
        # Save metrics to JSON if requested
        if args.save_json:
            json_file = os.path.join(args.log_dir, 'training_metrics.json')
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Metrics saved to: {json_file}")
        
        # Generate plots
        plot_learning_curves(data, args.output_dir)
        create_summary_table(data, args.output_dir)
        print(f"Plots saved to '{args.output_dir}' directory")
    else:
        print("No training data found in log file!")

if __name__ == "__main__":
    main()
