
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange
from util.write_file import WriteFile
from util.metrics import calculate_metrics
import dev_process
import test_process
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import os

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        import math
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_process(opt, train_loader, dev_loader, test_loader, model, log_summary_writer=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    optimizer_params = [p for n, p in model.named_parameters() if p.requires_grad]

    optimizer = AdamW(
        optimizer_params,
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )
    
    total_steps = len(train_loader) * opt.epoch
    warmup_steps = int(total_steps * opt.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    

    best_accuracy = 0.0
    best_model_state = None
    patience = opt.patience
    patience_counter = 0
    early_stopped = False

    for epoch in trange(opt.epoch, desc='Epoch'):
        model.train()
        y_true, y_pred = [], []
        total_loss = 0.0
        total_samples = 0
        
        # training
        for batch_idx, (text_features, image_features, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}', position=0)):
            
            text_features = text_features.to(device)
            image_features = image_features.to(device)
            labels = labels.to(device)
            final_pred, loss = model(image_features, text_features, labels)
            loss = loss / opt.acc_grad
            loss.backward()
            
            if (batch_idx + 1) % opt.acc_grad == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            _, predicted = torch.max(final_pred.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            total_loss += loss.item() * opt.acc_grad * labels.size(0)
            total_samples += labels.size(0)

        epoch_loss = total_loss / total_samples
        train_metrics = calculate_metrics(y_true, y_pred)
        train_accuracy, train_F1_weighted, train_R_weighted = train_metrics[:3]
        train_precision_weighted, train_F1, train_R, train_precision = train_metrics[3:]

        save_content = (
            f'Epoch {epoch} Results:\n'
            f'Train Accuracy: {train_accuracy:.4f}\n'
            f'Train F1 (weighted): {train_F1_weighted:.4f}\n'
            f'Train Precision (weighted): {train_precision_weighted:.4f}\n'
            f'Train Recall (weighted): {train_R_weighted:.4f}\n'
            f'Train F1 (macro): {train_F1:.4f}\n'
            f'Train Loss: {epoch_loss:.4f}\n'
        )
        
        WriteFile(opt.save_model_path+opt.dataset, 'train_log.txt', save_content + '\n', 'a+')
        print(save_content)


        if log_summary_writer:
            log_summary_writer.add_scalar('train/loss', epoch_loss, epoch)
            log_summary_writer.add_scalar('train/accuracy', train_accuracy, epoch)
            log_summary_writer.add_scalar('train/f1_weighted', train_F1_weighted, epoch)
            log_summary_writer.flush()

        dev_accuracy = dev_process.dev_process(opt, model, dev_loader, log_summary_writer)


        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_model_state = deepcopy(model.state_dict())
            save_path = os.path.join(opt.save_model_path+opt.dataset, 'best_model')
            os.makedirs(save_path, exist_ok=True)
            torch.save(best_model_state, os.path.join(save_path, 'best_model.pth'))
            print(f'New best model saved! Accuracy: {best_accuracy:.4f}')
            patience_counter = 0  
        else:
            patience_counter += 1
            print(f'Validation accuracy did not improve. Patience: {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                early_stopped = True
                break
        
        test_process.test_process(opt, model, test_loader, epoch)


    if early_stopped:
        print(f"\nTraining stopped early after {epoch + 1} epochs")
    else:
        print(f"\nTraining completed for all {opt.epoch} epochs")
        

    print('\nFinal Evaluation on Test Set (Best Model):')
    best_model_path = os.path.join(opt.save_model_path+opt.dataset, 'best_model', 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        test_process.test_process(opt, model, test_loader, epoch=-1)
    else:
        print(f"Best model not found at {best_model_path}")

    if log_summary_writer:
        log_summary_writer.close()