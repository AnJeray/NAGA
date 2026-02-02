
import torch
from tqdm import tqdm
from util.write_file import WriteFile
from util.metrics import calculate_metrics
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def dev_process(opt, model, dev_loader, log_summary_writer=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (text_features, image_features, labels) in enumerate(tqdm(dev_loader, desc='Validation', position=0)):

            text_features = text_features.to(device)
            image_features = image_features.to(device)
            labels = labels.to(device)

            final_pred = model(image_features, text_features)
            
            loss = nn.CrossEntropyLoss()(final_pred, labels)
            
            _, predicted = torch.max(final_pred.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    
    dev_metrics = calculate_metrics(y_true, y_pred)
    dev_accuracy, dev_F1_weighted, dev_R_weighted = dev_metrics[:3]
    dev_precision_weighted, dev_F1, dev_R, dev_precision = dev_metrics[3:]
    
    save_content = (
        f'Validation Results:\n'
        f'Accuracy: {dev_accuracy:.4f}\n'
        f'F1 (weighted): {dev_F1_weighted:.4f}\n'
        f'Precision (weighted): {dev_precision_weighted:.4f}\n'
        f'Recall (weighted): {dev_R_weighted:.4f}\n'
        f'F1 (macro): {dev_F1:.4f}\n'
        f'Loss: {avg_loss:.4f}\n'
    )
    
    WriteFile(opt.save_model_path+opt.dataset, 'dev_log.txt', save_content + '\n', 'a+')
    print(save_content)

    if log_summary_writer:
        log_summary_writer.add_scalar('validation/loss', avg_loss, global_step=1)
        log_summary_writer.add_scalar('validation/accuracy', dev_accuracy, global_step=1)
        log_summary_writer.add_scalar('validation/f1_weighted', dev_F1_weighted, global_step=1)
        log_summary_writer.add_scalar('validation/recall_weighted', dev_R_weighted, global_step=1)
        log_summary_writer.add_scalar('validation/precision_weighted', dev_precision_weighted, global_step=1)
        log_summary_writer.add_scalar('validation/f1_macro', dev_F1, global_step=1)
        log_summary_writer.flush()

    return dev_accuracy