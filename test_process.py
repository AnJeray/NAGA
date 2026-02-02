
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from util.write_file import WriteFile
from util.metrics import calculate_metrics

def test_process(opt, model, test_loader, epoch=None, log_summary_writer=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    y_true, y_pred = [], []
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        test_iter = tqdm(test_loader, desc='Testing', position=0)
        
        for text_features, image_features, labels in test_iter:

            text_features = text_features.to(device)
            image_features = image_features.to(device)
            labels = labels.to(device)
            
            try:

                final_pred = model(image_features, text_features)
                
                loss = nn.CrossEntropyLoss()(final_pred, labels)
                
                _, predicted = torch.max(final_pred.data, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                
                test_iter.set_description(
                    f"Testing - Loss: {loss.item():.4f}"
                )
                
            except Exception as e:
                print(f"Error in testing batch: {e}")
                continue
    

    avg_loss = total_loss / total_samples
    

    test_metrics = calculate_metrics(np.array(y_true), np.array(y_pred))
    test_accuracy, test_F1_weighted, test_R_weighted = test_metrics[:3]
    test_precision_weighted, test_F1, test_R, test_precision = test_metrics[3:]
    

    save_content = (
        f'\nTest Results:\n'
        f'Accuracy: {test_accuracy:.4f}\n'
        f'F1 (weighted): {test_F1_weighted:.4f}\n'
        f'Precision (weighted): {test_precision_weighted:.4f}\n'
        f'Recall (weighted): {test_R_weighted:.4f}\n'
        f'F1 (macro): {test_F1:.4f}\n'
        f'Precision: {test_precision:.4f}\n'
        f'Recall: {test_R:.4f}\n'
        f'Loss: {avg_loss:.4f}\n'
    )
    
    WriteFile(opt.save_model_path+opt.dataset, 'test_log.txt', save_content + '\n', 'a+')
    print(save_content)
    
    # 记录到TensorBoard
    if log_summary_writer and epoch is not None:
        log_summary_writer.add_scalar('test/loss', avg_loss, epoch)
        log_summary_writer.add_scalar('test/accuracy', test_accuracy, epoch)
        log_summary_writer.add_scalar('test/f1_weighted', test_F1_weighted, epoch)
        log_summary_writer.add_scalar('test/recall_weighted', test_R_weighted, epoch)
        log_summary_writer.add_scalar('test/precision_weighted', test_precision_weighted, epoch)
        log_summary_writer.add_scalar('test/f1_macro', test_F1, epoch)
        log_summary_writer.flush()
    
    return {
        'accuracy': test_accuracy,
        'f1_weighted': test_F1_weighted,
        'precision_weighted': test_precision_weighted,
        'recall_weighted': test_R_weighted,
        'f1_macro': test_F1,
        'loss': avg_loss
    }