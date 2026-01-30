import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.loss import NTXentLoss



def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    # 計算總 epoch 數和進度
    total_epochs = config.num_epoch
    
    for epoch in range(1, config.num_epoch + 1):
        # 計算並顯示 epoch 進度
        epoch_progress = (epoch / total_epochs) * 100
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{total_epochs} ({epoch_progress:.1f}%)")
        print(f"{'='*60}")
        
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode, epoch, total_epochs)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        # 根據訓練模式顯示不同的信息
        if training_mode == "self_supervised":
            # 在 self_supervised 模式下，accuracy 不適用（無監督學習）
            logger.debug(f'\nEpoch : {epoch}\n'
                         f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : N/A (Self-Supervised)\n'
                         f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : N/A (Self-Supervised)')
            
            # 同時打印到控制台
            print(f"Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : N/A (Self-Supervised)")
            print(f"Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : N/A (Self-Supervised)")
        else:
            logger.debug(f'\nEpoch : {epoch}\n'
                         f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                         f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')
            
            # 同時打印到控制台
            print(f"Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}")
            print(f"Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}")

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode, epoch=1, total_epochs=1):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    # 計算總 batch 數
    total_batches = len(train_loader)
    
    # 使用 tqdm 顯示進度條
    pbar = tqdm(enumerate(train_loader), total=total_batches, 
                desc=f"Epoch {epoch}/{total_epochs}", 
                leave=False, 
                ncols=100)

    for batch_idx, (data, labels, aug1, aug2) in pbar:
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1 
            zjs = temp_cont_lstm_feat2 

        else:
            output = model(data)

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
            
        else: # supervised training or fine tuining
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        # 檢查 loss 是否為 NaN 或 Inf
        if torch.isnan(loss) or torch.isinf(loss):
            pbar.write(f"警告：在第 {batch_idx} 個 batch 檢測到 NaN/Inf loss，跳過此 batch")
            continue

        total_loss.append(loss.item())
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(temporal_contr_model.parameters(), max_norm=1.0)
        
        model_optimizer.step()
        temp_cont_optimizer.step()
        
        # 更新進度條顯示當前 loss 和平均 loss
        current_loss = loss.item()
        batch_progress = ((batch_idx + 1) / total_batches) * 100
        avg_loss = sum(total_loss) / len(total_loss) if total_loss else 0.0
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Avg Loss': f'{avg_loss:.4f}',
            'Batch': f'{batch_idx+1}/{total_batches} ({batch_progress:.1f}%)'
        })

    pbar.close()
    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs
