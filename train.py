import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
import numpy as np
from utils import cal_performance, normalize_duration, cal_performance_focal, temporal_cluster_loss, temporal_contrastive_loss, generate_prompt


def get_warmup_factor(epoch, start_epoch, peak_epoch, end_epoch):
    """
    Dynamic warmup factor:
        - Increases linearly from `start_epoch` to `peak_epoch`.
        - Decreases linearly from `peak_epoch` to `end_epoch`.

    Args:
        epoch (int): Current epoch.
        start_epoch (int): When to start increasing the factor.
        peak_epoch (int): When the factor reaches 1.0.
        end_epoch (int): When the factor decreases back to 0.0.

    Returns:
        float: Warmup factor (0.0 to 1.0).
    """
    if epoch < start_epoch:
        return 0.0
    elif epoch < peak_epoch:
        return (epoch - start_epoch) / (peak_epoch - start_epoch)
    elif epoch < end_epoch:
        return 1.0 - (epoch - peak_epoch) / (end_epoch - peak_epoch)
    else:
        return 0.0

def get_cluster_intervals(gt):
    """
    Get cluster intervals where consecutive labels are the same.

    Args:
        gt (torch.Tensor): Ground truth labels of shape [B, T]
                          where B is batch size and T is time steps.

    Returns:
        List[List[Tuple]]: Cluster intervals for each batch.
                           Each batch contains a list of (start, end) tuples.
    """
    B, T = gt.shape
    cluster_intervals = []

    for b in range(B):  # Loop through batches
        intervals = []
        start = 0
        current_label = gt[b, 0].item()

        for t in range(1, T):
            if gt[b, t].item() != current_label:  # Label changes
                intervals.append((start, t - 1))  # Save previous interval
                start = t  # Update start
                current_label = gt[b, t].item()

        # Append the final interval
        intervals.append((start, T - 1))
        cluster_intervals.append(intervals)

    return cluster_intervals


def weighted_accuracy(pred, gold, pad_idx, t_n_labels, weight_same=1.0, weight_different=10.0):
    '''Calculate weighted accuracy based on comparison between t+n and t+m labels.'''
    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(pad_idx)

    total_weighted_correct = 0
    total_weighted_labels = 0

    weight = weight_different if gold[0] != t_n_labels[0] else weight_same

    for i in range(len(gold)):
        if non_pad_mask[i].any():
            if pred[i] == gold[i]:
                total_weighted_correct += weight
            total_weighted_labels += weight

    weighted_accuracy = total_weighted_correct / total_weighted_labels if total_weighted_labels > 0 else 0
    return weighted_accuracy

def get_last_non_padding_labels(past_label, pad_value):
        """
        For each sequence in `past_label`, find the last non-padding label.
        
        Args:
            past_label (torch.Tensor): Tensor of shape [batch_size, frame_count] with padding.
            pad_value (int): Value used for padding.

        Returns:
            torch.Tensor: Tensor of shape [batch_size] containing the last non-padding label for each sequence.
        """
        batch_size = past_label.size(0)
        t_n_last_labels = torch.zeros(batch_size, dtype=past_label.dtype, device=past_label.device)
        
        for i in range(batch_size):
            # Get the sequence for the batch item, reverse it, and find the first non-padding value
            non_pad_indices = (past_label[i] != pad_value).nonzero(as_tuple=True)[0]
            if non_pad_indices.numel() > 0:
                t_n_last_labels[i] = past_label[i, non_pad_indices[-1]]
            else:
                t_n_last_labels[i] = pad_value  # Fallback if entire sequence is padding

        return t_n_last_labels

def validate(model, val_loader, criterion, pad_idx, device):
    model.eval()
    val_loss = 0
    val_class_correct = 0
    val_class_total = 0
    val_seg_correct = 0
    val_seg_total = 0
    total_l3 = 0
    total_l3_correct = 0
    val_weighted_accuracy_total = 0
    with torch.no_grad():
        for data in val_loader:
            features, past_label, trans_dur_future, trans_future_target, query_label = data
            features = features.to(device)
            past_label = past_label.to(device)
            query_label = query_label.to(device)
            trans_dur_future = trans_dur_future.to(device)
            trans_future_target = trans_future_target.to(device)
            trans_dur_future_mask = (trans_dur_future != pad_idx).long().to(device)

            outputs = model((features, past_label), query_label)
            losses = 0

            output_l3 = outputs['l3']
            B, T, C = output_l3.size()
            output_l3 = output_l3.view(-1, C).to(device)
            query_label = query_label.view(-1)
            loss_l3, n_l3_correct, n_l3_total, _ = cal_performance_focal(output_l3, query_label, 48, 48, reference=None, target_ref=None)
            losses += loss_l3
            total_l3 += n_l3_total
            total_l3_correct += n_l3_correct
            output_seg = outputs['seg']
            loss_seg, n_seg_correct, n_seg_total, _ = cal_performance(output_seg.view(-1, output_seg.size(-1)),
                                                                    past_label.view(-1), pad_idx, exclude_class_idx=None, reference=None, target_ref=None)
            losses += loss_seg
            val_seg_correct += n_seg_correct
            val_seg_total += n_seg_total

            output = outputs['action']
            loss_class, n_class_correct, n_class_total, _ = cal_performance(output.view(-1, output.size(-1)),
                                                                            trans_future_target.view(-1), pad_idx, exclude_class_idx=None, reference=None, target_ref=trans_future_target[:,0])
            losses += loss_class
            val_class_correct += n_class_correct
            val_class_total += n_class_total

            # Calculate weighted accuracy using specific t+n and t+m labels
            val_weighted_acc = weighted_accuracy(
                output.view(-1, output.size(-1)), trans_future_target.view(-1), pad_idx, get_last_non_padding_labels(past_label, pad_idx)
            )
            val_weighted_accuracy_total += val_weighted_acc


            val_loss += losses.item()

    val_loss /= len(val_loader)
    l3_accuracy = total_l3_correct / total_l3 if total_l3 else 0
    val_accuracy = val_class_correct / val_class_total if val_class_total else 0
    val_seg_accuracy = val_seg_correct / val_seg_total if val_seg_total else 0
    val_weighted_accuracy = val_weighted_accuracy_total / len(val_loader)
    print(f"Validation Loss: {val_loss:.3f}, l3 accuracy: {l3_accuracy}, Class Accuracy: {val_accuracy:.3f}, Segmentation Accuracy: {val_seg_accuracy:.3f}, Weighted Accuracy: {val_weighted_accuracy:.3f}")
    return val_loss, val_accuracy, val_weighted_accuracy

def train(args, model, train_loader, optimizer, scheduler, criterion, model_save_path, pad_idx, device, val_loader, seed):
    model.to(device)
    model.train()
    print("Training Start")
    best_val_loss = float('inf')
    best_val_acc = 0
    best_weight_acc = 0
    for epoch in range(args.epochs):
        epoch_acc =0
        epoch_loss = 0
        epoch_loss_class = 0
        epoch_loss_dur = 0
        epoch_loss_seg = 0
        total_class = 0
        total_class_correct = 0
        total_seg = 0
        total_seg_correct = 0
        total_l3 = 0
        total_l3_correct = 0
        epoch_loss_l3 = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            features, past_label, trans_dur_future, trans_future_target, query_label = data
            features = features.to(device) #[B, S, C] 16, 480, 2048
            past_label = past_label.to(device) #[B, S] 16, 480
            query_label = query_label.to(device) #[16, 480] -> [16, 8]
            trans_dur_future = trans_dur_future.to(device) #[16, 8]
            trans_future_target = trans_future_target.to(device) #[16, 8]
            trans_dur_future_mask = (trans_dur_future != pad_idx).long().to(device)
            #image_path = image_path.to(device)

            B = trans_dur_future.size(0)
            target_dur = trans_dur_future*trans_dur_future_mask
            target = trans_future_target
            if args.input_type == 'i3d_transcript':
                inputs = (features, past_label)
            elif args.input_type == 'gt':
                gt_features = past_label.int()
                inputs = (gt_features, past_label)
            #print("----------------------------------------")
            if len(inputs[0]) < 8:
                continue            
            
            outputs = model(inputs, query_label)

            losses = 0
            target_past_label = past_label.view(-1) # (8, 34 )->
            first_target_past_label = get_last_non_padding_labels(past_label, pad_idx)

            ### ADDING ###
            
            ######################## Hard label loss #####################################################
            output_l3 = outputs['l3']
            B, T, C = output_l3.size()
            
            loss_supcon = temporal_cluster_loss(output_l3, get_cluster_intervals(query_label))
            output_l3 = output_l3.view(-1, C).to(device) #######
            query_label = query_label.view(-1)
            
            loss_l3, n_l3_correct, n_l3_total, l3_correct = cal_performance_focal(output_l3, query_label, 47, 48, reference=None, target_ref=None)
            
            total_l3 += n_l3_total
            total_l3_correct += n_l3_correct

            if args.seg :
                output_seg = outputs['seg']
                B, T, C = output_seg.size()
                output_seg = output_seg.view(-1, C).to(device)
                #target_past_label = past_label.view(-1)
                loss_seg, n_seg_correct, n_seg_total, l2_correct = cal_performance(output_seg, target_past_label, pad_idx, exclude_class_idx=None, reference=None, target_ref=None)
                #losses += loss_seg
                total_seg += n_seg_total
                total_seg_correct += n_seg_correct
                epoch_loss_seg += loss_seg.item()
            if args.anticipate :
                output = outputs['action']
                B, T, C = output.size()
                output = output.view(-1, C).to(device)
                target_first = target[:, 0]
                target = target.contiguous().view(-1)
                out = output.max(1)[1] #oneshot
                out = out.view(B, -1)
                loss, n_correct, n_total, _ = cal_performance(output, target, pad_idx, exclude_class_idx=None, reference=first_target_past_label, target_ref=target_first)
                acc = n_correct / n_total
                loss_class = loss.item()
                #losses += loss
                total_class += n_total
                total_class_correct += n_correct
                epoch_loss_class += loss_class

                output_dur = outputs['duration']
                output_dur = normalize_duration(output_dur, trans_dur_future_mask)
                target_dur = target_dur * trans_dur_future_mask
                loss_dur = torch.sum(criterion(output_dur, target_dur)) / \
                torch.sum(trans_dur_future_mask)

                #losses += loss_dur
                epoch_loss_dur += loss_dur.item()

            how_much_wrong = torch.where(l3_correct & l2_correct, torch.tensor(1.0, device=device), torch.tensor(5.0, device=device))

            ###################### L3 LOSS ###################################################################3
            warmup_factor = get_warmup_factor(epoch, start_epoch=0, peak_epoch=30, end_epoch=60)
            losses = (1 - 1/how_much_wrong.mean()) * ((1-warmup_factor) * loss_l3 + (warmup_factor) * loss_supcon) + (1/how_much_wrong.mean()) * (loss + loss_dur + loss_seg)
            
            epoch_loss_l3 += loss_l3.item()
            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()


        epoch_loss = epoch_loss / (i+1)
        print("Epoch [", (epoch+1), '/', args.epochs, '] Loss : %.3f'%epoch_loss)
        if args.anticipate :
            accuracy = total_class_correct/total_class
            epoch_loss_class = epoch_loss_class / (i+1)
            print('Training Acc :%.3f'%accuracy, 'CE loss :%.3f'%epoch_loss_class )
            if args.task == 'long' :
                epoch_loss_dur = epoch_loss_dur / (i+1)
                print('dur loss: %.5f'%epoch_loss_dur)

        if args.seg :
            acc_seg = total_seg_correct / total_seg
            epoch_loss_seg = epoch_loss_seg / (i+1)
            print('seg loss :%.3f'%epoch_loss_seg, ', seg acc : %.5f'%acc_seg)

        ####################### ADDING ################################
        acc_l3 = total_l3_correct / total_l3
        epoch_loss_l3 = epoch_loss_l3 / (i+1)
        print('l3 loss :%.3f'%epoch_loss_l3, ', l3 acc : %.5f'%acc_l3)
        ###############################################################

        scheduler.step()

        val_loss, val_acc, weight_acc = validate(model, val_loader, criterion, pad_idx, device)

        if val_acc > best_val_acc or weight_acc > best_weight_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_weight_acc = weight_acc
            save_path = os.path.join(model_save_path)
            save_file = os.path.join(save_path, 'seed_'+str(seed)+'_checkpoint'+str(epoch)+'.ckpt')
            torch.save(model.state_dict(), save_file)
            
            best_save_file = os.path.join(save_path, 'seed_'+str(seed)+'_best.ckpt')
            if os.path.exists(best_save_file):
                os.remove(best_save_file)
            torch.save(model.state_dict(), best_save_file)
            print(f"Best model saved with validation loss: {best_val_loss:.3f}")

    return model
