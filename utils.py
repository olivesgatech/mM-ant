'''
utility functions
'''
import numpy as np
import torch
import torch.nn as nn
import os
import pdb
import torch.nn.functional as F
from sklearn.manifold import TSNE
import pandas as pd

COARSE_LABEL_LIST = [
    "Bake_pancake", "Cleaning_Countertops", "Cleaning_Floor", "Get_ingredients", "Having_a_meal",
    "Mix_ingredients", "Prep_ingredients", "Prepare_Kitchen_appliance", "Scroll_on_tablet", "Setting_a_table",
    "Take_out_Kitchen_and_cooking_tools", "Take_out_smartphone", "Throw_out_leftovers",
    "Using_Smartphone", "Using_Tablet", "Washing_and_Drying_dishes_with_hands", "UNDEFINED", "UNDEFINED", "UNDEFINED", "UNDEFINED"
]

FINE_GRAINED_LABEL_LIST = [
    "Add_batter", "Add_coffee", "Add_flour", "Add_milk", "Add_sugar", "Add_water", "Check_cabinet",
    "Check_pancake", "Check_refrigerator", "Clean_with_broom", "Clean_with_mop", "Clean_with_paper_towel",
    "Clean_with_towel", "Conversation_on_the_phone", "Crack_egg", "Drink", "Dry_dishes", "Eat",
    "Fill_coffee_machine_with_water", "Fill_kettle_with_water", "Get_coffee", "Get_cup", "Get_filter",
    "Get_instant_coffee_", "Get_pan", "Get_spoon", "Load_dishwasher", "Place_cup", "Place_dishes",
    "Place_drink", "Place_filter", "Place_food", "Place_pan", "Place_silverware", "Prepare_for_activity",
    "Rinse_dishes", "Scroll_on_the_phone", "Scroll_on_the_tablet", "Stir_", "Stir_pancake_ingredients",
    "Take_out_Kitchen_and_cooking_tools", "Take_out_pancake_ingredients", "Turn_on_coffee_machine",
    "Turn_on_dishwasher", "Turn_on_kettle", "Turn_on_stove", "Unloading_dishwasher", "UNDEFINED", "Take_out_smartphone", "Throw_out_leftovers"
]

coarse_to_fine_mapping = {
    "UNDEFINED": [
        "UNDEFINED"
    ],
    "Prepare_Kitchen_appliance": [
        "Get_filter",
        "Place_filter",
        "Fill_coffee_machine_with_water",
        "Add_coffee",
        "Place_cup",
        "Turn_on_coffee_machine",
        "Fill_kettle_with_water",
        "Turn_on_kettle",
        "Load_dishwasher",
        "Turn_on_dishwasher",
        "Unloading_dishwasher",
        "Turn_on_stove",
        #"UNDEFINED"
    ],
    "Take_out_Kitchen_and_cooking_tools": [
        "Get_cup",
        "Get_spoon",
        "Take_out_pancake_ingredients",
        #"UNDEFINED"
    ],
    "Prep_ingredients": [
        "Get_coffee",
        "Get_instant_coffee_",
        "Check_refrigerator",
        "Check_cabinet",
        #"UNDEFINED"
    ],
    "Mix_ingredients": [
        "Add_water",
        "Add_coffee",
        "Stir_",
        "Add_sugar",
        "Add_flour",
        "Stir_pancake_ingredients",
        "Add_milk",
        "Crack_egg",
        #"UNDEFINED"
    ],
    "Using_Smartphone": [
        "Scroll_on_the_phone",
        "Conversation_on_the_phone",
        #"UNDEFINED"
    ],
    "Take_out_smartphone": [
        "Take_out_smartphone"
    ],
    "Throw_out_leftovers": [
        "Throw_out_leftovers"
    ],
    "Washing_and_Drying_dishes_with_hands": [
        "Place_dishes",
        "Rinse_dishes",
        "Dry_dishes",
        #"UNDEFINED"
    ],
    "Get_ingredients": [
        "Take_out_pancake_ingredients",
        "Check_refrigerator",
        "Check_cabinet",
        #"UNDEFINED"
    ],
    "Bake_pancake": [
        "Get_pan",
        "Place_pan",
        "Check_pancake",
        "Add_batter",
        #"UNDEFINED"
    ],
    "Cleaning_Countertops": [
        "Clean_with_towel",
        "Clean_with_paper_towel",
        #"UNDEFINED"
    ],
    "Cleaning_Floor": [
        "Clean_with_mop",
        "Clean_with_broom",
        "Clean_with_towel",
        #"UNDEFINED"
    ],
    "Setting_a_table": [
        "Add_water",
        "Place_drink",
        "Place_food",
        "Place_silverware",
        "Take_out_Kitchen_and_cooking_tools",
        #"UNDEFINED"
    ],
    "Having_a_meal": [
        "Prepare_for_activity",
        "Eat",
        "Drink",
        #"UNDEFINED"
    ],
    "Using_Tablet": [
        "Scroll_on_the_tablet",
        #"UNDEFINED"
    ],
    "Scroll_on_tablet": [
        "Scroll_on_the_tablet"
    ]
}


def convert_to_coarse_labels(batch_tensor):
    """
    Converts a batch of coarse-level label indices to their corresponding string labels.

    Args:
        batch_tensor (torch.Tensor): A tensor of shape (batch, t) containing label indices.
        coarse_label_list (list of str): A list of coarse-level labels corresponding to indices.

    Returns:
        list of list of str: A list of lists where each inner list contains the string labels for a batch.
    """
    labels = [COARSE_LABEL_LIST[idx] for idx in batch_tensor.tolist()]
    return labels

def generate_prompt(batch_tensor, t):
    """
    Generates a prompt for GPT-2 based on the coarse-level labels and the fine-grained label list.

    Args:
        coarse_labels (list of str): A list of t coarse-level labels corresponding to t images.
        fine_grained_label_list (list of str): A list of 48 fine-grained labels.

    Returns:
        str: The generated prompt for GPT-2.
    """
    
    batch_size = batch_tensor.size(0)
    prompts = [''] * batch_size

    for i in range(batch_size):
        coarse_labels = convert_to_coarse_labels(batch_tensor[i])
        label_candidates = []
        coarse_label_processed = []
        for coarse_label in coarse_labels:
            if coarse_label in coarse_to_fine_mapping and coarse_label not in coarse_label_processed:
                fine_labels = coarse_to_fine_mapping[coarse_label]
                label_candidates.append(f"{coarse_label}: {', '.join(fine_labels)}")
                coarse_label_processed.append(coarse_label)
        label_mapping = "\n".join([f"{i} {label}" for i, label in enumerate(FINE_GRAINED_LABEL_LIST)])

        label_candidates = "\n".join(label_candidates)

        prompt = (
            f"You are given {t} time-series of images that are arranged in chronological order."
            "It consists of the series of informative images and the series of black images."
            "These images capture a sequence of actions that unfold over time."
            "Each image has the corresponding coarse-level labels as follows: "
            f"{', '.join(coarse_labels)}. "
            "Your task is to predict the corresponding fine-grained labels for each image based only on what you observe in the images. "
            "To do this, let's think step by step.\n"
            "First, describe the images that you can see.\n"
            "Second, deduce the fine-grained label from the above candidates using your observation and your own reasoning."

            f"Please predict the {t} number of fine-grained labels for each image and provide the answer in the following format:\n\n"
            "Answer: <ONLY numbers separated by commas>\n\n"
            "For example: Answer: 39, 39, 39, 2, 2\n\n"
            
            "You must not rely on general knowledge or assumptions. "
            "Choose the fine-grained labels based solely on observable details in the images, such as objects, tools, ingredients, and actions.\n\n"
            "For each coarse-level label, you should choose the fine-grained label from the following candidates:\n\n"
            f"{label_candidates}\n\n"
            "Even if the visual information seems incomplete or unclear, you must select a fine-grained label from the above candidates. Do not default to the UNDEFINED label."
            "Here is the label mapping strategy:\n"
            f"{label_mapping}\n\n"
            f"When you give the answer, don't abbreviate anything."
            "For example, you cannot say like: Answer: 47, 47, 47, ..., (153 times). Give us exactly {t} number of fine-grained labels."
            "Also, you should always choose fine-grained labels for each images from all this information. You should not anticipate me giving you further information or clarification with this."
            "Provide the reasoning behind your predictions, specifically describing the visible details in the images that led to your choices."
        )

        prompts[i] = prompt

    return prompts

# def temporal_cluster_loss(predictions, cluster_intervals): # (616, 48), ()
#     """
#     predictions: Tensor of shape [B, T, C] -> Model outputs (softmax logits)
#     cluster_intervals: List of tuples defining cluster boundaries [(0, 2), (3, 5), (6, 10)].
#     """
#     loss = 0.0
#     for batch in cluster_intervals:
#         for start, end in batch:
#             cluster_preds = predictions[:, start:end + 1, :]  # Get predictions for the time range
#             cluster_mean = torch.mean(cluster_preds, dim=1, keepdim=True)  # Mean prediction within the cluster
#             loss += F.mse_loss(cluster_preds, cluster_mean.expand_as(cluster_preds))
    
#     return loss / len(cluster_intervals)


def temporal_contrastive_loss(predictions, cluster_intervals, temperature=0.07):
    """
    Temporal-aware supervised contrastive loss with both positive and negative pairs.

    Args:
        predictions: Tensor of shape [B, T, C].
        cluster_intervals: List of lists of tuples (temporal boundaries).
        temperature: Temperature scaling for contrastive loss.

    Returns:
        Total contrastive loss: intra-cluster consistency + inter-cluster separation.
    """
    device = predictions.device
    loss = 0.0

    for batch_idx, batch in enumerate(cluster_intervals):
        batch_preds = predictions[batch_idx]  # [T, C]
        batch_preds = F.normalize(batch_preds, p=2, dim=1)  # Normalize embeddings

        for start, end in batch:
            cluster_preds = batch_preds[start:end + 1, :]  # [N, C]
            N = cluster_preds.size(0)

            # Pairwise similarity matrix
            similarity = torch.mm(cluster_preds, batch_preds.T) / temperature  # Compare with entire batch
            exp_similarity = torch.exp(similarity)

            # Positive mask: Same cluster
            pos_mask = torch.zeros_like(similarity, device=device)
            pos_mask[:, start:end + 1] = 1  # Positive pairs
            pos_mask.fill_diagonal_(0)  # Remove self-similarity

            # Compute positive and negative losses
            pos_loss = -torch.log(exp_similarity / exp_similarity.sum(dim=1, keepdim=True) + 1e-5)
            pos_loss = (pos_loss * pos_mask).sum() / (pos_mask.sum() + 1e-5)  # Average positive loss

            # Aggregate loss
            loss += pos_loss

    return loss / len(cluster_intervals)


def temporal_cluster_loss(predictions, cluster_intervals):
    """
    Compute clustering loss with intra-cluster consistency and inter-cluster separation.

    Args:
        predictions: Tensor of shape [B, T, C] -> Model outputs (softmax logits).
        cluster_intervals: List of lists of tuples -> Cluster boundaries for each batch.
                           Example: [[(0, 2), (3, 5), (6, 10)], ...].

    Returns:
        Combined loss: intra-cluster + inter-cluster loss.
    """
    intra_loss = 0.0
    inter_loss = 0.0
    cluster_means = []  # Store cluster centroids for inter-cluster separation
    total_clusters = 0  # Count total clusters for normalization

    for batch_idx, batch in enumerate(cluster_intervals):  # Loop through batches
        batch_means = []  # Collect cluster means for the current batch
        for start, end in batch:  # Loop through each cluster
            # Get predictions for the current cluster
            cluster_preds = predictions[batch_idx, start:end + 1, :]  # Shape: [N, C]
            
            # Compute the cluster mean (centroid)
            cluster_mean = torch.mean(cluster_preds, dim=0, keepdim=True)  # Shape: [1, C]
            batch_means.append(cluster_mean)

            # Intra-cluster consistency loss (MSE loss within cluster)
            intra_loss += F.mse_loss(cluster_preds, cluster_mean.expand_as(cluster_preds))

        # Store batch cluster means for inter-cluster separation
        if len(batch_means) > 1:
            batch_means = torch.cat(batch_means, dim=0)  # Shape: [num_clusters, C]
            cluster_means.append(batch_means)
        
        total_clusters += len(batch)

    # Inter-cluster separation loss
    for batch_means in cluster_means:  # Loop through all batches
        num_clusters = batch_means.size(0)
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):  # Compare all cluster pairs
                inter_loss += 1.0 / (1e-5 + torch.norm(batch_means[i] - batch_means[j], p=2))  # L2 distance

    # Normalize losses
    intra_loss = intra_loss / total_clusters if total_clusters > 0 else 0.0
    inter_loss = inter_loss / (len(cluster_means) * (num_clusters - 1)) if len(cluster_means) > 0 else 0.0

    # Combine intra-cluster and inter-cluster loss
    total_loss = intra_loss + inter_loss  # Weight inter-cluster separation
    return total_loss



def normalize_duration(input, mask):
    # 수치적 안정성을 위해 최댓값을 빼줌 (Log-Sum-Exp trick과 유사)
    input_max = torch.max(input, dim=-1, keepdim=True)[0]
    input = torch.exp(input - input_max) * mask
    output = input / (input.sum(dim=-1, keepdim=True) + 1e-12)
    return output

def read_mapping_dict(file_path):
    # github.com/yabufarha/anticipating-activities
    '''This function read action index from the txt file'''
    file_ptr = open(file_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

def eval_file(gt_content, recog_content, obs_percentage, classes,log):
    # github.com/yabufarha/anticipating-activities
    last_frame = min(len(recog_content), len(gt_content))
    log.write(f"{obs_percentage}\n\n")
    recognized = recog_content[int(obs_percentage * len(gt_content)):last_frame]
    ground_truth = gt_content[int(obs_percentage * len(gt_content)):last_frame]

    n_T = np.zeros(len(classes))
    n_F = np.zeros(len(classes))
    for i in range(len(ground_truth)):
        ground_truth[i] = ground_truth[i].replace(' ', '')
        if ground_truth[i] == recognized[i]:
            log.write(f"{i}th, GT: {ground_truth[i]}\t,Pred: {recognized[i]}\t Correct \n")
            n_T[classes[ground_truth[i]]] += 1
        else:
            #log.write(f"GT: {ground_truth[i]}\t,Pred: {recognized[i]}\t Wrong \n")
            n_F[classes[ground_truth[i]]] += 1

    return n_T, n_F

def cal_performance(pred, gold, trg_pad_idx, exclude_class_idx=None, smoothing=False, reference=None, target_ref=None):
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''Apply label smoothing if needed'''
    
    l2_correct = None
    loss = 0
    if reference is not None:
        loss, l2_correct = cal_weighted_loss(pred, gold.long(), trg_pad_idx, reference, exclude_class_idx=exclude_class_idx, target_ref=target_ref, smoothing=smoothing)
    else:
        loss, l2_correct = cal_loss(pred, gold.long(), trg_pad_idx, exclude_class_idx=exclude_class_idx, smoothing=smoothing)
    pred = pred.max(1)[1]
#    gold = gold.contiguous().view(-1)
    #
    if exclude_class_idx == None:
        non_pad_mask = gold.ne(trg_pad_idx)
    else:
        non_pad_mask = gold.ne(trg_pad_idx) & gold.ne(exclude_class_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word, l2_correct

def cal_acc_l3(pred, gold, trg_pad_idx, exclude_class_idx=None):
    
    if exclude_class_idx == None:
        non_pad_mask = gold.ne(trg_pad_idx)
    else:
        non_pad_mask = gold.ne(trg_pad_idx) & gold.ne(exclude_class_idx)
    n_correct = pred.eq(gold.long()).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    print("L3 pred: ", pred)
    print("L3 gold: ", gold)

    return n_correct, n_word


def cal_performance_focal(pred, gold, trg_pad_idx, exclude_class_idx=None, smoothing=False, reference=None, target_ref=None):
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''Apply label smoothing if needed'''
    loss, l3_correct = focal_loss(pred, gold.long(), trg_pad_idx, exclude_class_idx=exclude_class_idx)
    pred = pred.max(1)[1]
#    gold = gold.contiguous().view(-1)
    #
    if exclude_class_idx == None:
        non_pad_mask = gold.ne(trg_pad_idx)
    else:
        non_pad_mask = gold.ne(trg_pad_idx) & gold.ne(exclude_class_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word, l3_correct

def cal_weighted_loss(pred, gold, trg_pad_idx, t_n_labels, exclude_class_idx=None, weight_same=1.0, weight_different=10.0, target_ref=None, smoothing=False):
    '''Calculate weighted cross entropy loss based on comparison between t+n and t+m labels.'''
    if smoothing:
        eps = 0.1
        n_class = pred.size(1) + 1
        B = pred.size(0)

        one_hot = torch.zeros((B, n_class)).to(pred.device).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        one_hot = one_hot[:, :-1]
        log_prb = F.log_softmax(pred, dim=1)

        #non_pad_mask = gold.ne(trg_pad_idx)
        non_pad_mask = gold.ne(trg_pad_idx) & gold.ne(exclude_class_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
        loss = loss / non_pad_mask.sum()
    else:
        mask = (gold != trg_pad_idx) & (gold != exclude_class_idx)

        # Apply mask to the gold labels and use ignore_index=-1 for masked entries
        masked_gold = gold.clone()
        masked_gold[~mask] = -1
        base_loss = F.cross_entropy(pred, masked_gold, ignore_index=-1, reduction='none')

        l2_correct = (pred.argmax(dim=-1) == masked_gold)

        # Get the first label in `gold` (t+m) and the last label in `t_n_labels` (t+n)
        # Create weights based on whether `t_m_labels` and `t_n_labels_last` are the same
        weights = torch.where(t_n_labels == target_ref, weight_same, weight_different)

        repeat_factor = base_loss.size(0) // weights.size(0)
        expanded_weights = weights.repeat_interleave(repeat_factor)

        # Apply weights to each sequence's loss
        weighted_loss = base_loss * expanded_weights
        loss = weighted_loss.mean()  # Average over all sequences in the batch
    return loss, l2_correct

def cal_loss(pred, gold, trg_pad_idx, exclude_class_idx=None, smoothing=False, penalty_weight=2.0):
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''Calculate cross entropy loss, apply label smoothing if needed'''

#    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1) + 1
        B = pred.size(0)

        one_hot = torch.zeros((B, n_class)).to(pred.device).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class -1)
        one_hot = one_hot[:, :-1]
        log_prb = F.log_softmax(pred, dim=1)

        #non_pad_mask = gold.ne(trg_pad_idx)
        non_pad_mask = gold.ne(trg_pad_idx) & gold.ne(exclude_class_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
        loss = loss / non_pad_mask.sum()
    else:
        #loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx)
        mask = (gold != trg_pad_idx) & (gold != exclude_class_idx)

        # Apply mask to the gold labels and use ignore_index=-1 for masked entries
        masked_gold = gold.clone()
        masked_gold[~mask] = -1
        base_loss = F.cross_entropy(pred, masked_gold, ignore_index=-1, reduction='none')
        l2_correct = (pred.argmax(dim=-1) == masked_gold)

        # Add penalty for non-pad_idx labeled items predicted as pad_idx
        pred_classes = pred.argmax(dim=1)
        #non_pad_mask = gold.ne(trg_pad_idx)  # True for non-pad_idx labels
        penalty_mask = (pred_classes == trg_pad_idx) & mask  # False for correct predictions

        # Calculate penalty for incorrect pad_idx predictions
        penalty = penalty_weight * penalty_mask.float()
        
        # Total loss including penalty for incorrect pad_idx predictions
        loss = (base_loss + penalty).mean()
    return loss, l2_correct


def focal_loss(pred, gold, trg_pad_idx, exclude_class_idx=None, alpha=1.0, gamma=2.0, penalty_weight=0.0):
    """
    Compute Focal Loss with optional penalty for misclassified pad_idx predictions.
    
    Args:
        pred (torch.Tensor): Predicted logits of shape [batch_size, num_classes].
        gold (torch.Tensor): Ground truth labels of shape [batch_size].
        trg_pad_idx (int): Index used for padding in the target.
        exclude_class_idx (int, optional): Index of a class to exclude from loss computation.
        alpha (float): Scaling factor for class imbalance.
        gamma (float): Focusing parameter to down-weight easy examples.
        penalty_weight (float): Weight for penalty applied to incorrect pad_idx predictions.
        
    Returns:
        torch.Tensor: Computed focal loss.
    """
    # Mask for valid labels (non-padding and not excluded class)
    mask = (gold != trg_pad_idx)
    if exclude_class_idx is not None:
        mask &= (gold != exclude_class_idx)
    
    # Apply mask to gold labels and set ignored indices to -1
    masked_gold = gold.clone()
    masked_gold[~mask] = -1

    # Compute standard cross-entropy loss
    ce_loss = F.cross_entropy(pred, masked_gold, ignore_index=-1, reduction='none')
    
    assert pred.shape[-1] > 0, "Last dimension of pred must be greater than 0!"

    l3_correct = (pred.argmax(dim=-1) == masked_gold)

    # Calculate probabilities for the true class
    pred_probs = F.softmax(pred, dim=1)  # Convert logits to probabilities
    true_probs = pred_probs[torch.arange(pred.size(0)), gold]  # Probabilities of the true class

    # Apply the focal loss scaling factor
    focal_weight = alpha * ((1 - true_probs) ** gamma)  # Down-weight easy examples
    focal_loss = focal_weight * ce_loss

    # Add penalty for misclassified padding predictions
    pred_classes = pred.argmax(dim=1)
    penalty_mask = (pred_classes == trg_pad_idx) & mask  # Predictions of pad_idx for valid labels
    penalty = penalty_weight * penalty_mask.float()

    # Compute final loss
    loss = (focal_loss + penalty).mean()
    return loss, l3_correct



