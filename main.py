import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from adam_atan2 import AdamATan2

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from common import compute_file_hash
from config import config
from dataloader import download_sudoku_dataset, create_dataloader, SudokuDataset
from model import HRM_model


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def train_batch(model, batch, carry):
    carry, outputs = model(batch, carry)
    
    # logit stablemax/softmax cross entropy
    token_logits = outputs["output_token_embeds"]
    token_labels = carry.current_data["labels"]
    
    # Q-halt BCEwithLogitsLoss
    # but first we will have to compute the correctness of the q_halt_logits
    # for that we need the labels, and compute if the predictions from the logits are all correct or not using argmax.
    # and this becomes the reward!
    # 
    # for all the samples in the batch where all the predictions are correct according to the labels, we set
    # the q_halt value to 1.0.
    # for all the samples in the batch where even one prediction is incorrect according to the lavels, we set
    # the q_halt value to 0.0.
    # i dont know why BCEwithLogits is being use here. could have just been BCE loss lol.
    token_predictions = torch.argmax(token_logits, dim=-1) # this has shape [B, 81, 11] and it gets argmaxed over the last dimension.
    
    
    # checked = False
    # assert checked, "TODO: you still have to check from the original repo if the originally placed digits are predicted or not."
    ### comments: on a cursory inspection of the original dataset, the labels do not seem to contain the ignore label index at all.
    ### THIS MEANS THAT EVERY ONE OF THE TOKEN POSITIONS IS BEING REGRESSED UPON, WHETHER OR NOT IT IS A PROBLEM STATEMENT INPUT,
    ### ... OR A BLANK SPACE TO BE REGRESSED! 
    
    prediction_count = torch.sum(torch.where(token_labels != config.ignore_index, 1, 0), dim=-1) # sum across the sample dimension. shape [B]
    correct_count = torch.sum(torch.where(token_labels == token_predictions, 1, 0), dim=-1) # sum across the sample dimension. shape [B]
    g_halt = torch.where(prediction_count == correct_count, 1.0, 0.0) # shape [B]
    
    # token loss
    token_loss = stablemax_cross_entropy(token_logits, token_labels, config.ignore_index).sum(dim=-1) / prediction_count # sum across the sample dim and mean across batch
    token_loss = token_loss.mean()
    
    # q_halt loss
    q_halt_loss = nn.functional.binary_cross_entropy_with_logits(outputs["q_halt_logits"], g_halt)
    
    # q_continue loss
    q_continue_loss = nn.functional.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["g_continue"])
    
    loss = token_loss + 0.5 * (q_halt_loss + q_continue_loss)

    return loss, carry
    

def eval_batch(model, batch, carry):
    pass


def main(config):
    # load the datasets
    if not os.path.exists(config.train_data_path) or not os.path.exists(config.test_data_path):
        download_sudoku_dataset()
    
    
    # try loading the datasets from cache, else save them
    train_hash = compute_file_hash(config.train_data_path)
    test_hash  = compute_file_hash(config.test_data_path)
    if not os.path.exists(f'/tmp/train_{train_hash}.pt'):
        train_dataset = SudokuDataset(config.train_data_path)
        torch.save(train_dataset, f'/tmp/train_{train_hash}.pt')
    if not os.path.exists(f'/tmp/test_{test_hash}.pt'):
        test_dataset = SudokuDataset(config.test_data_path)
        torch.save(test_dataset, f'/tmp/test_{test_hash}.pt')
    train_dataset = torch.load(f'/tmp/train_{train_hash}.pt', weights_only=False)
    test_dataset  = torch.load(f'/tmp/test_{test_hash}.pt'  , weights_only=False)
    
    # create the dataloaders
    train_dataloader = create_dataloader(train_dataset, config.batch_size)
    test_dataloader  = create_dataloader( test_dataset, config.batch_size)
    
    
    # create the model
    model = HRM_model(config).to(config.device)
    
    # create the optimizers
    optimizers = [
        AdamATan2(
            [p for n, p in model.named_parameters() if 'puzzle_emb' not in n], lr=config.lr
        ),
        AdamATan2(
            [p for n, p in model.named_parameters() if 'puzzle_emb' in n], lr=config.puzzle_emb_lr
        )
    ]
    
    # create the scheduler, cosine annealing with linear warmup. have no idea why in the original sapientinc config it was min_lr_ratio=1.0
    schedulers = [
        CosineAnnealingWarmupRestarts(
            optimizers[0],
            first_cycle_steps=config.epochs,       # entire training span
            warmup_steps=config.lr_warmup_steps,   # linear warmup steps
            max_lr=config.lr,
            min_lr=config.lr * config.lr_min_ratio,                # e.g. lr_min_ratio=0.1
            cycle_mult=1.0,
            gamma=1.0,
        ),
        CosineAnnealingWarmupRestarts(
            optimizers[1],
            first_cycle_steps=config.epochs,       # entire training span
            warmup_steps=config.lr_warmup_steps,   # linear warmup steps
            max_lr=config.puzzle_emb_lr,
            min_lr=config.puzzle_emb_lr * config.lr_min_ratio,                # e.g. lr_min_ratio=0.1
            cycle_mult=1.0,
            gamma=1.0,
        )
    ]
    
    # no loss funciton, it will be included directly inside of the training loop
    
    # training loop
    train_loader_iter = iter(train_dataloader)
    bar = tqdm(total=config.epochs, desc=f"Step {0}/{config.epochs}   Loss: {0:.4f}")
    for epoch in range(config.epochs):
        # train batch
        batch = next(train_loader_iter)
        batch = {k: v.to(config.device) for k, v in batch.items()}
        # forward pass
        # initialize the carry
        if epoch == 0:
            carry = model.initial_carry(batch)

        loss, carry = train_batch(model, batch, carry)
        if (epoch + 1) % config.eval_interval == 0:
            eval_batch(model, batch, carry)
        # backward pass
        loss.backward()
        # update the model and schedulers
        for optimizer, scheduler in zip(optimizers, schedulers):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        bar.update(1)
        bar.set_description(f"Epoch {epoch+1}/{config.epochs}   Loss: {loss.item():.4f}")
    
if __name__ == '__main__':
    main(config)