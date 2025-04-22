import json
import torch
import math
import os
import time
import yaml
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split # If splitting inside script
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer # Added just in case for flexibility
)
from functools import partial
import logging # Use logging module

# Import dataset and collate function from within the same directory
from .fid_dataset import QA_Dataset_FiD, collate_fn_fid

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(config_path: str):
    """
    Loads configuration, prepares data, and runs the T5-FiD training loop.

    Args:
        config_path: Path to the YAML configuration file (e.g., 'config/params.yaml').
    """
    # --- Load Configuration ---
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract relevant config sections
    data_cfg = config['data_paths']
    model_cfg = config['model_paths']
    train_cfg = config['training']

    # --- Device Setup ---
    if config['hardware']['device']:
        device = torch.device(config['hardware']['device'])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Tokenizer ---
    tokenizer_name = model_cfg['t5_base_model_name']
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    # Consider using AutoTokenizer for flexibility if base model changes
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

    # --- Load and Prepare Data ---
    # This assumes the augmented data is already created by a previous script.
    # If you need to split it here:
    train_data_path = train_cfg['train_data_path']
    logger.info(f"Loading training data from: {train_data_path}")
    with open(train_data_path, 'r') as f:
        full_augmented_dataset = json.load(f)

    # Optional: Split data if not done previously
    # train_data, val_data = train_test_split(
    #     full_augmented_dataset,
    #     test_size=0.2, # Or use a dedicated validation set path from config
    #     random_state=train_cfg['seed']
    # )
    # Using the whole set for training as per preliminary notebook setup
    train_data = full_augmented_dataset
    logger.info(f"Loaded {len(train_data)} training samples.")
    # logger.info(f"Using {len(val_data)} validation samples.") # If using validation

    train_dataset = QA_Dataset_FiD(
        train_data,
        tokenizer,
        max_input_length=train_cfg['max_input_length'],
        max_target_length=train_cfg['max_target_length']
    )
    # val_dataset = QA_Dataset_FiD(...) # If using validation

    # --- Calculate Gradient Accumulation ---
    if train_cfg['effective_train_batch_size'] % train_cfg['per_device_train_batch_size'] != 0:
        raise ValueError("Effective batch size must be divisible by per-device batch size")
    gradient_accumulation_steps = train_cfg['effective_train_batch_size'] // train_cfg['per_device_train_batch_size']
    logger.info(f"Effective Batch Size: {train_cfg['effective_train_batch_size']}")
    logger.info(f"Per-Device Batch Size: {train_cfg['per_device_train_batch_size']}")
    logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")


    # --- Create DataLoaders ---
    collate_partial = partial(
        collate_fn_fid,
        tokenizer=tokenizer,
        max_docs_per_item=train_cfg['max_docs_per_item'],
        max_input_length=train_cfg['max_input_length'],
        max_target_length=train_cfg['max_target_length']
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg['per_device_train_batch_size'],
        shuffle=True,
        collate_fn=collate_partial
    )
    # val_dataloader = DataLoader(val_dataset, ...) # If using validation

    # --- Initialize Model ---
    model_name = model_cfg['t5_base_model_name']
    logger.info(f"Initializing model: {model_name}")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # --- Optimizer and Scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        eps=train_cfg['adam_epsilon'],
        weight_decay=train_cfg['weight_decay']
    )

    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if num_update_steps_per_epoch == 0:
         # Handle case where dataloader is smaller than accumulation steps
         num_update_steps_per_epoch = 1
         logger.warning(f"Train dataloader size ({len(train_dataloader)}) is smaller than gradient accumulation steps ({gradient_accumulation_steps}). Effective steps per epoch: 1")


    total_training_steps = train_cfg['num_train_epochs'] * num_update_steps_per_epoch
    num_warmup_steps = int(total_training_steps * train_cfg['warmup_proportion'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )

    logger.info(f"--- Training Configuration ---")
    logger.info(f"Num Epochs: {train_cfg['num_train_epochs']}")
    logger.info(f"Total Optimizer Steps: {total_training_steps}")
    logger.info(f"Warmup Steps: {num_warmup_steps}")
    logger.info(f"Logging Steps: {train_cfg['logging_steps']}")
    logger.info(f"Save Checkpoints Every {train_cfg['save_epochs']} Epochs")
    logger.info(f"Output Directory: {model_cfg['t5_finetuned_dir']}")
    logger.info(f"-----------------------------")


    # --- Training Loop ---
    global_optimizer_step = 0
    total_training_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()

    for epoch in range(train_cfg['num_train_epochs']):
        logger.info(f"Starting Epoch {epoch+1}/{train_cfg['num_train_epochs']}")
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        processed_batches_in_epoch = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} Batches")):
            input_ids_batch = batch['input_ids'].to(device)
            attention_mask_batch = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if input_ids_batch.numel() == 0:
                logger.warning(f"Skipping empty batch at step {step} in epoch {epoch+1}")
                continue

            bsz, n_docs, seq_len = input_ids_batch.shape

            # Reshape for encoder
            input_ids_enc = input_ids_batch.view(bsz * n_docs, seq_len)
            attention_mask_enc = attention_mask_batch.view(bsz * n_docs, seq_len)

            # Prepare decoder inputs
            cross_attention_mask = attention_mask_batch.view(bsz, n_docs * seq_len)

            # Forward pass
            outputs = model(
                input_ids=input_ids_enc,              # Only needed if not using encoder_outputs
                attention_mask=attention_mask_enc,    # Only needed if not using encoder_outputs
                decoder_attention_mask=cross_attention_mask, # Pass the combined mask here for FiD
                labels=labels,
                return_dict=True,
                # Pass encoder outputs directly to avoid recomputing (more standard FiD)
                # encoder_outputs=model.encoder(input_ids=input_ids_enc, attention_mask=attention_mask_enc, return_dict=True)
            )
            loss = outputs.loss

            if loss is None:
                logger.warning(f"Loss is None at step {step} in epoch {epoch+1}. Skipping backward/step.")
                continue

            # Scale loss for gradient accumulation
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            epoch_loss += loss.item()
            total_training_loss += loss.item()
            processed_batches_in_epoch += 1

            # --- Optimizer Step ---
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                # Optional: Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step() # Update learning rate schedule
                optimizer.zero_grad()
                global_optimizer_step += 1

                # Logging
                if global_optimizer_step % train_cfg['logging_steps'] == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    loss_since_last_log = (total_training_loss - logging_loss) / (train_cfg['logging_steps'] * gradient_accumulation_steps)
                    logger.info(f"Epoch: {epoch+1}, Opt Step: {global_optimizer_step}/{total_training_steps}, LR: {current_lr:.2e}, Avg Loss (last {train_cfg['logging_steps']} steps): {loss_since_last_log:.4f}")
                    logging_loss = total_training_loss

        # --- End of Epoch ---
        epoch_end_time = time.time()
        avg_epoch_loss = epoch_loss / processed_batches_in_epoch if processed_batches_in_epoch > 0 else 0
        logger.info(f"Epoch {epoch+1} Finished. Average Loss: {avg_epoch_loss:.4f}. Time: {epoch_end_time - epoch_start_time:.2f}s")

        # --- Save Model Checkpoint Per Epoch ---
        if (epoch + 1) % train_cfg['save_epochs'] == 0:
            output_dir = model_cfg['t5_finetuned_dir']
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint for epoch {epoch+1} to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Model checkpoint saved.")

            # Optional: Save training args and config as well
            # torch.save(train_cfg, os.path.join(output_dir, "training_args.bin"))
            # with open(os.path.join(output_dir, "run_config.yaml"), 'w') as f_cfg:
            #    yaml.dump(config, f_cfg)


    logger.info("Training finished.")
    avg_total_loss = total_training_loss / (len(train_dataloader) * train_cfg['num_train_epochs']) if len(train_dataloader) > 0 and train_cfg['num_train_epochs'] > 0 else 0
    logger.info(f"Average training loss across all epochs: {avg_total_loss:.4f}")

# --- Main Execution Block ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune T5-FiD model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    train_model(args.config)
