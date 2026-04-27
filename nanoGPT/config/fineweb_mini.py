# Train a small baseline nanoGPT on FineWeb-EDU
# Launch with: python train.py config/fineweb_mini.py

out_dir = 'out-fineweb-mini'
eval_interval = 250 # Evaluation loop frequency
eval_iters = 200    # How many batches to use for evaluation
log_interval = 10   # Log training loss every N steps

# Checkpointing
always_save_checkpoint = False # Only save when val loss improves

# WandB logging (Optional: set to True if you have a wandb account)
wandb_log = False
wandb_project = 'fineweb-mini'
wandb_run_name = 'mini-gpt-baseline'

# Dataset
dataset = 'finewebedu/edu_fineweb10B' # Expects data in data/fineweb_edu/
gradient_accumulation_steps = 1 # Increase if you run out of VRAM
batch_size = 32     # Adjust based on VRAM (64 is good for 384 embedding)
block_size = 256    # Context window size (Matches your setup)

# Model Configuration (Your "Small Baseline" Specs)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False 

# Optimization (Standard nanoGPT defaults adjusted for small models)
learning_rate = 1e-3 # Higher LR for smaller models
max_iters = 5000
lr_decay_iters = 5000 # Make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # Make a bit bigger because number of tokens per step is small

# Hardware settings
device = 'cuda' # or 'cpu' or 'mps' for Mac
compile = False # PyTorch 2.0 compilation (Set False if errors occur)
