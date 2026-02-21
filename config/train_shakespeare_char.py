out_dir = 'out_custom'  
  
# Evaluation settings  
eval_interval = 10  # Adjusted for longer runs  
eval_iters = 50  
log_interval = 10  
  
# Dataset settings - using your custom dataset  
dataset = 'custom'  
  
# Model architecture  
n_layer = 4  
n_head = 4  
n_embd = 1024 
block_size = 8192  # 8192 context length as requested  
dropout = 0.1  
  
# Batch configuration - total batch size = 16384  
batch_size = 128  
gradient_accumulation_steps = 128  # 128 * 128 = 16384 effective batch size  
  
# Training duration - calculated for 8 epochs  
# Train tokens: 1,544,247,575 * 8 epochs = 12,353,980,600 tokens  
# Tokens per iter: 16384 * 8192 = 134,217,728  
# Iterations needed: ~92  
max_iters = 100  # Slightly over 92 to ensure 8 epochs  
lr_decay_iters = 100  
  
# Learning rate  
learning_rate = 3e-4  
min_lr = 3e-5  
warmup_iters = 5  # Increased slightly for longer run  
decay_lr = True  
  
# Optimizer settings  
weight_decay = 1e-1  
beta1 = 0.9  
beta2 = 0.95  
grad_clip = 1.0  
  
# Checkpointing  
always_save_checkpoint = True  
  
# GPU settings  
device = 'cuda'  
compile = True  
dtype = 'bfloat16'
