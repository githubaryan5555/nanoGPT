out_dir = 'out-shakespeare-char'  
eval_interval = 5  # Changed from 100 to ensure evaluation occurs  
eval_iters = 50  
log_interval = 10  
  
# Dataset settings - using shakespeare_char dataset  
dataset = 'shakespeare_char'  
  
# Model architecture - 4 layers, 4096 context  
n_layer = 4  
n_head = 4  
n_embd = 256  
block_size = 4096  
dropout = 0.1  
  
# Batch configuration  
batch_size = 16  
gradient_accumulation_steps = 32  
  
# Training duration  
max_iters = 8  
lr_decay_iters = 8  
  
# Learning rate  
learning_rate = 3e-4  
min_lr = 3e-5  
warmup_iters = 2  
decay_lr = True  
  
# Optimizer settings  
weight_decay = 1e-1  
beta1 = 0.9  
beta2 = 0.95  
grad_clip = 1.0  
  
# Checkpointing - added this  
always_save_checkpoint = True  # Force saving at each evaluation  
  
# Single GPU settings  
device = 'cuda'  
compile = True  
dtype = 'bfloat16'
