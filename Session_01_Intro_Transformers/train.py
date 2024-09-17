
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from data import train_data, val_data, chars, GPTConfig
from model import GPT

def main():
    learning_rate = 6e-4 # max learning rate
    max_iters = 600000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'float32'
    # dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    # -----------------------------------------------------------------------------
    # config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    # tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    # print(f"tokens per iteration will be: {tokens_per_iter:,}")

    out_dir = "trained_models/"
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    # torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # poor man's data loader
    batch_size = 128 #12
    gptconf = GPTConfig()

    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            # data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
            data = train_data
        else:
            # data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
            data = val_data
        # print(data.shape)
            
        ix = torch.randint(len(data) - gptconf.block_size, (batch_size,))
        x = torch.stack([data[i:i+gptconf.block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+gptconf.block_size] for i in ix])
        # print('BATCH DIM', x.shape, y.shape)
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # # attempt to derive vocab_size from the dataset
    # meta_path = os.path.join(data_dir, 'meta.pkl')
    # meta_vocab_size = None
    # if os.path.exists(meta_path):
    #     with open(meta_path, 'rb') as f:
    #         meta = pickle.load(f)
    #     meta_vocab_size = meta['vocab_size']
    #     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training]
    gptconf = GPTConfig()
    gptconf.vocab_size = len(chars) + 1
    model = GPT(gptconf)
    # crop down the model block size if desired, using model surgery
    # if block_size < model.config.block_size:
    #     model.crop_block_size(block_size)
    #     model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    checkpoint = None # free up memory

    # compile the model
    # if compile:
    #     print("compiling the model... (takes a ~minute)")
    #     unoptimized_model = model
    #     model = torch.compile(model) # requires PyTorch 2.0


    # helps estimate an arbitrarily accurate loss over either split using many batches
    eval_iters = 200

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                # print(f"evaluating loss on {split} set, batch {k+1}/{eval_iters}")
                X, Y = get_batch(split)
                # print(X.shape, Y.shape)
                with ctx:
                    try:
                        logits, loss = model(X, Y)
                    except Exception as e:
                        print(f"error in batch {k} of {split} set, skipping", torch.min(X), torch.max(X), torch.min(Y), torch.max(Y))
                        raise e
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model # unwrap DDP container if needed
    running_mfu = -1.0

    log_interval = 100
    eval_interval = 2000
    always_save_checkpoint = True
    eval_only = False
    gradient_accumulation_steps = 1
    for iter_num in (pbar := tqdm(range(0, max_iters, gradient_accumulation_steps))):
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            # print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            pbar.set_description(f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        # 'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        
        for micro_step in range(gradient_accumulation_steps):
            # if ddp:
            #     # in DDP training we only need to sync gradients at the last micro step.
            #     # the official way to do this is with model.no_sync() context manager, but
            #     # I really dislike that this bloats the code and forces us to repeat code
            #     # looking at the source of that context manager, it just toggles this variable
            #  model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            pbar.set_description(f"loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        # if iter_num > max_iters:
        #     break

if __name__ == '__main__':
    main()