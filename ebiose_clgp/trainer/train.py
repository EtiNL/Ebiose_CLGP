import torch
import torch.nn.functional as F
import os
from omegaconf import OmegaConf
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from ebiose_clgp.data_utils.dataloader import get_dataloader
from ebiose_clgp.data_utils.dataset import CLGP_Ebiose_dataset
from ebiose_clgp.model.CLGP import CLGP
from ebiose_clgp.trainer.train_utils import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, set_seed
from ebiose_clgp.trainer.utils import mkdir, load_config_file
from ebiose_clgp.data_utils.tokenizer import get_max_position_embedding

from torch.optim import AdamW

import wandb

DATA_CONFIG_PATH = 'Ebiose_CLGP/ebiose_clgp/data_utils/data_config.yaml'
TRAINER_CONFIG_PATH = 'Ebiose_CLGP/ebiose_clgp/trainer/train_config.yaml'
MODEL_CONFIG_PATH = 'Ebiose_CLGP/ebiose_clgp/model/model_config.yaml'

def log_gradients(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().data.numpy())}, step=step)

def train(config, train_dataset, model):
    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    train_dataloader = get_dataloader(config, train_dataset, is_train=True)

    # Total training iterations
    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
    
    optimizer = AdamW(model.parameters(), lr=config.optimizer.lr, eps=config.optimizer.eps, weight_decay=config.optimizer.weight_decay)

    # Warmup iterations = 20% of total iterations
    num_warmup_steps = int(0.20 * t_total)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(torch.device(config.device))
    model.train()
    
    wandb.config.update({
        "learning_rate": config.optimizer.lr,
        "batch_size": config.train_batch_size,
        "epochs": config.num_train_epochs,
        "weight_decay": config.optimizer.weight_decay,
        "eps": config.optimizer.eps,
        "gradient_accumulation_steps": config.gradient_accumulation_steps
    })

    global_step, global_loss, global_acc = 0, 0.0, 0.0
    model.zero_grad()

    scaler = GradScaler()
    
    for epoch in range(int(config.num_train_epochs)):
        for step, batch in tqdm(enumerate(train_dataloader)):
            torch.cuda.empty_cache()  # Clear the cache
            with autocast():
                input_graphs, input_texts = batch

                input_graphs = input_graphs.to(torch.device(config.device))
                input_texts = input_texts.to(torch.device(config.device))
                
                graph_features, text_features = model(input_graphs, input_texts)

                graph_features = graph_features / graph_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                if config.n_gpu <= 1:
                    logit_scale = model.logit_scale.exp()
                elif config.n_gpu > 1:
                    logit_scale = model.module.logit_scale.exp()

                logits_per_graph = logit_scale * graph_features @ text_features.t()
                logits_per_text = logit_scale * text_features @ graph_features.t()

                labels = torch.arange(len(logits_per_graph)).to(logits_per_graph.device)

                graph_loss = F.cross_entropy(logits_per_graph, labels)
                text_loss = F.cross_entropy(logits_per_text, labels)

                loss = (graph_loss + text_loss) / 2

                if config.n_gpu > 1: 
                    loss = loss.mean()
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1
                scaler.step(optimizer)
                scaler.update()
                
                if config.n_gpu == 1:
                    model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
                elif config.n_gpu > 1:
                    model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

                optimizer.zero_grad()

                if scheduler:
                    scheduler.step()

                if global_step % config.logging_steps == 0:
                    wandb.log({'epoch': epoch, 'loss': loss.item(), 'lr': optimizer.param_groups[0]["lr"]}, step=global_step)
                    log_gradients(model, global_step)

            if (config.save_steps > 0 and global_step % config.save_steps == 0) or global_step == t_total:
                save_checkpoint(config, epoch, global_step, model, optimizer)
                    
    wandb.save(config.model_save_name)
    
    return global_step, global_loss / global_step

def save_checkpoint(config, epoch, global_step, model, optimizer):
    '''
    Checkpointing. Saves model and optimizer state_dict() and current epoch and global training steps.
    '''
    checkpoint_path = os.path.join(config.saved_checkpoints, f'checkpoint_{epoch}_{global_step}.pt')
    save_num = 0
    while save_num < 10:
        try:
            if config.n_gpu > 1:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
            break
        except Exception as e:
            save_num += 1
            print(f"Failed to save checkpoint: {e}")
    if save_num == 10:
        print("Failed to save checkpoint after 10 trials.")
    return

def main():
    data_config = load_config_file(DATA_CONFIG_PATH)
    train_config = load_config_file(TRAINER_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)

    config = OmegaConf.merge(train_config, data_config, model_config)

    wandb.init(project='Ebiose_CLGP')

    mkdir(path=config.saved_checkpoints)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.n_gpu = torch.cuda.device_count()
    config.graph_node_tokenizer_max_pos = get_max_position_embedding(config.graph_feature_tokenizer)
    config.prompt_tokenizer_max_pos = get_max_position_embedding(config.prompt_tokenizer)
    set_seed(seed=11, n_gpu=config.n_gpu)
    
    model = CLGP(config)

    train_dataset = CLGP_Ebiose_dataset(config)
    
    global_step, avg_loss = train(config, train_dataset, model)
    
    print("Training done: total_step = {}, avg loss = {}".format(global_step, avg_loss))

if __name__ == "__main__":
    main()