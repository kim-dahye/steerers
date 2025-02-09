from types import SimpleNamespace

import sys
import torch
sys.path.append("..")
from training.config import SDSAERunnerConfig
from training.sd_activations_store import SDActivationsStore
from typing import Optional
import wandb
import tqdm
from training.k_sparse_autoencoder import SparseAutoencoder,  unit_norm_decoder_, unit_norm_decoder_grad_adjustment_
import argparse

def weighted_average(points: torch.Tensor, weights: torch.Tensor):
    weights = weights / weights.sum()
    return (points * weights.view(-1, 1)).sum(dim=0)
    
@torch.no_grad()
def geometric_median_objective(
    median: torch.Tensor, points: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:

    norms = torch.linalg.norm(points - median.view(1, -1), dim=1)  # type: ignore

    return (norms * weights).sum()


def compute_geometric_median(
    points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    maxiter: int = 100,
    ftol: float = 1e-20,
    do_log: bool = False,
):
    with torch.no_grad():

        if weights is None:
            weights = torch.ones((points.shape[0],), device=points.device)
        new_weights = weights
        median = weighted_average(points, weights)
        objective_value = geometric_median_objective(median, points, weights)
        if do_log:
            logs = [objective_value]
        else:
            logs = None

        early_termination = False
        pbar = tqdm.tqdm(range(maxiter))
        for _ in pbar:
            prev_obj_value = objective_value

            norms = torch.linalg.norm(points - median.view(1, -1), dim=1)  # type: ignore
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average(points, new_weights)
            objective_value = geometric_median_objective(median, points, weights)

            if logs is not None:
                logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break

            pbar.set_description(f"Objective value: {objective_value:.4f}")

    median = weighted_average(points, new_weights)  # allow autodiff to track it
    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination=(
            "function value converged within tolerance"
            if early_termination
            else "maximum iterations reached"
        ),
        logs=logs,
    )

class FeaturesStats:
    def __init__(self, dim, logger, device):
        self.dim = dim
        self.logger = logger
        self.device = device
        self.reinit()

    def reinit(self):
        self.n_activated = torch.zeros(self.dim, dtype=torch.long, device=self.device)
        self.n = 0
    
    def update(self, inds):
        self.n += inds.shape[0]
        inds = inds.flatten().detach()
        self.n_activated.scatter_add_(0, inds, torch.ones_like(inds))

    def log(self):
        self.logger.logkv('activated', (self.n_activated / self.n + 1e-9).log10().cpu().numpy())
RANK = 0
class Logger:
    def __init__(self, sae_name, **kws):
        self.vals = {}
        self.enabled = (RANK == 0) and not kws.pop("dummy", False)
        self.sae_name = sae_name

    def logkv(self, k, v):
        if self.enabled:
            self.vals[f'{k}'] = v.detach() if isinstance(v, torch.Tensor) else v

        return v

    def dumpkvs(self, step):
        if self.enabled:
            wandb.log(self.vals, step=step)
            self.vals = {}

def init_from_data_(ae, stats_acts_sample):
    ae.pre_bias.data = (
        compute_geometric_median(stats_acts_sample[:32768].float().cpu()).median.to(ae.device).float()
    )

def explained_variance(recons, x):
    # Compute the variance of the difference
    diff = x - recons
    diff_var = torch.var(diff, dim=0, unbiased=False)

    # Compute the variance of the original tensor
    x_var = torch.var(x, dim=0, unbiased=False)

    # Avoid division by zero
    explained_var = 1 - diff_var / (x_var + 1e-8)

    return explained_var.mean()

def train_ksae_on_sd(
    k_sparse_autoencoder: SparseAutoencoder,
    activation_store: SDActivationsStore,
    cfg: SDSAERunnerConfig
):
    batch_size =cfg.batch_size
    total_training_tokens = cfg.total_training_tokens

    logger = Logger(
        sae_name=cfg.sae_name,
        dummy=False,
    ) 

    n_training_steps = 0
    n_training_tokens = 0
    
    optimizer = torch.optim.Adam(k_sparse_autoencoder.parameters(), lr=cfg.lr, eps=cfg.eps, fused=True)

    stats_acts_sample = torch.cat(
        [activation_store.next_batch().cpu() for _ in range(8)], dim=0
    )
    init_from_data_(k_sparse_autoencoder, stats_acts_sample)

    mse_scale = (
        1 / ((stats_acts_sample.float().mean(dim=0) - stats_acts_sample.float()) ** 2).mean()
    )
    mse_scale = mse_scale.item()
    k_sparse_autoencoder.mse_scale = mse_scale
    if cfg.log_to_wandb:
            wandb.init(
                config = vars(cfg),
                project=cfg.wandb_project,
                tags = [
                    str(cfg.batch_size),
                    cfg.block_name,
                    str(cfg.d_in),
                    str(cfg.k),
                    str(cfg.auxk),
                    str(cfg.lr),
                ]
            )
    fstats = FeaturesStats(cfg.d_sae, logger, cfg.device)
    k_sparse_autoencoder.train()
    k_sparse_autoencoder.to(cfg.device)
    pbar = tqdm.tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
            
        optimizer.zero_grad()
        
        sae_in = activation_store.next_batch().to(cfg.device)
        
        sae_out, loss, info = k_sparse_autoencoder(
            sae_in,
        )
        
        n_training_tokens += batch_size

        with torch.no_grad():
            fstats.update(info['inds'])
            bs = sae_in.shape[0]
            logger.logkv('l0', info['l0'])
            logger.logkv('not-activated 1e4', (k_sparse_autoencoder.stats_last_nonzero > 1e4 / bs).mean(dtype=float).item())
            logger.logkv('not-activated 1e6', (k_sparse_autoencoder.stats_last_nonzero > 1e6 / bs).mean(dtype=float).item())
            logger.logkv('not-activated 1e7', (k_sparse_autoencoder.stats_last_nonzero > 1e7 / bs).mean(dtype=float).item())
            logger.logkv('explained variance', explained_variance(sae_out, sae_in))
            logger.logkv('l2_div', (torch.linalg.norm(sae_out, dim=1) / torch.linalg.norm(sae_in, dim=1)).mean())
            logger.logkv('train_recons', info['train_recons'])
            logger.logkv('train_maxk_recons', info['train_maxk_recons'])

            if cfg.log_to_wandb and ((n_training_steps + 1) % cfg.wandb_log_frequency == 0):
                fstats.log()
                fstats.reinit()                

                if "cuda" in str(cfg.device):
                    torch.cuda.empty_cache()
            if ((n_training_steps + 1) % cfg.save_interval == 0):
                k_sparse_autoencoder.save_to_disk(f"{cfg.save_path}/{n_training_steps + 1}")

            pbar.set_description(
                f"{n_training_steps}| MSE Loss {loss.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()

        unit_norm_decoder_(k_sparse_autoencoder)
        unit_norm_decoder_grad_adjustment_(k_sparse_autoencoder)

        optimizer.step()
        n_training_steps += 1
        logger.dumpkvs(n_training_steps)

    return k_sparse_autoencoder

def main(cfg):
    k_sparse_autoencoder = SparseAutoencoder(n_dirs_local=cfg.d_sae, 
                                                d_model=cfg.d_in,
                                                k=cfg.k,
                                                auxk=cfg.auxk,
                                                dead_steps_threshold=cfg.dead_toks_threshold //cfg.batch_size,
                                                auxk_coef = cfg.auxk_coef)

    activations_loader = SDActivationsStore(path_to_chunks=cfg.paths_to_latents,  
                                            block_name=cfg.block_name,
                                            batch_size=cfg.batch_size)

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cfg, name=cfg.run_name)
    
    # train SAE
    k_sparse_autoencoder = train_ksae_on_sd(
        k_sparse_autoencoder, activations_loader, cfg
    )

    k_sparse_autoencoder.save_to_disk(f"{cfg.save_path}/final")    # # save sae to checkpoints folder

    if cfg.log_to_wandb:
        wandb.finish()
        
    return k_sparse_autoencoder


def parse_args():
    parser = argparse.ArgumentParser(description="Parse SDSAERunnerConfig parameters")

    # Add arguments with defaults
    parser.add_argument('--paths_to_latents', type=str, default="I2P", help="Directory for extracted features")
    parser.add_argument('--block_name', type=str, default="text_encoder.text_model.encoder.layers.10.28", help="Block name")
    parser.add_argument('--use_cached_activations', action='store_true', help="Use cached activations", default=True)
    parser.add_argument('--d_in', type=int, default=2048, help="Input dimensionality")
    parser.add_argument('--auxk', type=str, default=256, help='Auxiliary k coefficient (auxk_coef)')

    # SAE Parameters
    parser.add_argument('--expansion_factor', type=int, default=32, help="Expansion factor")
    parser.add_argument('--b_dec_init_method', type=str, default='mean', help="Decoder initialization method")
    parser.add_argument('--k', type=int, default=32, help="Number of clusters")

    # Training Parameters
    parser.add_argument('--lr', type=float, default=0.0004, help="Learning rate")
    parser.add_argument('--lr_scheduler_name', type=str, default='constantwithwarmup', help="Learning rate scheduler name")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size")
    parser.add_argument('--lr_warm_up_steps', type=int, default=500, help="Number of warm-up steps")
    parser.add_argument('--epoch', type=int, default=1000, help="Total training epochs")

    parser.add_argument('--total_training_tokens', type=int, default=83886080, help="Total training tokens")
    parser.add_argument('--dead_feature_threshold', type=float, default=1e-6, help="Dead feature threshold")
    parser.add_argument('--auxk_coef', type=str, default="1/32", help='Auxiliary k coefficient (auxk_coef)')

    # WANDB
    parser.add_argument('--log_to_wandb', action='store_true', default=True, help="Log to WANDB")
    parser.add_argument('--wandb_project', type=str, default='steerers', help="WANDB project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="WANDB entity")
    parser.add_argument('--wandb_log_frequency', type=int, default=500, help="WANDB log frequency")

    # Misc
    parser.add_argument('--device', type=str, default="cuda", help="Device to use (e.g., cuda, cpu)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--checkpoint_path', type=str, default="Checkpoints", help="Checkpoint path")
    parser.add_argument('--dtype', type=str, default="float32", help="Data type (e.g., float32)")
    parser.add_argument('--save_interval', type=int, default=5000, help='Save interval (save_interval)')

    return parser.parse_args()

def args_to_config(args):
    return SDSAERunnerConfig(
        paths_to_latents=args.paths_to_latents,
        block_name=args.block_name,
        use_cached_activations=args.use_cached_activations,
        d_in=args.d_in,
        expansion_factor=args.expansion_factor,
        b_dec_init_method=args.b_dec_init_method,
        k=args.k,
        auxk = args.auxk,
        lr=args.lr,
        lr_scheduler_name=args.lr_scheduler_name,
        batch_size=args.batch_size,
        lr_warm_up_steps=args.lr_warm_up_steps,
        total_training_tokens=args.total_training_tokens,
        dead_feature_threshold=args.dead_feature_threshold,
        log_to_wandb=args.log_to_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_log_frequency=args.wandb_log_frequency,
        device=args.device,
        seed=args.seed,
        save_path_base=args.checkpoint_path,
        dtype=getattr(torch, args.dtype)
    )

if __name__ == "__main__":

    args = parse_args()
    cfg = args_to_config(args)
    print(cfg)

    torch.cuda.empty_cache()
    k_sparse_autoencoder = main(cfg)
