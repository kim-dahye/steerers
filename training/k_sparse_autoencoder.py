import os
import json
import torch
from torch import  nn

class SparseAutoencoder(nn.Module):

    def __init__(
        self,
        n_dirs_local: int,
        d_model: int,
        k: int,
        auxk: int | None,
        dead_steps_threshold: int,
        auxk_coef: float
    ):
        super().__init__()
        self.n_dirs_local = n_dirs_local
        self.d_model = d_model
        self.k = k
        self.auxk = auxk
        self.dead_steps_threshold = dead_steps_threshold
        self.auxk_coef = auxk_coef
        self.encoder = nn.Linear(d_model, n_dirs_local, bias=False)
        self.decoder = nn.Linear(n_dirs_local, d_model, bias=False)

        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs_local))

        self.stats_last_nostats_last_nonzeronzero: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dirs_local, dtype=torch.long))

        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > dead_steps_threshold
            x.data *= dead_mask  # inplace to save memory
            return x

        self.auxk_mask_fn = auxk_mask_fn
        ## initialization

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

        # store decoder in column major layout for kernel
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T
        self.mse_scale = 1
        unit_norm_decoder_(self)

    def save_to_disk(self, path: str):
        PATH_TO_CFG = 'config.json'
        PATH_TO_WEIGHTS = 'state_dict.pth'

        cfg = {
            "n_dirs_local": self.n_dirs_local,
            "d_model": self.d_model,
            "k": self.k,
            "auxk": self.auxk,
            "dead_steps_threshold": self.dead_steps_threshold,
            "auxk_coef": self.auxk_coef
        }

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, PATH_TO_CFG), 'w') as f:
            json.dump(cfg, f)
        
        torch.save({
            "state_dict": self.state_dict(),
        }, os.path.join(path, PATH_TO_WEIGHTS))

    @classmethod
    def load_from_disk(cls, path: str):
        PATH_TO_CFG = 'config.json'
        PATH_TO_WEIGHTS = 'state_dict.pth'

        with open(os.path.join(path, PATH_TO_CFG), 'r') as f:
            cfg = json.load(f)

        ae = cls(
            n_dirs_local=cfg["n_dirs_local"],
            d_model=cfg["d_model"],
            k=cfg["k"],
            auxk=cfg["auxk"],
            dead_steps_threshold=cfg["dead_steps_threshold"],
            auxk_coef = cfg["auxk_coef"] if "auxk_coef" in cfg else 1/32
        )

        state_dict = torch.load(os.path.join(path, PATH_TO_WEIGHTS))["state_dict"]
        ae.load_state_dict(state_dict)

        return ae

    @property
    def n_dirs(self):
        return self.n_dirs_local

    def encode(self, x):
        x = x - self.pre_bias
        latents_pre_act = self.encoder(x) + self.latent_bias

        vals, inds = torch.topk(
            latents_pre_act,
            k=self.k,
            dim=-1
        )   
        
        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(-1, inds, torch.relu(vals))

        return latents

    def encode_with_k(self, x, k):
        x = x - self.pre_bias
        latents_pre_act = self.encoder(x) + self.latent_bias

        vals, inds = torch.topk(
            latents_pre_act,
            k=k,
            dim=-1
        )   
        
        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(-1, inds, torch.relu(vals))

        return latents

    def encode_without_topk(self, x):
        x = x - self.pre_bias
        latents_pre_act = torch.relu(self.encoder(x) + self.latent_bias)
        return latents_pre_act


    def forward(self, x):
        x = x - self.pre_bias
        latents_pre_act = self.encoder(x) + self.latent_bias
        l0 = (latents_pre_act  > 0).float().sum(-1).mean()
        vals, inds = torch.topk(
            latents_pre_act,
            k=self.k,
            dim=-1
        )
        with torch.no_grad():  # Disable gradients for statistics
            ## set num nonzero stat ##
            tmp = torch.zeros_like(self.stats_last_nonzero)
            tmp.scatter_add_(
                0,
                inds.reshape(-1),
                (vals > 1e-3).to(tmp.dtype).reshape(-1),
            )
            self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
            self.stats_last_nonzero += 1

            del tmp
        ## auxk
        if self.auxk is not None:  # for auxk
            auxk_vals, auxk_inds = torch.topk(
                self.auxk_mask_fn(latents_pre_act),
                k=self.auxk,
                dim=-1
            )
        else:
            auxk_inds = None
            auxk_vals = None

        ## end auxk

        vals = torch.relu(vals)
        if auxk_vals is not None:
            auxk_vals = torch.relu(auxk_vals)

        rows, cols = latents_pre_act.size()
        row_indices = torch.arange(rows).unsqueeze(1).expand(-1, self.k).reshape(-1)
        vals = vals.reshape(-1)
        inds = inds.reshape(-1)

        indices = torch.stack([row_indices.to(inds.device), inds])

        sparse_tensor = torch.sparse_coo_tensor(indices, vals, torch.Size([rows, cols]))

        recons = torch.sparse.mm(sparse_tensor, self.decoder.weight.T) + self.pre_bias

        mse_loss = self.mse_scale * self.mse(recons, x)

        ## Calculate AuxK loss if applicable
        if auxk_vals is not None:
            auxk_recons = self.decode_sparse(auxk_inds, auxk_vals)
            auxk_loss =self.auxk_coef * self.normalized_mse(auxk_recons, x - recons.detach() + self.pre_bias.detach()).nan_to_num(0)
        else:
            auxk_loss = 0.0

        total_loss = mse_loss + auxk_loss

        return recons, total_loss, {
            "inds": inds,
            "vals": vals,
            "auxk_inds": auxk_inds,
            "auxk_vals": auxk_vals,
            "l0": l0,
            "train_recons": mse_loss,
            "train_maxk_recons": auxk_loss
        }

    
    def decode_sparse(self, inds, vals):
        rows, cols = inds.shape[0], self.n_dirs
        
        row_indices = torch.arange(rows).unsqueeze(1).expand(-1, inds.shape[1]).reshape(-1)
        vals = vals.reshape(-1)
        inds = inds.reshape(-1)

        indices = torch.stack([row_indices.to(inds.device), inds])

        sparse_tensor = torch.sparse_coo_tensor(indices, vals, torch.Size([rows, cols]))

        recons = torch.sparse.mm(sparse_tensor, self.decoder.weight.T) + self.pre_bias
        return recons

    @property
    def device(self):
        return next(self.parameters()).device

    def mse(self, recons, x):
        # return ((recons - x) ** 2).sum(dim=-1).mean()
        return ((recons - x) ** 2).mean()

    def normalized_mse(self, recon: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        # only used for auxk
        xs_mu = xs.mean(dim=0)

        loss = self.mse(recon, xs) / self.mse(
            xs_mu[None, :].broadcast_to(xs.shape), xs
        )

        return loss

def unit_norm_decoder_(autoencoder: SparseAutoencoder) -> None:

    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(dim=0)


def unit_norm_decoder_grad_adjustment_(autoencoder) -> None:

    assert autoencoder.decoder.weight.grad is not None

    autoencoder.decoder.weight.grad +=\
        torch.einsum("bn,bn->n", autoencoder.decoder.weight.data, autoencoder.decoder.weight.grad) *\
        autoencoder.decoder.weight.data * -1