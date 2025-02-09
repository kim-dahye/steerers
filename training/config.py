from dataclasses import dataclass, field
from typing import Optional
import torch
import datetime

@dataclass
class SDSAERunnerConfig():

    image_size: int = 512,
    num_sampling_steps: int = 25,
    vae: str = "mse"
    model_name: str = None
    model_name_proc: str= None
    timestep: int = 0
    module_name: str = "mid_block"
    paths_to_latents: str = None
    layer_name:str = None
    block_layer: int = 10
    block_name: str = "text_encoder.text_model.encoder.layers.10.28"
    use_cached_activations: bool = False
    block_name :str = 'mid_block'
    image_key: str = 'image'

    # SAE Parameters
    d_in: int = 768
    k: int = 32
    auxk_coef: float = 1 / 32
    auxk: int = 32
    # Activation Store Parameters
    epoch:int = 1000
    total_training_tokens: int = 2_000_000
    eps: float = 6.25e-10

    # SAE Parameters
    b_dec_init_method: str = "mean"
    expansion_factor: int = 4
    from_pretrained_path: Optional[str] = None

    # Training Parameters
    lr: float = 3e-4
    lr_scheduler_name: str = "constant"  
    lr_warm_up_steps: int = 500
    batch_size: int = 4096
    sae_batch_size: int = 1024,
    dead_feature_threshold: float = 1e-8
    dead_toks_threshold: int = 10_000_000
    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "steerers"
    wandb_entity: str = None
    wandb_log_frequency: int = 10

    
    # Misc
    device: str = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32
    save_path_base: str = "checkpoints"
    max_batch_size: int = 32
    ct: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    save_interval: int = 5000

    def __post_init__(self):
        
        self.d_sae = self.d_in * self.expansion_factor

        self.run_name = f"{self.block_name}_k{self.k}_hidden{self.d_sae}_auxk{self.auxk}_bs{self.batch_size}_lr{self.lr}"
        self.checkpoint_path = f"{self.save_path_base}/{self.run_name}_{self.ct}"

        if self.b_dec_init_method not in ["mean"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.b_dec_init_method}"
            )

        self.device = torch.device(self.device)

        print(
            f"Run name: {self.d_sae}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"
        )
        # Print out some useful info:

        total_training_steps = self.total_training_tokens // self.batch_size
        print(f"Total training steps: {total_training_steps}")

        total_wandb_updates = total_training_steps // self.wandb_log_frequency
        print(f"Total wandb updates: {total_wandb_updates}")
        
    @property
    def sae_name(self) -> str:
        """Returns the name of the SAE model based on key parameters."""
        return f"{self.block_name}_k{self.k}_hidden{self.d_sae}_auxk{self.auxk}_bs{self.batch_size}_lr{self.lr}"
    
    @property
    def save_path(self) -> str:
        """Returns the path where the SAE model will be saved."""
        return self.checkpoint_path

    def __getitem__(self, key):
        """Allows subscripting the config object like a dictionary."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} does not exist in SDSAERunnerConfig.")

