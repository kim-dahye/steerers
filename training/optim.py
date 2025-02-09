
import math
from typing import Optional
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(
    scheduler_name: Optional[str], optimizer: optim.Optimizer, **kwargs
):

    def get_warmup_lambda(warm_up_steps, training_steps):
        def lr_lambda(steps):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                return (training_steps - steps) / (
                    training_steps - warm_up_steps
                )

        return lr_lambda

    # heavily derived from hugging face although copilot helped.
    def get_warmup_cosine_lambda(warm_up_steps, training_steps, lr_end):
        def lr_lambda(steps):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                progress = (steps - warm_up_steps) / (
                    training_steps - warm_up_steps
                )
                return lr_end + 0.5 * (1 - lr_end) * (
                    1 + math.cos(math.pi * progress)
                )

        return lr_lambda
    
    if scheduler_name is None or scheduler_name.lower() == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)
    elif scheduler_name.lower() == "constantwithwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / warm_up_steps),
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")