from typing import Callable, Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LinearLR

from shortcutfm.config import MyleSchedulerConfig, LinearSchedulerConfig
from shortcutfm.nn import MyleLR


class SchedulerFactory:
    """Factory for time schedulers"""
    registry: dict[str, Any] = {}

    @classmethod
    def get_scheduler(cls, name: str, optimizer: Optimizer, config) -> LRScheduler:
        if name not in cls.registry:
            raise ValueError(f"Unknown scheduler {name}")

        scheduler_class = cls.registry[name]
        return scheduler_class(optimizer, config)

    @classmethod
    def register(cls, name: str) -> Callable:
        """decorator for adding schedulers to the registry"""

        def inner_wrapper(wrapped_class: LRScheduler) -> LRScheduler:
            assert name not in cls.registry, f"{name} already registered"

            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper


@SchedulerFactory.register("myle")
def get_myle_scheduler(optimizer: Optimizer, cfg: MyleSchedulerConfig) -> LRScheduler:
    return MyleLR(
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        start_lr=cfg.start_lr,
    )


@SchedulerFactory.register("linear")
def get_linear_scheduler(optimizer: Optimizer, cfg: LinearSchedulerConfig) -> LRScheduler:
    return LinearLR(
        optimizer,
        start_factor=cfg['start_factor'],
        end_factor=cfg['end_factor'],
        total_iters=cfg['total_steps'],
    )
