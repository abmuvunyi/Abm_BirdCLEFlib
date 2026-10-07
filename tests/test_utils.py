import pytest
import torch
import torch.nn as nn
from utils.create_optimizer import create_optimizer
from utils.create_lr_scheduler import create_scheduler

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

@pytest.fixture
def dummy_model():
    return DummyModel()

def test_create_optimizer_adam(dummy_model):
    opt_config = {
        'name': 'adam',
        'lr': 0.001,
        'weight_decay': 0.01,
        'adam': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'amsgrad': False
        }
    }
    optimizer = create_optimizer(dummy_model, opt_config)
    assert isinstance(optimizer, torch.optim.Adam)

def test_create_scheduler_reduce_lr(dummy_model):
    opt_config = {
        'name': 'adam',
        'lr': 0.001,
        'weight_decay': 0.01,
        'adam': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'amsgrad': False
        }
    }
    optimizer = create_optimizer(dummy_model, opt_config)

    scheduler_config = {
        'name': 'reduce_lr',
        'reduce_lr': {
            'mode': 'min',
            'factor': 0.1,
            'patience': 10
        }
    }

    class DummyLoader:
        def __len__(self):
            return 10

    scheduler = create_scheduler(optimizer, scheduler_config, DummyLoader())
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
