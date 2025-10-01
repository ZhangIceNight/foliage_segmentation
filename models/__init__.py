from .dhmamba_pl import DHMamba_pl
from .dhmamba_mse_pl import DHMamba_mse_pl

model_dict = {
    'dhmamba': DHMamba_pl,
    'dhmamba_mse': DHMamba_mse_pl
}

def build_model(config):
    model_type = config.model.model_type
    model_class = model_dict[model_type]
    return model_class(config)
