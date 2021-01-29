from torch import optim

def _find_optimizer(optimizer):
    # . . check the pytorch optim directory
    dir_optim = dir(optim)
    # . . create a list of candidate optimizers
    opts = [o.lower() for o in dir_optim]
    # . . if the user provides the optimizer name (string)
    if isinstance(optimizer, str):
        try:
            # . . get the optimizer
            str_index = opts.index(optimizer.lower())    
        except:
            # . . not in the optim list, so not valid
            raise ValueError('No valid optimizers found. The optimizer name must match torch.optim functions.')
        # . . return the found optimizer
        return getattr(optim, dir_optim[str_index])
        # . . if the user provides a torch.optim object, return it back
    elif hasattr(optimizer, 'step') and hasattr(optimizer, 'zero_grad'):
        return optimizer
    else:
        raise ValueError('Invalid optimizer input')    
