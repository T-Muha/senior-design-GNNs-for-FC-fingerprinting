
import os
import numpy as np
import datetime

def make_directories(config):
    """
    Creates the directories that hold the results of the run.
    """

    # Create results directory if in Colab environment
    try:
        import google.colab
        os.mkdir('results')
    except:
        pass

    save_name = 'avg_runs_' + datetime.now().strftime("%m_%d_%H%M%S")
    save_dir = 'results/' + save_name
    os.mkdir(save_dir)
    os.makedirs(save_dir+'/plots/accuracies')
    os.makedirs(save_dir+'/plots/AUC')
    os.makedirs(save_dir+'/plots/loss')
    os.makedirs(save_dir+'/plots/loss/at_step')
    os.makedirs(save_dir+'/plots/margin')
    os.makedirs(save_dir+'/plots/grad_norm')
    os.makedirs(save_dir+'/plots/confusion')
    os.makedirs(save_dir+'/classifications')
    os.makedirs(save_dir+'/classifications/only_pos')
    os.makedirs(save_dir+'/classifications/only_neg')

    if config['train']['save_params']:
        os.makedirs(save_dir+'/plots/parameters')

    return save_name, save_dir


def record_params(model, param_dict, epoch):
    """Record the parameters of this model in the parameter dictionary."""

    epoch_params = np.array([])
    for name, param in model.named_parameters():
        np_param = param.clone().cpu().detach().numpy().flatten().astype(np.float32)
        epoch_params = np.append(epoch_params, np_param)
        np_param = np.expand_dims(np_param, axis=1)
        if epoch == 0:
            param_dict[name] = np_param
        else:
            param_dict[name] = np.append(param_dict[name], np_param, axis=1)

    return param_dict