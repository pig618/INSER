import torch
import pdb


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def init_args():
    args = {}
    
    #setup
    args["train"] = False
    args["test"] = False
    # device
    args["cuda"] = False
    args["num_gpus"] = 0
    args["debug_mode"] = False
    args["class_balance"] = False
    # learning
    args["objective"] = 'mse'
    args["aspect"] = 'overall'
    args["init_lr"] = 0.001
    args["epochs"] = 100
    args["batch_size"] = 64
    args["patience"] = 10
    args["tuning_metric"] = 'loss'
    args["weight_decay"] = 1e-3
    #paths
    args["save_dir"] = "models"
    args["results_path"] = ""
    args["model_path"] = "models/rnn_market_model.pt"
    args["gen_path"] = "models/rnn_market_gen.pt"
    args["opt_path"] = "models/rnn_market_opt.pt"
    args["snapshot"] = None
    # data loading
    args["num_workers"] = 2
    # generator
    args["hidden_dim"] = 100
    args["num_layers"] = 2
    args["dropout"] = 0.1
    args["gen_bidirectional"] = True
    # encoder
    args["hidden_dim_enc"] = 100
    args["num_layers_enc"] = 2
    args["dropout_enc"] = 0.1
    args["gen_bidirectional_enc"] = True
    # data
    args['glove_path'] = 'data/glove.6B.100d.txt'
    args["embedding_dim"] = 100
    args["embedding_length"] = 300
    # gumbel
    args["temperature"] = 1
    args["gumbel_decay"] = 1e-5
    # rationale
    args["get_rationales"] = True
    args["selection_lambda"] = 0.00002
    args["continuity_lambda"] = 0.00002
    args["num_class"] = 2
    # tagging task support. Note, does not support rationales x tagging
    args["use_as_tagger"] = False
    args["tag_lambda"] = 0.5
    # update args and print
    if args["objective"] == 'mse':
        args["num_class"] = 1

    return dotdict(args)


def update_args(params, args):
    for key, val in params.items():
        if key in args:
            args[key] = val


def get_gen_path(model_path):
    '''
        -model_path: path of encoder model
        returns: path of generator
    '''
    return '{}.gen'.format(model_path)







