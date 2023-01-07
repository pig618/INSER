import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
import sklearn.metrics
import numpy as np


def get_pretained_glove(word2idx, word2embedding, embedding_dim):
    """
    Construct a numpy embedding matrix. 
    The column number indicates the word index.
    For the words do not appear in pretrained embeddings, 
    we use random embeddings.
    Inputs:
        word2idx -- a dictionary, key -- word, value -- word index
        fpath -- the path of pretrained embedding.
    Outputs:
        embedding_matrix -- an ordered numpy array, 
                            shape -- (embedding_dim, len(word2idx))
    """

    embedding_matrix = np.random.randn(embedding_dim, len(word2idx))

    # replace the embedding matrix by pretrained embedding
    counter = 0
    for word, index in word2idx.items():
        if word in word2embedding:
            embedding_matrix[:, index] = word2embedding[word]
            counter += 1

    # replace the embedding to all zeros for <pad>
    embedding_matrix[:, word2idx["<pad>"]] = np.zeros(embedding_dim)
    print("%d out of %d words are covered by the pre-trained embedding." %
          (counter, len(word2idx)))

    return embedding_matrix


def get_optimizer(models, args):
    '''
    -models: List of models (such as Generator, classif, memory, etc)
    -args: experiment level config
    returns: torch optimizer over models
    '''
    params = []
    for model in models:
        params.extend(
            [param for param in model.parameters() if param.requires_grad]
        )
    return torch.optim.Adam(
        params, lr=args["lr"],  weight_decay=args["weight_decay"]
    )


def gumbel_softmax(inp, temperature):
    noise = torch.rand(inp.size())
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = autograd.Variable(noise)
    
    if torch.cuda.is_available():
        noise = noise.cuda()
    x = (inp + noise) / temperature
    x = F.softmax(x.view(-1,  x.size()[-1]), dim=-1)
    return x.view_as(inp)


def get_hard_mask(z, return_ind=False):
    '''
        -z: torch Tensor where each element probablity of element
        being selected
        -args: experiment level config
        returns: A torch variable that is binary mask of z >= .5
    '''
    max_z, ind = torch.max(z, dim=-1)
    if return_ind:
        del z
        return ind
    masked = torch.ge(z, max_z.unsqueeze(-1)).float()
    del z
    return masked


def get_metrics(preds, golds, args):
    metrics = {}

    if args["objective"] in ['cross_entropy', 'margin']:
        metrics['accuracy'] = sklearn.metrics.accuracy_score(
            y_true=golds, y_pred=preds
        )
        metrics['confusion_matrix'] = sklearn.metrics.confusion_matrix(
            y_true=golds,y_pred=preds
        )
        metrics['precision'] = sklearn.metrics.precision_score(
            y_true=golds, y_pred=preds, average="weighted"
        )
        metrics['recall'] = sklearn.metrics.recall_score(
            y_true=golds,y_pred=preds, average="weighted"
        )
        metrics['f1'] = sklearn.metrics.f1_score(
            y_true=golds,y_pred=preds, average="weighted"
        )

        metrics['mse'] = "NA"

    elif args["objective"] == 'mse':
        metrics['mse'] = sklearn.metrics.mean_squared_error(
            y_true=golds, y_pred=preds
        )
        metrics['confusion_matrix'] = "NA"
        metrics['accuracy'] = "NA"
        metrics['precision'] = "NA"
        metrics['recall'] = "NA"
        metrics['f1'] = 'NA'

    return metrics


def init_metrics_dictionary(modes):
    '''
    Create dictionary with empty array for each metric in each mode
    '''
    epoch_stats = {}
    metrics = [
        'loss', 'obj_loss', 'k_selection_loss', 'k_continuity_loss',
        'accuracy', 'precision', 'recall', 'f1',
        'confusion_matrix', 'mse'
    ]
    for metric in metrics:
        for mode in modes:
            key = "{}_{}".format(mode, metric)
            epoch_stats[key] = []

    return epoch_stats


def get_train_loader(train_data, args):
    if args["class_balance"]:
        sampler = data.sampler.WeightedRandomSampler(
                weights=train_data.weights,
                num_samples=len(train_data),
                replacement=True)
        train_loader = data.DataLoader(
                train_data,
                num_workers= args["num_workers"],
                sampler=sampler,
                batch_size= args["batch_size"])
    else:
        train_loader = data.DataLoader(
            train_data,
            batch_size=args["batch_size"],
            shuffle=True,
            num_workers=args["num_workers"],
            drop_last=False)

    return train_loader


def get_dev_loader(dev_data, args):
    dev_loader = data.DataLoader(
        dev_data,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        drop_last=False)
    return dev_loader


def get_x_indx(batch, args, eval_model):
    x_indx = autograd.Variable(batch['x'], volatile=eval_model)
    return x_indx


def tensor_to_numpy(tensor):
    return tensor.item()


def get_rationales(mask, text):
    if mask is None:
        return text
    masked_text = []
    for i, t in enumerate(text):
        sample_mask = list(mask.data[i])
        original_words = t.split()
        words = [
            w if m  > .5 else "_" for w,m in zip(original_words, sample_mask)
        ]
        masked_sample = " ".join(words)
        masked_text.append(masked_sample)
    return masked_text


def collate_epoch_stat(stat_dict, epoch_details, mode, args):
    '''
        Update stat_dict with details from epoch_details and create
        log statement
        - stat_dict: a dictionary of statistics lists to update
        - epoch_details: list of statistics for a given epoch
        - mode: train, dev or test
        - args: model run configuration
        returns:
        -stat_dict: updated stat_dict with epoch details
        -log_statement: log statement sumarizing new epoch
    '''
    log_statement_details = ''
    for metric in epoch_details:
        loss = epoch_details[metric]
        stat_dict['{}_{}'.format(mode, metric)].append(loss)

        log_statement_details += ' -{}: {}'.format(metric, loss)

    log_statement = '\n {} - {}\n--'.format(
        args["objective"], log_statement_details )

    return stat_dict, log_statement


def get_gen_path(model_path):
    '''
        Get generator path
        -model_path: path of encoder model
        returns: path of generator
    '''
    return '{}.gen'.format(model_path)














