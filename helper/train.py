import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import tqdm
import numpy as np
import pdb
import sklearn.metrics
import helper.utils as utils


def train_model(train_data, dev_data, model, gen, args):
    if args["cuda"]:
        model = model.cuda()
        gen = gen.cuda()
    
    args["lr"] = args["init_lr"]
    
    optimizer = utils.get_optimizer([model, gen], args)
    
    num_epoch_sans_improvement = 0
    epoch_stats = utils.init_metrics_dictionary(modes=['train', 'dev'])
    step = 0
    tuning_key = "dev_{}".format(args["tuning_metric"])
    best_epoch_func = min if tuning_key == 'dev_loss' else max
    
    train_loader = utils.get_train_loader(train_data, args)
    dev_loader = utils.get_dev_loader(dev_data, args)
    
    for epoch in range(1, args["epochs"] + 1):
        print("-------------\nEpoch {}:\n".format(epoch))
        for mode, dataset, loader in [
            ('Train', train_data, train_loader),
            ('Dev', dev_data, dev_loader)
        ]:
            train_model = mode == 'Train'
            print('{}'.format(mode))
            key_prefix = mode.lower()
            epoch_details, step, _, _, _, _ = run_epoch(
                data_loader=loader,
                train_model=train_model,
                model=model,
                gen=gen,
                optimizer=optimizer,
                step=step,
                args=args)

            epoch_stats, log_statement = utils.collate_epoch_stat(
                epoch_stats, epoch_details, key_prefix, args
            )

            # Log  performance
            print(log_statement)
        
        # Save model if beats best dev
        best_func = min if args["tuning_metric"] == 'loss' else max
        if (
            best_func(epoch_stats[tuning_key]) ==
            epoch_stats[tuning_key][-1]
            or epoch % 5 == 0
        ):
            print('Saving')
            num_epoch_sans_improvement = 0
            if not os.path.isdir(args["save_dir"]):
                os.makedirs(args["save_dir"])
            # Subtract one because epoch is 1-indexed
            # and arr is 0-indexed
            epoch_stats['best_epoch'] = epoch - 1
            torch.save(model.state_dict(), args["model_path"])
            torch.save(gen.state_dict(), args["gen_path"])
        else:
            num_epoch_sans_improvement += 1
            
        if not train_model:
            print('---- Best Dev {} is {:.4f} at epoch {}'.format(
                args["tuning_metric"],
                epoch_stats[tuning_key][epoch_stats['best_epoch']],
                epoch_stats['best_epoch'] + 1))
        
        if num_epoch_sans_improvement >= args["patience"]:
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0
            model.cpu()
            gen.cpu()
            #model = torch.load(args["model_path"])
            model.load_state_dict(torch.load(args["model_path"]))
            gen.load_state_dict(torch.load(args['gen_path']))
            #gen = torch.load(args["gen_path"])

            if args["cuda"]:
                model = model.cuda()
                gen = gen.cuda()
            args["lr"] *= .5
            optimizer = utils.get_optimizer([model, gen], args)
    
    if os.path.exists(args["model_path"]):
        model.cpu()
        gen.cpu()
        
        # model = torch.load(args["model_path"])
        # gen = torch.load(args["gen_path"])
        
        # model.load_state_dict(args["model_path"])
        # gen.load_state_dict(args['gen_path'])
        
        model.load_state_dict(torch.load(args["model_path"]))
        gen.load_state_dict(torch.load(args['gen_path']))

    return epoch_stats, model, gen


def gumbel_softmax(logits, temperature):
    # g = -log(-log(u)), u ~ U(0, 1)

    noise = torch.rand_like(logits)

    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()

    # # uncomment this for GPU use
    # if torch.cuda.is_available():
    #     noise = noise.cuda()
        
    x = (logits + noise) / temperature
    probs = F.softmax(x, dim=-1)

    return probs


def run_epoch(data_loader, train_model, model, gen, optimizer, step, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    eval_model = not train_model
    data_iter = iter(data_loader)

    losses = []
    obj_losses = []
    k_selection_losses = []
    #k_continuity_losses = []
    preds = []
    golds = []
    losses = []
    rationales = []

    if train_model:
        model.train()
        gen.train()
    else:
        gen.eval()
        model.eval()
    
    num_batches_per_epoch = len(data_iter)
    #if train_model:
    #    num_batches_per_epoch = min(len(data_iter), 10000)
    
    for _ in tqdm.tqdm(range(num_batches_per_epoch)):
        batch = next(data_iter)
        if train_model:
            step += 1
            if step % 100 == 0 or args["debug_mode"]:
                args["temperature"] = (
                    max(np.exp((step+1) *-1* args["gumbel_decay"]), .05)
                )
        
        x_indx = batch['inputs']
        y = batch['labels'].type(torch.float32)
        input_mask = batch['input_mask'].type(torch.float32)
        
        if args["cuda"]:
            x_indx, y, input_mask = x_indx.cuda(), y.cuda(), input_mask.cuda()

        if train_model:
            optimizer.zero_grad()

        mask = gen(x_indx, input_mask)
        
        output = model(x_indx, mask)
        
        loss, require_back_grad = get_loss(output, y.unsqueeze(1), args)
        obj_loss = loss.detach().item()
        
        selection_cost = gen_loss(mask, output)
        loss += args["selection_lambda"] * selection_cost
        #loss += args["continuity_lambda"] * continuity_cost
        
        if train_model:
            if require_back_grad:
                loss.backward()
                optimizer.step()
        
        k_selection_losses.append( utils.tensor_to_numpy(selection_cost))
        #k_continuity_losses.append( utils.tensor_to_numpy(continuity_cost))
        
        obj_losses.append(obj_loss)
        losses.append( utils.tensor_to_numpy(loss) )
        preds.extend(output.detach().cpu().numpy())
        
        #texts.extend(text)
        #rationales.extend(utils.get_rationales(mask, text))
        
        golds.extend(batch['labels'].numpy())

    epoch_metrics = utils.get_metrics(preds, golds, args)

    epoch_stat = {
        'loss' : np.mean(losses),
        'obj_loss': np.mean(obj_losses)
    }

    for metric_k in epoch_metrics.keys():
        epoch_stat[metric_k] = epoch_metrics[metric_k]

    epoch_stat['k_selection_loss'] = np.mean(k_selection_losses)
    #epoch_stat['k_continuity_loss'] = np.mean(k_continuity_losses)

    return epoch_stat, step, losses, preds, golds, rationales


def get_loss(logit, y, args, thres = 0.0001):
    if args["objective"] == 'cross_entropy':
        loss = F.cross_entropy(logit, y)
    elif args["objective"] == 'mse':
        require_back_grad = False
        if args["cuda"]:
            loss = torch.tensor(0.0).cuda()
        else:
            loss = torch.tensor(0.0)
        for bth in range(logit.shape[0]):
            if torch.abs(logit[bth, 0]) >= thres:
                require_back_grad = True
                loss += torch.abs(logit[bth, 0] - y[bth, 0])
            else:
                loss += torch.abs(y[bth, 0])
        loss = loss/logit.shape[0]
    else:
        raise Exception(
            "Objective {} not supported!".format(args["objective"]))
    return loss, require_back_grad


def gen_loss(mask, output, thres = 0.0001):
    '''
    Compute the generator specific costs, i.e selection cost,
    continuity cost, and global vocab cost
    '''
    selection_cost = torch.tensor(0.0)
    for bth in range(output.shape[0]):
        if torch.abs(output[bth, 0]) >= thres:
            selection_cost += torch.sum(mask[bth, :])
    #l_padded_mask = torch.cat( [mask[:,0].unsqueeze(1), mask] , dim=1)
    #r_padded_mask = torch.cat( [mask, mask[:,-1].unsqueeze(1)] , dim=1)
    #continuity_cost = torch.mean( torch.sum( torch.abs( l_padded_mask - r_padded_mask ) , dim=1) )
    return selection_cost / output.shape[0]


def test_model(test_data, model, gen, args):
    '''
    Run model on test data, and return loss, accuracy.
    '''
    if args["cuda"]:
        model = model.cuda()
        gen = gen.cuda()

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        drop_last=False)

    test_stats = utils.init_metrics_dictionary(modes=['test'])

    mode = 'Test'
    train_model = False
    key_prefix = mode.lower()
    print("-------------\nTest")
    epoch_details, _, losses, preds, golds, rationales = run_epoch(
        data_loader=test_loader,
        train_model=train_model,
        model=model,
        gen=gen,
        optimizer=None,
        step=None,
        args=args)

    test_stats, log_statement = utils.collate_epoch_stat(
        test_stats, epoch_details, 'test', args
    )
    test_stats['losses'] = losses
    test_stats['preds'] = preds
    test_stats['golds'] = golds
    test_stats['rationales'] = rationales

    print(log_statement)

    return test_stats
