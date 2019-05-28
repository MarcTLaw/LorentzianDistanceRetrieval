#!/usr/bin/env python3
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch as th
import numpy as np
import logging
import argparse
from torch.autograd import Variable
from collections import defaultdict as ddict
import torch.multiprocessing as mp
import lorentzian_model as model 
import lorentzian_train_with_normalized_rank as train 
import rsgd
from data import slurp
from rsgd import RiemannianSGD
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
import gc
import sys



def ranking(types, model, distfn, data, order_rank = None):
    lt = th.from_numpy(model.embedding())
    embedding = Variable(lt, volatile=True)
    ranks = []
    ap_scores = []
    norms = []
    ordered_ranks = []
    for s, s_types in types.items():
        if order_rank is not None:
            lts = lt[s]            
            ltsnorm = th.sum(lts * lts, dim=-1)
            
            norms.append(float(ltsnorm[0]))
            ordered_ranks.append(order_rank[data.objects[s]])
            
        s_e = Variable(lt[s].expand_as(embedding), volatile=True)
        _dists = model.dist()(s_e, embedding).data.cpu().numpy().flatten()
        _dists[s] = 1e+12
        _labels = np.zeros(embedding.size(0))
        _dists_masked = _dists.copy()
        _ranks = []
        for o in s_types:
            _dists_masked[o] = np.Inf
            _labels[o] = 1
        ap_scores.append(average_precision_score(_labels, -_dists))
        for o in s_types:
            d = _dists_masked.copy()
            d[o] = _dists[o]
            r = np.argsort(d)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks
    rho = None
    if order_rank is not None:
        rho, pval = spearmanr(ordered_ranks,norms)

    return np.mean(ranks), np.mean(ap_scores), rho


def control(queue, log, types, data, fout, distfn, nepochs, processes, dataset_name = "_", order_rank = None):
    min_rank = (np.Inf, -1)
    max_map = (0, -1)
    max_rho = (-2, -1)
    while True:
        gc.collect()
        msg = queue.get()
        if msg is None:
            for p in processes:
                p.terminate()
            break
        else:
            epoch, elapsed, loss, model = msg
        if model is not None:
            # save model to fout
            th.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'objects': data.objects,
            }, fout)
            # compute embedding quality
            if True:
                mrank, mAP, rho = ranking(types, model, distfn, data, order_rank)
            else:
                mrank = np.Inf
                mAP = 0
                rho = None
            if mrank < min_rank[0]:
                min_rank = (mrank, epoch)
            if mAP > max_map[0]:
                max_map = (mAP, epoch)
            if rho is not None:
                if rho > max_rho[0]:
                    max_rho = (rho, epoch)
            else:
                rho = -2
            log.info(
                ('eval: {'
                 '"epoch": %d, '
                 '"elapsed": %.2f, '
                 '"loss": %.3f, '
                 '"mean_rank": %.2f, '
                 '"mAP": %.4f, '
                 '"rho": %.4f, '
                 '"best_rank": %.2f, '
                 '"best_mAP": %.4f,'
                 '"best_rho": %.4f,') % (
                     epoch, elapsed, loss, mrank, mAP, rho, min_rank[0], max_map[0], max_rho[0])
            )
            th.save('"epoch": %d "mAP": %g, "mAP epoch": %d, "mean rank": %g, "mean rank epoch": %d, "rho": %g, "max rho": %d\n' % (epoch, max_map[0], max_map[1], min_rank[0], min_rank[1], max_rho[0], max_rho[1]), "logs/with_rank_current_results_%s_%d.txt" % (dataset_name, epoch))
        else:
            log.info(f'json_log: {{"epoch": {epoch}, "loss": {loss}, "elapsed": {elapsed}}}')

        if epoch >= nepochs - 1:
            log.info(
                ('results: {'
                 '"mAP": %g, '
                 '"mAP epoch": %d, '
                 '"mean rank": %g, '
                 '"mean rank epoch": %d'
                 '"rho": %g, '
                 '"rho epoch": %d, '
                 '}') % (
                     max_map[0], max_map[1], min_rank[0], min_rank[1], max_rho[0], max_rho[1])
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Poincare Embeddings')
    parser.add_argument('-dim', help='Embedding dimension', type=int)
    parser.add_argument('-dset', help='Dataset to embed', type=str)
    parser.add_argument('-fout', help='Filename where to store model', type=str)
    parser.add_argument('-rin', help='Filename with ranks', type=str)
    parser.add_argument('-distfn', help='Distance function', type=str)
    parser.add_argument('-lr', help='Learning rate', type=float)
    parser.add_argument('-epochs', help='Number of epochs', type=int, default=200)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=50)
    parser.add_argument('-beta', help='Beta', type=float, default=0.01)
    parser.add_argument('-lambdaparameter', help='Regularization parameter lambda', type=float, default=0.0)
    parser.add_argument('-eta', help='Number of examples randomly chosen', type=int, default=150)
    parser.add_argument('-negs', help='Number of negatives', type=int, default=20)
    parser.add_argument('-nproc', help='Number of processes', type=int, default=5)
    parser.add_argument('-ndproc', help='Number of data loading processes', type=int, default=2)
    parser.add_argument('-eval_each', help='Run evaluation each n-th epoch', type=int, default=10)
    parser.add_argument('-burnin', help='Duration of burn in', type=int, default=20)
    parser.add_argument('-debug', help='Print debug output', action='store_true', default=False)
    opt = parser.parse_args()

    th.set_default_tensor_type('torch.FloatTensor')
    if opt.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    log = logging.getLogger('lorentzian-icml19')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)
    idx, objects = slurp(opt.dset)
    dataset_name = (opt.dset.split(".")[0]).replace("/", "_")
    
    order_rank = {}
    order_rank_file = open(opt.rin, "r")

    for l in order_rank_file:
        l_table = l.split(" : ")
        order_rank[l_table[0]] = float(l_table[1])
    order_rank_file.close()
    print("Successfully loaded %s" % opt.rin)
    # create adjacency list for evaluation
    adjacency = ddict(set)
    for i in range(len(idx)):
        s, o, _ = idx[i]
        adjacency[s].add(o)
    adjacency = dict(adjacency)
    max_possible_norm = 1
    # setup Riemannian gradients for distances
    opt.retraction = rsgd.euclidean_retraction
    if opt.distfn == 'poincare':
        distfn = model.PoincareDistance
        opt.rgrad = rsgd.poincare_grad
    elif opt.distfn == 'euclidean':
        distfn = model.EuclideanDistance
        opt.rgrad = rsgd.euclidean_grad
    elif opt.distfn == 'dist_lorentz':
        distfn = model.LorentzianDistance
        opt.rgrad = rsgd.euclidean_grad
        max_possible_norm = None
    elif opt.distfn == 'transe':
        distfn = model.TranseDistance
        opt.rgrad = rsgd.euclidean_grad
    else:
        raise ValueError('Unknown distance function {opt.distfn}')

    # initialize model and data
    model, data, model_name, conf = model.SNGraphDataset.initialize(distfn, opt, idx, objects, max_norm=max_possible_norm)
    # Build config string for log
    conf = [
        ('distfn', '"{:s}"'),
        ('dim', '{:d}'),
        ('lr', '{:g}'),
        ('batchsize', '{:d}'),
        ('negs', '{:d}'),
    ] + conf
    conf = ', '.join(['"{}": {}'.format(k, f).format(getattr(opt, k)) for k, f in conf])
    log.info(f'json_conf: {{{conf}}}')


    optimizer = th.optim.SGD(model.parameters(), lr = opt.lr, momentum=0.9)

    # if nproc == 0, run single threaded, otherwise run Hogwild
    if opt.nproc == 0:
        train.train(model, data, optimizer, opt, log, 0)
    else:
        queue = mp.Manager().Queue()
        model.share_memory()
        processes = []
        for rank in range(opt.nproc):
            p = mp.Process(
                target=train.train_mp,
                args=(model, data, optimizer, opt, log, order_rank, rank + 1, queue)
            )
            p.start()
            processes.append(p)

        ctrl = mp.Process(
            target=control,
            args=(queue, log, adjacency, data, opt.fout, distfn, opt.epochs, processes,dataset_name, order_rank)
        )
        ctrl.start()
        ctrl.join()
