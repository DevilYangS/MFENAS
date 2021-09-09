import argparse, logging, collections
import random, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils import dagnode,create__dir, count_parameters_in_MB
import utils
from Build_Dataset import build_train_cifar10,build_train_cifar100, build_train_Optimizer_Loss

from Node import NetworkCIFAR


# The difference between 'training' and 'searching':
#  (1) use_aux_head:        True     ,      False
#  (2) dropout_proability:  0.6      ,      1.0(0.6)
#  (3) channels :           24:48    ,       16
#  (4) autoaugment:         True     ,      False
#  (5) cutout size:         16       ,      None
#  (6) epochs:              600      ,      50/25
#  (7) lr_min:              0        ,      0.001
#  (8) l2_reg:             5e-4      ,      3e-4
#  (9) train_batch size:   128       ,      64/128
#  (10)layers(cell repeat): 6        ,       1/3
#  (11)drop_path_keep_prob: 0.8      ,      0.8(None)

# # same:
#  (1)grad_bound: 5.0
#  (2)lr_max: 0.025
#  (3)eval_batch size：  500
#  (4)auxiliary_weight: 0.4


class individual():
    def __init__(self, dec):
        #dec
        #dag
        #num_node
        self.dec = dec
        self.re_duplicate()
        #self.trans2bin()# if dec is (int10,op)
        self.trans2dag()

    # def trans2bin(self):
    #     self.bin_dec = []
    #     self.conv_bin_dec = []
    #     self.redu_bin_dec =[]
    #
    #     for i in range(2):
    #         temp_dec = []
    #         for j in range(int(len(self.dec[i])/2)):
    #             bin_value = bin(self.dec[i][2*j])
    #             temp_list = [int(i) for i in bin_value[2:] ]
    #             if len(temp_list)<j+2:
    #                 A = [0]*(j+2 - len(temp_list))
    #                 A.extend(temp_list)
    #                 temp_list = A.copy()
    #             temp_list.extend([self.dec[i][2*j+1]])
    #             temp_dec.append(temp_list)
    #         self.bin_dec.append(temp_dec)
    #
    #     temp = [self.conv_bin_dec.extend(i) for i in self.bin_dec[0]]
    #     del temp
    #     temp = [self.redu_bin_dec.extend(i) for i in self.bin_dec[1]]
    #     del temp
    def re_duplicate(self):
        #used for deleting the nodes not actived

        for i,cell_dag in enumerate(self.dec):
            L = 0
            j = 0
            zero_index = []
            temp_dec = []
            while L <len(cell_dag):
                S = L
                L +=3+j
                node_j_A = np.array(cell_dag[S:L]).copy()
                node_j = node_j_A[:-1]
                if node_j.sum()- node_j[zero_index].sum()==0:
                    zero_index.extend([j+2])
                else:
                    temp_dec.extend(np.delete(node_j_A, zero_index))
                j+=1

            self.dec[i] = temp_dec.copy()

    def trans2dag(self):
        self.dag = []
        self.num_node = []
        for i in range(2):
            dag = collections.defaultdict(list)
            dag[-1] = dagnode(-1, [], None)
            dag[0] = dagnode(0, [0], None)


            j = 0
            L = 0
            while L < len(self.dec[i]):
                S = L
                L += 3+j
                node_j = self.dec[i][S:L]
                dag[j+1] = dagnode(j+1,node_j[:-1],node_j[-1])
                j+=1
            self.num_node.extend([j])
            self.dag.append(dag)
            del dag





def train_cifar10(train_queue, model, train_criterion, optimizer, args,epoch,global_step,since_time):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    total = len(train_queue)

    for step, (inputs, targets) in enumerate(train_queue):
        print('\r  Epoch{0:>2d}/600,   Training {1:>2d}/{2:>2d}, used_time {3:.2f}min]'.format(epoch,step + 1, total, (time.time() - since_time) / 60),end='')

        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()

        outputs = model(inputs,global_step[0])
        global_step[0] += 1

        if args.use_aux_head:
            outputs, outputs_aux = outputs[0], outputs[1]

        loss = train_criterion(outputs, targets)
        if args.use_aux_head:
            loss_aux = train_criterion(outputs_aux, targets)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        # if (step + 1) % 100 == 0:
        #     print('epoch:{}, step:{}, loss:{}, top1:{}, top5:{}'.format(epoch+1, step+1, objs.avg, top1.avg, top5.avg))

    return top1.avg, top5.avg, objs.avg

def evaluation_cifar10(valid_queue, model, eval_criterion,args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            outputs = model(input)

            if args.use_aux_head:
                outputs, outputs_aux = outputs[0], outputs[1]

            loss = eval_criterion(outputs, target)

            prec1, prec5 = utils.accuracy(outputs, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

    return top1.avg, top5.avg, objs.avg

def run_main(args):

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True



    solution = individual([[1, 0, 3, 1, 1, 0, 5, 1, 0, 1, 0, 3, 1, 1, 1, 1, 0, 2, 0, 1, 0, 1, 0, 0, 3, 1, 1, 1, 1, 0, 0, 1, 11,
                            1, 1, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 1, 1, 0, 0, 3, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 8, 1, 1, 0, 0, 1,
                            0, 1, 0, 0, 0, 0, 3, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0
                               , 3, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 1, 8, 0, 0, 1, 8, 1, 1, 0, 0, 9, 0, 0, 0, 0, 1, 5,
                                                                                1, 1, 0, 0, 1, 0, 3]] )





    if args.dataset == 'cifar10':
        train_queue, valid_queue = build_train_cifar10(args=args, cutout_size=args.cutout_size, autoaugment=args.autoaugment)
        args.classes = 10
    elif args.dataset == 'cifar100':
        args.classes = 100
        train_queue, valid_queue = build_train_cifar100(args=args, cutout_size=args.cutout_size, autoaugment=args.autoaugment)



    model = NetworkCIFAR(args, args.classes, args.layers, args.channels, solution.dag, args.use_aux_head,
                         args.keep_prob,args.steps,args.drop_path_keep_prob,)

    print('Model Parameters: {} MB'.format(count_parameters_in_MB(model)))

    train_criterion, eval_criterion, optimizer, scheduler = build_train_Optimizer_Loss(model, args, epoch=-1)




    global_step = [0]
    epoch = 0
    best_acc_top1= 0
    since_time = time.time()
    while epoch<args.epochs:


        logging.info('epoch %d lr %e', epoch+1, scheduler.get_lr()[0])
        print('epoch:{}, lr:{}, '.format(epoch+1, scheduler.get_lr()[0]))

        train_acc, top5_avg, train_obj = train_cifar10(train_queue, model, train_criterion, optimizer, args, epoch,global_step,since_time)
        scheduler.step()

        logging.info('train_accuracy: %f, top5_avg: %f, loss: %f', train_acc, top5_avg, train_obj)
        print('\n       train_accuracy: {}, top5_avg: {}, loss: {}'.format(train_acc, top5_avg, train_obj))

        valid_acc_top1, valid_acc_top5, valid_obj = evaluation_cifar10(valid_queue, model, eval_criterion, args)

        logging.info('valid_accuracy: %f, valid_top5_accuracy: %f', valid_acc_top1,valid_acc_top5)
        print('         valid_accuracy: {}, valid_top5_accuracy: {}'.format(valid_acc_top1,valid_acc_top5))

        epoch += 1
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

            utils.save(args.save, args, model, epoch, epoch*(int(np.ceil(50000 / args.train_batch_size))), optimizer, best_acc_top1, is_best)

    return

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='train on cifar')
    # ***************************  common setting******************
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('-save', type=str, default='result')
    parser.add_argument('-device', type=str, default='cuda')
    # ***************************  dataset setting******************
    parser.add_argument('-data', type=str, default="data")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100'])
    parser.add_argument('-classes', type=int, default=10)  # 16
    parser.add_argument('-autoaugment', action='store_true', default=True)  # True
    parser.add_argument('-cutout_size', type=int, default=16)  # 16
    # ***************************  optimization setting******************
    parser.add_argument('-epochs', type=int, default=600)
    parser.add_argument('-lr_max', type=float, default=0.025)#0.1
    parser.add_argument('-lr_min', type=float, default=0)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-l2_reg', type=float, default=5e-4)
    parser.add_argument('-grad_bound', type=float, default=5.0)
    parser.add_argument('-train_batch_size', type=int, default=80)
    parser.add_argument('-eval_batch_size', type=int, default=500)
    # ***************************  structure setting******************
    parser.add_argument('-use_aux_head', action='store_true', default=True)
    parser.add_argument('-auxiliary_weight', type=float, default=0.4)
    parser.add_argument('-keep_prob', type=float, default=0.6)
    parser.add_argument('-drop_path_keep_prob', type=float,
                        default=0.8)
    parser.add_argument('-channels', type=int, default=40)
    parser.add_argument('-layers', type=int, default=6)
    args = parser.parse_args()

#=====================================setting=======================================
    args.save = '{}/train_{}'.format(args.save, time.strftime("%Y-%m-%d-%H-%M-%S"))
    create__dir(args.save)

    args.steps = int(np.ceil(50000 / args.train_batch_size)) * args.epochs
    args.cutout_size = 16
    args.use_aux_head = True

    args.dataset = 'cifar100'
    args.autoaugment = False # True: 83.049995 False: 82.439995
    
#=====================================setting=======================================


    # ===================================  logging  ===================================
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/logs.log'.format(args.save),
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

    logging.info("[Experiments Setting]\n" + "".join(
        ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))

    run_main(args)

