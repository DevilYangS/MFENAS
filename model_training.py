import argparse, logging
import random, time, sys
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from Build_Dataset import build_search_cifar10, build_search_Optimizer_Loss
from Node import NetworkCIFAR

from utils import dagnode, Plot_network,create__dir, count_parameters_in_MB, Calculate_flops
import collections,utils

# import os
# os.environ["CUDA_VISIBLE_DEVICE"] = '1'


#
def Model_train(train_queue, model, train_criterion, optimizer, scheduler, args,valid_queue,eval_criterion, print_ = False):

    since_time = time.time()

    global_step = 0
    total = len(train_queue)
    for epoch in range(args.search_epochs):

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.train()

        for step, (inputs, targets) in enumerate(train_queue):
            print('\r[Epoch:{0:>2d}/{1:>2d}, Training {2:>2d}/{3:>2d}, used_time {4:.2f}min]'.format(epoch+1, args.search_epochs,step + 1, total, (time.time()-since_time)/60), end='')

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs,step=global_step)
            global_step += 1
            if args.search_use_aux_head:
                outputs, outputs_aux =outputs[0], outputs[1]

            loss = train_criterion(outputs, targets)
            if args.search_use_aux_head:
                loss_aux = train_criterion(outputs_aux, targets)
                loss += args.search_auxiliary_weight * loss_aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.search_grad_bound)
            optimizer.step()

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

        scheduler.step()



        if print_ or (epoch+1)==args.search_epochs:
            logging.info('epoch %d lr %e', epoch + 1, scheduler.get_lr()[0])
            print('train accuracy top1:{0:.3f}, train accuracy top5:{1:.3f}, train loss:{2:.5f}'.format(top1.avg,top5.avg,objs.avg))
            logging.info('train accuracy top1:{0:.3f}, train accuracy top5:{1:.3f}, train loss:{2:.5f}'.format(top1.avg,top5.avg,objs.avg))
            valid_top1_acc, valid_top5_acc, loss = Model_valid(valid_queue,model,eval_criterion,args)

    used_time = (time.time()-since_time)/60
    return top1.avg, top5.avg, objs.avg, valid_top1_acc, valid_top5_acc, loss, used_time


def Model_valid(valid_queue, model, eval_criterion,args):
    total = len(valid_queue)

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    with torch.no_grad():
        model.eval()
        for step, (inputs, targets) in enumerate(valid_queue):
            print('\r[-------------Validating {0:>2d}/{1:>2d}]'.format(step + 1, total), end='')

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)

            if args.search_use_aux_head:
                outputs, outputs_aux =outputs[0], outputs[1]

            loss = eval_criterion(outputs, targets)

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

    print('valid accuracy top1:{0:.3f}, valid accuracy top5:{1:.3f}, valid loss:{2:.5f}'.format(top1.avg,top5.avg,objs.avg))
    logging.info('valid accuracy top1:{0:.3f}, valid accuracy top5:{1:.3f}, valid loss:{2:.5f}'.format(top1.avg,top5.avg,objs.avg))

    return top1.avg, top5.avg, objs.avg


def solution_evaluation(model, train_queue, valid_queue,args):
    num_parameters = count_parameters_in_MB(model)
    # ============================================ build optimizer, loss and scheduler ============================================
    train_criterion, eval_criterion, optimizer, scheduler = build_search_Optimizer_Loss(model, args, epoch=-1)
    # ============================================ training the individual model and get valid accuracy ============================================
    result = Model_train(train_queue, model, train_criterion, optimizer, scheduler, args, valid_queue, eval_criterion,print_=False)#True

    Flops = Calculate_flops(model)

    return 1-result[3]/100, num_parameters, Flops,result[6]




#===================================  main  ===================================
def Model_Evaluation(args):



    # ============================================ get solution's dag and draw its dag =================================
    dag = collections.defaultdict(list)
    dag[-1] = dagnode(-1, [], None)
    dag[0] = dagnode(0, [0], None)
    dag[1] = dagnode(1, [0, 1], 0)
    dag[2] = dagnode(2, [1, 0, 1], 2)
    dag[3] = dagnode(3, [0, 0, 1, 0], 1)
    dag[4] = dagnode(4, [1, 1, 0, 1, 1], 5)
    dag[5] = dagnode(5, [0, 1, 0, 0, 1, 0], 8)#10
    dag[6] = dagnode(6, [0, 1, 0, 1, 0, 0, 1], 11)
    dag[7] = dagnode(7, [0, 0, 0, 0, 0, 1, 0, 1], 0)
    dag[8] = dagnode(8, [1, 0, 0, 0, 0, 0, 1, 1, 0], 4)
    dag[9] = dagnode(9, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0], 7)
    dag[10]= dagnode(10,[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], 2)
    DAG_ = []
    DAG_.append(dag)

    Plot_network(dag, '{}/conv_dag.png'.format(args.save))
    del dag

    dag = collections.defaultdict(list)
    dag[-1] = dagnode(-1, [], None)
    dag[0] = dagnode(0, [0], None)
    dag[1] = dagnode(1, [0, 1], 6)
    dag[2] = dagnode(2, [0, 1, 1], 3)
    dag[3] = dagnode(3, [0, 0, 1, 0], 0)#10
    dag[4] = dagnode(4, [0, 0, 0, 1, 1], 2)
    dag[5] = dagnode(5, [0, 0, 1, 0, 1, 0], 1)
    dag[6] = dagnode(6, [0, 1, 0, 1, 0, 0, 1], 8)
    dag[7] = dagnode(7, [1, 0, 0, 0, 0, 0, 0, 0], 10)
    dag[8] = dagnode(8, [0, 1, 0, 0, 0, 0, 0, 0, 1], 11)
    dag[9] = dagnode(9, [0, 0, 1, 0, 0, 0, 0, 0, 0, 1], 2)
    dag[10] =dagnode(10,[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 1)

    DAG_.append(dag)
    Plot_network(dag, '{}/reduc_dag.png'.format(args.save))

    #============================================ get train and valid data sequence ===================================================
    train_queue, valid_queue = build_search_cifar10(args=args, ratio=0.9,num_workers=args.search_num_work)
    # ============================================ build model based on corresponding dag ============================================
    model = NetworkCIFAR(args, 10, args.search_layers, args.search_channels, DAG_, args.search_use_aux_head,
                         args.search_keep_prob,args.search_steps,args.search_drop_path_keep_prob,args.search_channels_double)
    num_parameters = count_parameters_in_MB(model)
    print('Model Parameters: {0:.3f} MB'.format(num_parameters))
    # ============================================ build optimizer, loss and scheduler ============================================
    train_criterion, eval_criterion, optimizer, scheduler = build_search_Optimizer_Loss(model, args, epoch=-1)
    # ============================================ training the individual model and get valid accuracy ============================================
    result = Model_train(train_queue, model, train_criterion,optimizer, scheduler,args,valid_queue,eval_criterion,print_=False)

    # ============================================ save model's parameters  ============================================
    Flops = Calculate_flops(model)

    torch.save(model.state_dict(), '{}/model.pkl'.format(args.save))
    logging.info('Final, total_used time:{0:.3f} min, valid_top1_acc:{1:.3f}%, valid_top5_acc:{2:.3f}%, Parameters:{3:.4f}MB, Flops:{4:.4f}MB'
        .format(result[6], result[3], result[4],   count_parameters_in_MB(model) ,Flops ))
    # ============================= return model's accuracy, parameters and flops ===============================
    return result[3], num_parameters, Flops,result[6]






if __name__ == '__main__':




    # ===================================  args  ===================================
    # ***************************  common setting******************
    parser = argparse.ArgumentParser(description='test argument')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-save', type=str, default='result')
    # ***************************  EMO setting******************
    parser.add_argument('-range_node',type=list,default=[5,12])
    parser.add_argument('-popsize',type=int,default=40)
    parser.add_argument('-Max_Gen',type=int,default=25)

    # ***************************  dataset setting******************
    parser.add_argument('-data', type=str, default="data")
    parser.add_argument('-search_cutout_size', type=int, default=None)#16
    parser.add_argument('-search_autoaugment', action='store_true', default=False)
    parser.add_argument('-search_num_work', type=int, default=12,help='the number of the data worker.')

    # ***************************  optimization setting******************
    parser.add_argument('-search_epochs', type=int, default=25)#50
    parser.add_argument('-search_lr_max', type=float, default=0.1)#0.025 NAO
    parser.add_argument('-search_lr_min', type=float, default=0.001)#0 for final training
    parser.add_argument('-search_momentum', type=float, default=0.9)
    parser.add_argument('-search_l2_reg', type=float, default=3e-4)#5e-4 for final training
    parser.add_argument('-search_grad_bound', type=float, default=5.0)
    parser.add_argument('-search_train_batch_size', type=int, default=128)
    parser.add_argument('-search_eval_batch_size', type=int, default=500)
    parser.add_argument('-search_steps',type=int,default=50000)
    # ***************************  structure setting******************
    parser.add_argument('-search_use_aux_head', action='store_true', default=True)
    parser.add_argument('-search_auxiliary_weight', type=float, default=0.4)
    parser.add_argument('-search_layers',type=int,default=1)#  3 ,6for final Network
    parser.add_argument('-search_keep_prob', type=float, default=0.6)# 0.6 also for final training
    parser.add_argument('-search_drop_path_keep_prob',type=float,default=0.8)#None 会在训练时提高 精度 和速度, 0.8等 更加耗时但最终训练会提升
    parser.add_argument('-search_channels', type=int, default=16)# 24:48 for final training
    parser.add_argument('-search_channels_double',action='store_true', default=False)# False for Cifar, True for ImageNet model

    args = parser.parse_args()
    args.search_steps = int(np.ceil(45000 / args.search_train_batch_size)) * args.search_epochs
    args.save = '{}/search_{}'.format(args.save, time.strftime("%Y-%m-%d-%H-%M-%S"))
    create__dir(args.save)
    # ----------------------------------- args  -------------------------------------



    # ===================================  logging  ===================================
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/logs.log'.format(args.save),
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

    logging.info("[Experiments Setting]\n" + "".join(
        ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))

    # ----------------------------------- logging  -------------------------------------


    # ===================================  random seed setting  ===================================
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
    # -----------------------------------  random seed setting  -----------------------------------




    Model_Evaluation(args)
    # from threading import Timer
    # t = Timer(2*3600, Model_Evaluation)
    # t.start()








