import torch
import random

import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np

from Node import Operations_11_name, NetworkCIFAR

from Build_Dataset import build_search_cifar10, build_search_Optimizer_Loss

from utils import dagnode, Plot_network,create__dir, count_parameters_in_MB, Calculate_flops
import collections,utils, argparse,time,logging,sys



def Model_train(train_queue, model, train_criterion, optimizer, scheduler, args,valid_queue,eval_criterion):

    valid_list = []
    train_list =[]

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





        logging.info('epoch %d lr %e', epoch + 1, scheduler.get_lr()[0])
        print('train accuracy top1:{0:.3f}, train accuracy top5:{1:.3f}, train loss:{2:.5f}'.format(top1.avg,top5.avg,objs.avg))
        logging.info('train accuracy top1:{0:.3f}, train accuracy top5:{1:.3f}, train loss:{2:.5f}'.format(top1.avg,top5.avg,objs.avg))
        valid_top1_acc, valid_top5_acc, loss = Model_valid(valid_queue,model,eval_criterion,args)

        train_list.append(top1.avg)
        valid_list.append(valid_top1_acc)

    used_time = (time.time()-since_time)/60
    return train_list, valid_list, used_time


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
    train_list, valid_list, used_time = Model_train(train_queue, model, train_criterion, optimizer, scheduler, args, valid_queue, eval_criterion)#True

    Flops = Calculate_flops(model)

    result = [train_list, valid_list, num_parameters,Flops,used_time]


    return result



class individual():
    def __init__(self, dec):
        #dec
        #dag
        #num_node
        self.dec = dec
        self.fitness = None
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

    def evaluate(self,train_queue, valid_queue,args):

        model = NetworkCIFAR(args, 10, args.search_layers, args.search_channels, self.dag, args.search_use_aux_head,
                         args.search_keep_prob,args.search_steps,args.search_drop_path_keep_prob,args.search_channels_double)
        self.fitness = solution_evaluation(model,train_queue,valid_queue,args)
        del  model
    def save(self,order,path):

        whole_path = '{}/{}/'.format(path,order)
        create__dir(whole_path)

        if self.fitness is None:
            return
        train_list, valid_list, num_parameters, Flops, used_time = self.fitness

        train_accuracy = whole_path + 'train_accuracy.txt'
        np.savetxt(train_accuracy, np.array(train_list), delimiter=' ')

        valid_accuracy = whole_path + 'valid_accuracy.txt'
        np.savetxt(valid_accuracy, np.array(valid_list), delimiter=' ')

        attribute = whole_path + 'attribute.txt'
        with open(attribute, "w") as file:
            file.write('Numer of parameters: {} \n'.format(num_parameters))
            file.write('Flops: {} \n'.format(Flops))
            file.write('used_time: {} \n'.format(used_time))


        dec = whole_path + 'dec.txt'
        with open(dec, "w") as file:
            file.write('{}'.format(self.dec))



class Sample():
    def __init__(self,  args,):#[5,8]
        self.args = args
        self.popsize = args.popsize

        self.Gen = 0
        self.initial_range_node = args.range_node
        self.save_dir =args.save

        self.get_op_index()
        self.op_num = len(Operations_11_name)
        self.max_length = self.op_index[-1]+1
        self.coding = 'Binary'



        self.Population = []
        self.Pop_fitness=[]
        self.finess_best = 0

        self.build_dataset()

        self.threshold = 0.08#0.08


    def get_op_index(self):
        self.op_index = []
        L = 0
        for i in range(self.initial_range_node[1]):
            L += 3+i
            self.op_index.extend([L-1])
    def build_dataset(self):
        train_queue, valid_queue = build_search_cifar10(args=self.args, ratio=0.9,num_workers=self.args.search_num_work)
        self.train_queue = train_queue
        self.valid_queue = valid_queue
    def initialization(self):
        for i in range(self.popsize):
            rate = (i+1)/self.popsize # used for controlling the network structure between 'line' and 'Inception'
            node_ = np.random.randint(self.initial_range_node[0],self.initial_range_node[1]+1, 2)

            list_individual = []

            for i,num in enumerate(node_):
                op = np.random.randint(0, 12, num)
                if i==0:
                    op_c = np.random.randint(0,4,num)
                else:
                    op_c = np.random.randint(4, 10, num)
                in_dicator = np.random.rand(num, ) < 0.8#0.8
                op[in_dicator] = op_c[in_dicator]

                L = 2
                dag_list =[]
                for j in range(num):
                    L += 1
                    link = np.random.rand(L-1)
                    link[-1] = link[-1] > rate
                    link[0:2] = link[0:2] < rate
                    link[2:-1] = link[2:-1] < 2 / len(link[2:-1]) if len(link[2:-1]) != 0 else []  # 2

                    if link.sum()==0:
                        if rate<0.5:
                            link[-1] = 1
                        else:
                            if np.random.rand(1)<0.5:
                                link[1] = 1
                            else:
                                link[0] = 1

                    link = np.int64(link)
                    link = link.tolist()
                    link.extend([op[j]])
                    dag_list.extend(link)
                list_individual.append(dag_list)

            self.Population.append(individual(list_individual))
        self.evaluation(self.Population)







    def evaluation(self, Pop):
        # 是否 normalize fitness
        # return np.random.rand(len(Pop),2)
        whole_path = '{}'.format(self.save_dir)

        for i, solution in enumerate(Pop):
            logging.info('solution: {0:>2d}'.format(i + 1))
            print('solution: {0:>2d}'.format(i + 1))
            solution.evaluate(self.train_queue, self.valid_queue, self.args)
            solution.save(i, whole_path)


        return None

    def Main_loop(self):
        self.initialization()


if __name__ == "__main__":

    # ===================================  args  ===================================
    # ***************************  common setting******************
    parser = argparse.ArgumentParser(description='test argument')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-save', type=str, default='result_sample')
    # ***************************  EMO setting******************
    parser.add_argument('-range_node', type=list, default=[5, 15])  # [5,12]
    parser.add_argument('-popsize', type=int, default=2) # 200

    # ***************************  dataset setting******************
    parser.add_argument('-data', type=str, default="data")
    parser.add_argument('-search_cutout_size', type=int, default=None)  # 16
    parser.add_argument('-search_autoaugment', action='store_true', default=False)
    parser.add_argument('-search_num_work', type=int, default=12, help='the number of the data worker.')

    # ***************************  optimization setting******************
    parser.add_argument('-search_epochs', type=int, default=2)  # 25
    parser.add_argument('-search_lr_max', type=float, default=0.1)  # 0.025 NAO
    parser.add_argument('-search_lr_min', type=float, default=0.001)  # 0 for final training
    parser.add_argument('-search_momentum', type=float, default=0.9)
    parser.add_argument('-search_l2_reg', type=float, default=3e-4)  # 5e-4 for final training
    parser.add_argument('-search_grad_bound', type=float, default=5.0)
    parser.add_argument('-search_train_batch_size', type=int, default=128)
    parser.add_argument('-search_eval_batch_size', type=int, default=500)
    parser.add_argument('-search_steps', type=int, default=50000)
    # ***************************  structure setting******************
    parser.add_argument('-search_use_aux_head', action='store_true', default=True)
    parser.add_argument('-search_auxiliary_weight', type=float, default=0.4)
    parser.add_argument('-search_layers', type=int, default=1)  # 3 for final Network
    parser.add_argument('-search_keep_prob', type=float, default=0.6)  # 0.6 also for final training
    parser.add_argument('-search_drop_path_keep_prob', type=float,
                        default=0.8)  # None 会在训练时提高 精度 和速度, 0.8等 更加耗时但最终训练会提升
    parser.add_argument('-search_channels', type=int, default=16)  # 24:48 for final training
    parser.add_argument('-search_channels_double', action='store_true',
                        default=False)  # False for Cifar, True for ImageNet model

    args = parser.parse_args()
    args.search_steps = int(np.ceil(45000 / args.search_train_batch_size)) * args.search_epochs
    args.save = '{}/sample-solutions-{}'.format(args.save, time.strftime("%Y-%m-%d-%H-%M-%S"))
    create__dir(args.save)

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

    EMO_NAS = Sample(args)
    EMO_NAS.Main_loop()