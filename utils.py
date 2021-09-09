import torch
from Operation import Operations_11_name, Operations_11
# import pygraphviz as pgv

import os

import gc
import threading
from io import BytesIO
from PIL import Image
import torch.utils.data as data

from utils_package import  *

from misc.flops_counter import add_flops_counting_methods


import shutil

#========================================= save & load =================================
def save(model_path, args, model, epoch, step, optimizer, best_acc_top1, is_best=True):
    if hasattr(model, 'module'):
        model = model.module
    state_dict = {
        'args': args,
        'model': model.state_dict() if model else {},
        'epoch': epoch,
        'step': step,
        'optimizer': optimizer.state_dict(),
        'best_acc_top1': best_acc_top1,
    }
    filename = os.path.join(model_path, 'checkpoint{}.pt'.format(epoch))
    torch.save(state_dict, filename)
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    shutil.copyfile(filename, newest_filename)
    if is_best:
        best_filename = os.path.join(model_path, 'checkpoint_best.pt')
        shutil.copyfile(filename, best_filename)

def load(model_path):
    if model_path is None:
        return None, None, 0, 0, None, 0
    else:
        newest_filename = os.path.join(model_path, 'checkpoint.pt')
        if not os.path.exists(newest_filename):
            return None, None, 0, 0, None, 0
        state_dict = torch.load(newest_filename)
        args = state_dict['args']
        model_state_dict = state_dict['model']
        epoch = state_dict['epoch']
        step = state_dict['step']
        optimizer_state_dict = state_dict['optimizer']
        best_acc_top1 = state_dict.get('best_acc_top1')

        return args, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1

#=================================================ImageNet==============================

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def convert_to_pil(bytes_obj):
    img = Image.open(BytesIO(bytes_obj))
    return img.convert('RGB')


class ReadImageThread(threading.Thread):
    def __init__(self, root, fnames, class_id, target_list):
        threading.Thread.__init__(self)
        self.root = root
        self.fnames = fnames
        self.class_id = class_id
        self.target_list = target_list

    def run(self):
        for fname in self.fnames:
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                path = os.path.join(self.root, fname)
                with open(path, 'rb') as f:
                    image = f.read()
                item = (image, self.class_id)
                self.target_list.append(item)


class InMemoryDataset(data.Dataset):
    def __init__(self, path, transform=None, num_workers=1):
        super(InMemoryDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.samples = []
        classes, class_to_idx = self.find_classes(self.path)
        dir = os.path.expanduser(self.path)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                if num_workers == 1:
                    for fname in sorted(fnames):
                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                            path = os.path.join(root, fname)
                            with open(path, 'rb') as f:
                                image = f.read()
                            item = (image, class_to_idx[target])
                            self.samples.append(item)
                else:
                    fnames = sorted(fnames)
                    num_files = len(fnames)
                    threads = []
                    res = [[] for i in range(num_workers)]
                    num_per_worker = num_files // num_workers
                    for i in range(num_workers):
                        start_index = num_per_worker * i
                        end_index = num_files if i == num_workers - 1 else num_per_worker * (i+1)
                        thread = ReadImageThread(root, fnames[start_index:end_index], class_to_idx[target], res[i])
                        threads.append(thread)
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    for item in res:
                        self.samples += item
                    del res, threads
                    gc.collect()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, target = self.samples[index]
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def find_classes(root):
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx




#========================================= AvgrageMeter & Accuracy =================================
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self._print = '{time:.6f} ({avg:.6f})'.format(time=val,avg=self.avg)

#========================================= creat dir =================================
def create__dir(path):
    if not os.path.exists(path):
        try :
            os.mkdir(path)
        except Exception:
            os.makedirs(path)
        print('Make Dir : {}'.format(path))


#========================================= DAG Node ===========================================
class dagnode():
    #Usage：
    # dag = collections.defaultdict(list)
    # dag[-1] = dagnode(-1, [], None)
    # dag[0] = dagnode(0, [0], None)
    # dag[1] = dagnode(1, [1, 1], 1)
    # dag[2] = dagnode(2, [1, 0, 1], 4)
    # dag[3] = dagnode(3, [0, 0, 1, 0], 7)
    # dag[4] = dagnode(4, [1, 1, 0, 1, 1], 9)
    def __init__(self, node_id, adj_node, op_id):
        self.node_id = node_id
        self.adj_node = adj_node
        if node_id<1:
            self.op_id = 'cell_'+str(node_id)
            self.op_name = 'Cell operation '+str(node_id)
        else:
            self.op_id = op_id
            self.op_name = Operations_11_name[op_id]


#========================================= Calculate  Network's parameters ===========================================
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6



def Calculate_flops(model):
    # copy from NSGA-Net
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()
    random_data = torch.randn(1, 3, 32, 32).cuda()
    model(random_data)
    flops = np.round(model.compute_average_flops_cost() / 1e6, 4) # MB
    return flops


#========================================= Draw  Network ===========================================

def construct_plot_dags(cell_dag):
    # which is different to  cell_dag, representation in 'Successor'
    # but cell_dag in 'Precursor'

    # Note :
    #cell_dag's index start with [-1, 0](denoting first two cell output) to end
    #plot_dags's index start with [-2, -1]

    # construct adjacent matrix
    Num_nodes = len(cell_dag)
    Adj = Get_Adjmatrix_from_celldag(cell_dag)

    dag = collections.defaultdict(list)
    # add first node (cell) and second node (cell)
    for i in range(Num_nodes):
        Successor_i = Adj[i]
        for node_j, flag in enumerate(Successor_i):
            if flag and node_j>i:
                dag[i-2].append( Node(node_j-2, cell_dag[node_j-1].op_name) )
                #node_j-2 is plot_dag, node_j-1 is cell_dag

    leaf_nodes = set(range(-2,Num_nodes-2)) - dag.keys()
    #leaf_nodes have done to be consistent with plot_dag
    for idx in leaf_nodes:
        dag[idx] = [Node(Num_nodes-2, 'Concat')]
        # leaf_nodes need to  connect to 'Concat' Node

    # 'Concat' Node then to 'Output' Node
    dag[Num_nodes-2] = [Node(Num_nodes-2 +1, 'Output')]

    return dag



def add_node(graph, node_id, label, shape='box', style='filled'):
    if label.startswith('h[t]'):
        color = 'white'
    elif label.startswith('h[t-1]'):
        color = 'skyblue'
    elif label.startswith('Conv') or label.startswith('SepConv') or label.startswith('DilConv') :
        color = 'seagreen3'
    elif label.startswith('MaxPool'):
        color = 'pink'
    elif label.startswith('Identity'):
        color = 'yellow'
    elif label.startswith('AvgPool'):
        color = 'greenyellow'
    elif label == 'Concat':
        color = 'orange'
    elif label == 'SELayer':
        color = 'tomato'
    else:
        color = 'cyan'

    if not any(label.startswith(word) for word in  ['Concat', 'Output', 'h']):
        label = f"{label}\n({node_id})"

    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style,
    )

# def draw_network(dag, path):
#     # Here dag is in the form of plot_dag
#
#     create__dir(os.path.dirname(path))
#
#     graph = pgv.AGraph(directed=True, strict=True,
#                        fontname='Helvetica', arrowtype='open')
#
#
#     checked_ids = [-2, -1 ]
#
#     if -1 in dag:
#         add_node(graph, -1, 'h[t]')
#     if -2 in dag:
#         add_node(graph, -2, 'h[t-1]')
#
#     # add_node(graph, 0, dag[-1][0].name)
#
#     for idx in dag:
#         for node in dag[idx]:
#             if node.id not in checked_ids:
#                 add_node(graph, node.id, node.name)
#                 checked_ids.append(node.id)
#             graph.add_edge(idx, node.id)
#
#     graph.layout(prog='dot')
#     graph.draw(path)
#     del graph
#
# def Plot_network(dag, path):
#     plot_dag = construct_plot_dags(dag)
#
#     draw_network(plot_dag, path)


## ================================TensorboardX====================

# from tensorboardX import  SummaryWriter
# input_2 = torch.rand([1, 3, 32, 32])
#
# with SummaryWriter(comment='NetworkCIFAR')as w:
#     w.add_graph(model, (input_2,))
#
## =========================================使用GRAPHVIZ+TORCHVIZ来可视化模型====================================
# from torchviz import make_dot
# model.eval()
# out = model(input_2)
# g = make_dot(out)
# g.render('espnet_model', view=True)
# =========================================使用GRAPHVIZ+TORCHVIZ来可视化模型====================================

#
# if __name__ == '__main__':
#
#
#     dag = collections.defaultdict(list)
#     dag[-1] = dagnode(-1, [], None)
#     dag[0] = dagnode(0, [0], None)
#     dag[1] = dagnode(1, [0, 1], 1)
#     dag[2] = dagnode(2, [1, 0, 1], 4)
#     dag[3] = dagnode(3, [0, 0, 1, 0], 7)
#     dag[4] = dagnode(4, [1, 1, 0, 1, 0], 9)
#     dag[5] = dagnode(5, [0, 1, 1, 0, 0, 0], 10)
#     dag[6] = dagnode(6, [0, 1, 0, 1, 0, 0, 1], 3)
#
#
#     plot_dag = construct_plot_dags(dag)
#
#     draw_network(plot_dag,'logs1/test_6.png')
