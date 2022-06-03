## Introduction
This is the source code of the paper entitled [*Accelerating Envolutionary Neural Architecture via Multi-Fidelity Evaluation*](https://arxiv.org/abs/2108.04541)

## Notes
The source code inlcluding searching, training and best models found by MFENAS will be released once the paper is accepted.

-------------------------------------------- **Current Version Notes**------------------------------------------

The current version is just a baseline approach for reference, the detailed explanation and useage will be given when this paper is accepted.
Of course, you can study this source code yourself to help your work.

To plot the architecture of found cells, you have to install *pygraphviz* package and use *Plot_network* function in **utils.py**. 


## Dependency Install
```
pytorch>=1.4.0
pygraphviz # used for plotting neural architectures
```

## Usage
```
# Search process
python EMO.py

#Best solution
Baseline:  [[1, 1, 0, 1, 0, 0, 3, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 6, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 1, 0, 0, 1, 0, 8, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 10, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 3], [1, 0, 8, 0, 1, 0, 9, 0, 1, 0, 0, 8, 1, 1, 0, 0, 0, 7, 1, 0, 0, 0, 0, 0, 8, 1, 1, 0, 0, 0, 1, 0, 3]], channel=46

MFENAS: [[1, 1, 3, 1, 0, 0, 10, 1, 1, 1, 0, 3, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 7, 0, 1, 1, 0, 0, 0, 1, 11, 0, 1, 0, 0, 1, 0, 0, 0, 8, 1, 1, 0, 0, 0, 1, 0, 0, 0, 3, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 3, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 8, 1, 1, 0, 6, 1, 1, 1, 0, 9, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 6, 0, 0, 1, 0, 0, 1, 1, 7, 1, 1, 0, 0, 0, 0, 0, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 5]],channel =44 



# Training process
python train_cifar.py # validate on CIFAR-10 and CIFAR-100 datasets
python train_imagenet.py # validate on ImageNet dataset

# Plotting
Given a solution in EMO.py
the solution's normal cell and reduction cell can be plotted by
executing 'utils.Plot_network(solution.dag[0], path)' and 'utils.Plot_network(solution.dag[1], path)',
where 'path' is the path to save figures.

```




## Citation
If you find this work helpful in your research, please use the following BibTex entry to cite our paper.
```
@article{yang2021accelerating,
  title={Accelerating Evolutionary Neural Architecture Search via Multi-Fidelity Evaluation},
  author={Yang, Shangshang and Tian, Ye and Xiang, Xiaoshu and Peng, Shichen and Zhang, Xingyi},
  journal={arXiv preprint arXiv:2108.04541},
  year={2021}
}
```

## Acknowledgement
Thanks for the help of [NAO](https://github.com/renqianluo/NAO_pytorch/tree/master/NAO_V2), [NSGA-Net](https://github.com/ianwhale/nsga-net) and [ACE-NAS](https://github.com/anonymone/ACE-NAS).

