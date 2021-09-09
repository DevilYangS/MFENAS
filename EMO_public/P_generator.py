import numpy as np

def P_generator(MatingPool,Boundary,Coding,MaxOffspring,op_index):
# % 交叉, 变异并生成新的种群
# % 输入: MatingPool, 交配池, 其中每第i个和第i + 1
# 个个体交叉产生两个子代, i为奇数
# % Boundary, 决策空间, 其第一行为空间中每维的上界, 第二行为下界
# % Coding, 编码方式, 不同的编码方式采用不同的交叉变异方法
# % MaxOffspring, 返回的子代数目, 若缺省则返回所有产生的子代, 即和交配池的大小相同
# % 输出: Offspring, 产生的子代新种群

    Num_Op = max(Boundary[0])+1 # kexuan number of operations

    N = len(MatingPool)
    if MaxOffspring < 1 or MaxOffspring > N:
       MaxOffspring = N
    if Coding == "Real":
       N, D = MatingPool.shape
       ProC = 1
       ProM = 1/D
       DisC = 20
       DisM = 20
       Offspring = np.zeros((N, D))
       for i in range(0,N,2):
           beta = np.zeros((D,))
           miu = np.random.random((D,)) #np.random.rand(D,)
           beta[miu <= 0.5] = (2 * miu[miu <= 0.5])**(1/(DisC+1))
           beta[miu > 0.5] = (2-2 * miu[miu > 0.5]) ** (-1 / (DisC + 1))
           beta = beta * ((-1) ** (np.random.randint(0, 2, (D,))))
           beta[np.random.random((D,)) > ProC] = 1

           Offspring[i, :] = ((MatingPool[i, :] + MatingPool[i+1, :] )/2) + (np.multiply(beta, (MatingPool[i, :] - MatingPool[i+1, :])/2 ))
           Offspring[i+1, :] = ((MatingPool[i, :] + MatingPool[i+1, :] )/2) - (np.multiply(beta, (MatingPool[i, :] - MatingPool[i+1, :])/2 ))
       Offspring_temp = Offspring[:MaxOffspring,:]
       # print(range(MaxOffspring,Offspring.shape[0]))
       # np.delete(Offspring, range(MaxOffspring,Offspring.shape[0]), axis=0) 并没有真正的对 对象进行操作，仅仅你是个浅操作
       Offspring = Offspring_temp

       if MaxOffspring == 1:
           MaxValue = Boundary[0,:]
           MinValue = Boundary[1,:]
       else:
           MaxValue = np.tile(Boundary[0,:],(MaxOffspring,1))
           MinValue = np.tile(Boundary[1,:],(MaxOffspring,1))

       #np.bitwise_and 用于矩阵的逻辑运算
       k = np.random.random((MaxOffspring, D))
       miu = np.random.random((MaxOffspring, D))
       Temp = np.bitwise_and(k <= ProM, miu <0.5)

       Offspring[Temp] = Offspring[Temp] + np.multiply((MaxValue[Temp] - MinValue[Temp]), ((2 * miu[Temp] + np.multiply(
           1 - 2 * miu[Temp],
           (1 - (Offspring[Temp] - MinValue[Temp]) / (MaxValue[Temp] - MinValue[Temp])) ** (DisM + 1))) ** (1 / (
                   DisM + 1)) - 1))

       Temp = np.bitwise_and(k <= ProM, miu >= 0.5)
       
       Offspring[Temp] = Offspring[Temp] + np.multiply((MaxValue[Temp] - MinValue[Temp]), (1-((2 *(1-miu[Temp])) + np.multiply(
           2 * (miu[Temp]-0.5),
           (1 - (MaxValue[Temp] - Offspring[Temp]) / (MaxValue[Temp] - MinValue[Temp])) ** (DisM + 1))) ** (1 / (
                   DisM + 1)) ))

       Offspring[Offspring > MaxValue] = MaxValue[Offspring>MaxValue]
       Offspring[Offspring < MinValue] = MinValue[Offspring < MinValue]

    elif Coding == "Binary":

        Offspring = []



        cross_ratio = 0.1 # 0.2

        for i in range(0, N, 2):
            P1 = MatingPool[i].dec.copy()
            P2 = MatingPool[i + 1].dec.copy()

            cross_flag = np.random.rand(1)<cross_ratio

            for j in range(2):
                p1 = np.array(P1[j]).copy()
                p2 = np.array(P2[j]).copy()

                # ----------------------------crossover-------------------------------
                L1, L2 = len(p1),len(p2)
                L_flag = L1>L2
                large_L = L1 if L_flag else L2
                common_L = L2 if L_flag else L1
                cross_L = np.random.choice(common_L)


                if cross_flag:
                    p1[:cross_L], p2[:cross_L] = p2[:cross_L], p1[:cross_L]


                #----------------------------mutation-------------------------------
                # muta_indicator_1 = np.random.rand(2,common_L)<1 / common_L
                # muta_indicator_2 = np.random.rand(large_L,) <2 / common_L
                # muta_indicator_2[:common_L] = muta_indicator_1[1]
                # muta_indicator_1 = muta_indicator_1[0]

                muta_indicator_1,muta_indicator_2 = mutation_indicator(p1.copy(),p2.copy(),op_index)

                muta_p1 = mutation(p1.copy(), op_index,Num_Op)
                muta_p2 = mutation(p2.copy(), op_index,Num_Op)

                if not L_flag:
                    p1[muta_indicator_1] = muta_p1[muta_indicator_1]
                    p2[muta_indicator_2] = muta_p2[muta_indicator_2]
                else:
                    p1[muta_indicator_2] = muta_p1[muta_indicator_2]
                    p2[muta_indicator_1] = muta_p2[muta_indicator_1]

                p1 = Bound2_least_node(p1, op_index)
                p2 = Bound2_least_node(p2, op_index)

                P1[j] = list(p1.copy())
                P2[j] = list(p2.copy())



            # ----------------------------crossover between cell-------------------------------
            if not cross_flag:
                temp_p1 = P1.copy()
                P1[1] = P2[1]
                P2[1] = temp_p1[1]

            Offspring.append(P1)
            Offspring.append(P2)

    return Offspring[:MaxOffspring]

def mutation_indicator(solution_1,solution_2, op_index):

    op_index_1 = [i for i in op_index if i <len(solution_1)]
    op_index_2 = [i for i in op_index if i <len(solution_2)]

    mutation_indicator_1 = np.random.rand(len(solution_1),) < 3/(len(solution_1)-len(op_index_1))
    mutation_indicator_2 = np.random.rand(len(solution_2),) < 3/(len(solution_2)-len(op_index_2))

    mutation_indicator_1[op_index_1] = np.random.rand(len(op_index_1),) < 1/len(op_index_1)
    mutation_indicator_2[op_index_2] = np.random.rand(len(op_index_2),) < 1/len(op_index_2)

    if  len(mutation_indicator_1)<=len(mutation_indicator_2):
        return mutation_indicator_1, mutation_indicator_2
    else:
        return mutation_indicator_2, mutation_indicator_1

def mutation(solution,op_index,Num_Op):

    solution_index = [i for i in op_index if i <len(solution)]
    op_candidate = []
    for j,index in enumerate(solution_index):
        A = np.random.choice(Num_Op)
        while A == solution[index]:
            A = np.random.choice(Num_Op)

        op_candidate.append(A)
    op_candidate = np.array(op_candidate)

    zero_index = solution==0
    one_index = solution==1
    solution[zero_index] = 1
    solution[one_index] = 0
    solution[solution_index] = op_candidate

    return solution

def Bound2_least_node(solution, op_index):

    solution_index = [i for i in op_index if i < len(solution)]
    Length_before = len(solution_index)

    L = 0
    j = 0
    zero_index = []
    while L < len(solution):
        S = L
        L += 3 + j
        node_j_A = np.array(solution[S:L]).copy()
        node_j = node_j_A[:-1]
        if node_j.sum() - node_j[zero_index].sum() == 0:
            zero_index.extend([j + 2])
        j += 1

    length_now = Length_before -len(zero_index)
    if length_now<5:
        L = 0
        j = 0
        zero_index = []
        while L < len(solution):
            S = L
            L += 3 + j
            node_j_A = np.array(solution[S:L]).copy()
            node_j = node_j_A[:-1]
            if node_j.sum() - node_j[zero_index].sum() == 0:
                zero_index.extend([j + 2])
                solution[S:L][1] =1

            j += 1

    return solution














