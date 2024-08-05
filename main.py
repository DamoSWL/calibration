import torch
from ddpg_agent import DDPG
from env_gcn import gcn_env_GCN
import numpy as np
import random
import argparse
import logging
from utils.util import manual_seed
import scipy.sparse as sp
import os


def set_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter("")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
 
    l.setLevel(level)
    l.addHandler(fileHandler)



def main(args):

    logging.info('Beginning')
    torch.cuda.empty_cache()

    max_episodes = 20
    best_acc = 0
    best_ece = 0
    best_result = 100

    iter_cnt = 0


    env = gcn_env_GCN(dataset=args.dataset,datapath=args.datapath,args=args)

    agent = DDPG(memory_size=env.agent_memory_size,
                dataset=env.dataset,nb_states=env.nfeat,
                gamma=0.5,model_path=args.model_path,gpu=args.gpu)


    env.policy = agent

    batch_size = 10

    if args.training:
        logging.info('Training start')

        for i_episode in range(1,max_episodes+1): 

            if i_episode == 0:
                env.init_model()
                env.train(env.adj,args.total_epochs,save=True,last=False) 
                env.validate(env.adj) 

            env.update_bin_cof(env.adj)
            env.init_init_candidate_adj_s()

            logging.info('Training actor critic')

            batch = 0
            while batch < env.candidate_node_num:
                node_index = list(range(env.candidate_node_num))[batch:batch+batch_size]
                agent.learn(env,node_index,iter_cnt) 
                iter_cnt += 1
                batch += batch_size


            logging.info('validate')
            
            env.update_adj() 
            env.init_model()
            env.train(env.adj,args.total_epochs,save=True,last=False)      
            acc, ece, loss = env.validate(env.adj) 

            result = loss
                 
            if i_episode > 1 and result < best_result:
                best_result = result
                best_acc = acc
                best_ece = ece
            agent.save_model() 

            logging.info(f'Training DDPG episode: {i_episode}/{max_episodes}  Best_acc: {best_acc} Best_ece : {best_ece} acc: {acc} ece: {ece}')
        

        agent.load_model()
        env.update_adj()
                
    
    else:
        logging.info(f'Training GNN seed {args.seed}')
        agent.load_model()
        env.update_adj()

        env.train(env.adj,args.total_epochs,save=True)  

        logging.info('testing GNN')
        acc,ece= env.test(env.adj)

        logging.info(f'acc {acc} ece {ece}')

        logging.getLogger('acc').info(acc)
        logging.getLogger('ece').info(ece)




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='RL graph OOD')

    # Tuning options
    parser.add_argument('--tune_scales', '-tscl', action='store_true', default=True, help='whether to tune scales of perturbations by penalizing entropy on training set')
    parser.add_argument('--tune_dropedge', '-tdrope', action='store_true', default=True, help='whether to tune edge dropout rates')
    parser.add_argument('--tune_dropout', '-tdrop', action='store_true', default=True, help='whether to tune dropout rates (per layer)')
    parser.add_argument('--tune_weightdecay', '-tweidec', action='store_true', default=True, help='whether to tune weight decay')
    # Initial hyperparameter settings
    parser.add_argument('--start_dropedge', '-drope', type=float, default=0.3, help='starting edge dropout rate')
    parser.add_argument('--start_dropout', '-drop', type=float, default=0.1, help='starting dropout rate') 
    parser.add_argument('--start_weightdecay', '-weidec', type=float, default=5e-5, help='starting weightdecay rate') 
    # Optimization hyperparameters
    parser.add_argument('--total_epochs', '-totep', type=int, default=20, help='number of training epochs to run for (warmup epochs are included in the count)')
    parser.add_argument('--warmup_epochs', '-wupep', type=int, default=10, help='number of warmup epochs to run for before tuning hyperparameters')
    parser.add_argument('--train_lr', '-tlr', type=float, default=5e-4, help='learning rate on parameters') 
    parser.add_argument('--valid_lr', '-vlr', type=float, default=3e-3, help='learning rate on hyperparameters') 
    parser.add_argument('--encoder_lr', '-elr', type=float, default=1e-4, help='learning rate on hyperparameters') 
    parser.add_argument('--scale_lr', '-slr', type=float, default=1e-3, help='learning rate on scales (used if tuning scales)')
    parser.add_argument('--momentum', '-mom', type=float, default=0.9, help='amount of momentum on usual parameters')
    parser.add_argument('--train_steps', '-tstep', type=int, default=2, help='number of batches to optimize parameters on training set')
    parser.add_argument('--valid_steps', '-vstep', type=int, default=1, help='number of batches to optimize hyperparameters on validation set')
    # Regularization hyperparameters
    parser.add_argument('--entropy_weight', '-ewt', type=float, default=1e-5, help='penalty applied to entropy of perturbation distribution')
    parser.add_argument('--perturb_scale', '-pscl', type=float, default=0.5, help='scale of perturbation applied to continuous hyperparameters')
    # Training parameter 
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=20, help='Random seed.')
    parser.add_argument("--mixmode", action="store_true", default=False, help="Enable CPU GPU mixing mode.")
    parser.add_argument('--dataset', default="cora", help="The data set")
    parser.add_argument('--model_path', default="cora_HyperU_RL", help="The data set")
    parser.add_argument('--datapath', default="./data/", help="The data path.")
    parser.add_argument('--log', default="log_cora.txt", help="The log file name.")
    parser.add_argument("--early_stopping", type=int, default=0, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
    parser.add_argument('--OOD_detection', '-ood', type=int, default=1, help="0 for Misclassification, 1 for OOD detection.")
    # Model parameter
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument("--normalization", default="BingGeNormAdj", help="The normalization on the adj matrix.")
    parser.add_argument("--task_type", default="semi", help="The node classification task type (full and semi).")

    #experiments
    parser.add_argument('--adj_weight', '-adjw', type=float, default=1.0, help='ratio set for edge, use 99,98,97,96 for small,large,average, bi-party vacuity experiments')
    parser.add_argument('--test_times', type=int, default=10, help='how many time to run the test and results are averaged')
    parser.add_argument('--training', action='store_true', default=False,help='Enable taining of Q network')\
    
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu')


    args = parser.parse_args()
    manual_seed(args.seed)
    logging.basicConfig(filemode='a',filename=args.log, level=logging.DEBUG,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    set_logger('acc','acc.txt')
    set_logger('ece','ece.txt')


    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main(args)

