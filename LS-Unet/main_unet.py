
import argparse


import torch
from torch.utils.data import DataLoader 

from unet_train import train, test

from dataset import load_sum_train_data, load_test_data





if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--train_folder', type=str)
    parser.add_argument('--test_noisy', type=str)
    parser.add_argument('--test_clean', type=str)
    parser.add_argument('--gpus', type=str, default="cuda:") #determine gpu to use
    parser.add_argument('--mode', type=str, default='train') 
    parser.add_argument('--weight_path', type=str) #determine weight to save
    parser.add_argument('--weight_path1', type=str) 
    parser.add_argument('--result_path', type=str) #determine result to save

    args = parser.parse_args()
    
    #torch.cuda.set_device(args.gpus)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    #device = torch.device("cuda:"+args.gpus if torch.cuda.is_available() else "cpu")    
    device = torch.device(args.gpus)
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpus)
    else:
        device = torch.device("cpu")
    '''
    #pdb.set_trace()
    
    if args.mode == 'train':                        
        datas = load_sum_train_data(args.train_folder)
        train_loader = DataLoader(datas ,batch_size = 1, shuffle = False)
        train(train_loader, args.weight_path, device)
    
    if args.mode == 'test':
        datas = load_test_data(args.test_noisy ,args.test_noisy)
        test_loader = DataLoader(datas, shuffle=False)
        test(test_loader, args.weight_path, args.result_path, device)

    