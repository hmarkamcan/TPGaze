import torch
from trainer import Trainer
import os
import numpy as np
import random
import argparse
import configparser
import reader

def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def run(config):
    if config.use_gpu:
        # ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(config.rand_seed)
        torch.manual_seed(config.rand_seed)
        np.random.seed(config.rand_seed)
        torch.cuda.manual_seed(config.rand_seed)
        os.environ['PYTHONHASHSEED'] = str(config.rand_seed)
        torch.cuda.manual_seed_all(config.rand_seed)
        print('We use random seed: ', config.rand_seed)

    
    # instantiate data loaders
    if config.train_status in ['train', 'meta_train']:
        data_loader = reader.txtload(config.train_loader_flag, config.dataset_paths, 'train', config.batch_size,
                                         shuffle=True, num_workers=config.num_workers)

    if config.train_status in ['meta_train', 'test']:
        val_loader = reader.txtload(config.val_loader_flag, config.dataset_paths, 'test', config.batch_size, num_workers=config.num_workers)
      
            
    if config.train_status == 'persona':
        if config.val_loader_flag == 'MPII':
            subject_id_list = np.array(range(15)) + 1
            subject_id_list_str = ['p00', 'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12', 'p13', 'p14']
        elif config.val_loader_flag == 'EyeDiap':
            subject_id_list = np.array([1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 14, 15, 16])
            subject_id_list_str = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p14', 'p15', 'p16']


    # instantiate trainer
    if config.train_status == 'train':
        print('We are performing regular training !!!')
        print('Training data directory: ', config.data_dir)
        trainer = Trainer(config, data_loader=data_loader)
        trainer.train()
    elif config.train_status == 'meta_train':
        print('We are performing meta training !!!')
        print('Meta training data directory: ', config.data_dir)
        trainer = Trainer(config, data_loader=data_loader, val_loader=val_loader)
        trainer.meta_train()
    elif config.train_status == 'persona':
        print('We are performing personalized adaptation !!!')
        print('Personalization data directory: ', config.test_data_dir)
        best_dict = {}
        best_list_all = []
        loss_before = []
        loss_after = []
        for i in range(len(subject_id_list)): # we are working on each subject 

            persona_train_loader = reader.txtload(config.val_loader_flag, config.dataset_paths, 'train', config.batch_size,
                                shuffle=False, subject_id=subject_id_list_str[i], img_num=config.num_samples, num_workers=config.num_workers)
            persona_test_loader = reader.txtload(config.val_loader_flag, config.dataset_paths, 'test', config.batch_size,
                                shuffle=False, subject_id=subject_id_list_str[i], num_workers=config.num_workers)


            trainer = Trainer(config, data_loader=persona_train_loader, val_loader=persona_test_loader)
            
            best_results = trainer.persona(subject_id_list[i])
            best_list_all.append(best_results[1])
            best_dict[subject_id_list[i]] = best_results
        print('Personalized Results: ' )
        print(best_dict)
        print('Overall Results:')
        print(sum(best_list_all)/len(best_list_all))
        
    else:
        print('We are performing test !!!')
        print('Test data directory: ', config.test_data_dir)
        trainer = Trainer(config, val_loader=val_loader)
        trainer.test()
        


if __name__ == '__main__':
    arg_lists = []
    parser = argparse.ArgumentParser(description='RAM')

    # data params
    data_arg = add_argument_group('Data Params')
    data_arg.add_argument('--data_dir', type=str, default='/data/xgaze',
                          help='Directory of the training data')
    data_arg.add_argument('--test_data_dir', type=str, default='/data/mpiiface',
                          help='Directory of the test data')
    data_arg.add_argument('--batch_size', type=int, default=120,
                          help='# of images in each batch of data')
    data_arg.add_argument('--num_workers', type=int, default=5,
                          help='# of subprocesses to use for data loading')
    data_arg.add_argument('--num_samples', type=int, default=5,
                          help='# samples for persona')
    data_arg.add_argument('--train_loader_flag', type=str, default='ETH',
                          help='choose from ETH and Gaze360')
    data_arg.add_argument('--val_loader_flag', type=str, default='MPII',
                          help='choose from MPII and EyeDiap')
    # training params
    train_arg = add_argument_group('Training Params')
    train_arg.add_argument('--train_status', type=str, default='train',
                           help='Whether to train or test the model, choose from [train, meta_train, persona, test]')
    train_arg.add_argument('--epochs', type=int, default=10,
                           help='# of epochs to train for')
    train_arg.add_argument('--init_lr', type=float, default=0.0001,
                           help='Initial learning rate value')
    train_arg.add_argument('--lr_patience', type=int, default=25,
                           help='Number of epochs to wait before reducing lr (for pretrain)')
    train_arg.add_argument('--lr_decay_factor', type=float, default=0.1,
                           help='Factor to use to reduce lr by (for pretrain)')
    train_arg.add_argument('--rand_seed', type=int, default=42,
                           help='random seed')
    # other params
    misc_arg = add_argument_group('Misc.')
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                          help="Whether to run on the GPU")
    misc_arg.add_argument('--pre_trained_model_path', type=str, default='',
                          help='Directory in which to load model checkpoints')
    misc_arg.add_argument('--pre_trained_model_name', type=str, default='epoch_0_ckpt.pth.tar',
                          help='Name of the model')
    misc_arg.add_argument('--resnet_model_path', type=str, default='',
                          help='Directory in which resnet pretrained model is saved')
    misc_arg.add_argument('--print_freq', type=int, default=100,
                          help='How frequently to print training details')
    misc_arg.add_argument('--model_save_dir', type=str, default='./',
                          help='Directory in which to save model checkpoints')
    
    config, unparsed = parser.parse_known_args()
            
    config.dataset_paths = {'ETH': config.data_dir,
            'Gaze360': config.data_dir,
            'MPII': config.test_data_dir,
            'EyeDiap': config.test_data_dir}
    run(config)