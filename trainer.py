import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

import math
import os
import time
import numpy as np

from utils import AverageMeter, angular_error
from model import gaze_network
import shutil
from modules.custom_conv import Conv2d

class Trainer(object):
    def __init__(self, config, data_loader=None,val_loader=None):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader, val_loader: data iterator
        """
        self.config = config

        # data params
        if config.train_status in ['train', 'meta_train', 'persona']:
            self.train_loader = data_loader
            self.num_train = len(self.train_loader.dataset)
        if config.train_status in ['meta_train', 'persona', 'test']:
            self.val_loader = val_loader

        # training params
        self.batch_size = config.batch_size
        self.epochs = config.epochs  # the total number of epochs to train
        self.lr = config.init_lr
        self.lr_patience = config.lr_patience
        self.lr_decay_factor = config.lr_decay_factor

        # misc params
        self.use_gpu = config.use_gpu
        self.ckpt_dir = config.model_save_dir  # output dir
        self.print_freq = config.print_freq
        self.train_iter = 0
        self.pre_trained_model_path = config.pre_trained_model_path
        self.resnet_model_path = config.resnet_model_path
        
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        if self.use_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        # build model
        self.model = gaze_network(self.resnet_model_path)

        if config.train_status in ['persona', 'meta_train']:
            print('In personalization/meta_train mode, loading pretrained model from: ', self.pre_trained_model_path)
            
            if config.train_status=='persona':
                ## allow padding tunable
                self.freeze_params()
                self.make_padding_trainable()

            # load model
            ckpt = torch.load(os.path.join(self.pre_trained_model_path, self.config.pre_trained_model_name))
            # load variables from checkpoint
            self.model.load_state_dict(ckpt['model_state'], strict=True)
            
            if config.train_status=='meta_train':
                self.freeze_params()
                self.make_padding_trainable()
            
            for name, w in self.model.named_parameters():
                if 'prompt' in name:
                    w.requires_grad = True

        if self.use_gpu:
            self.model.cuda()
        
        print('[*] Number of model parameters: {:,}'.format(sum([p.data.nelement() for p in self.model.parameters()])))
        print('[*] Number of trainable model parameters: {:,}'.format(sum([p.data.nelement() if p.requires_grad else 0 for p in self.model.parameters() ])))

        if config.train_status in ['persona']:
            # set different learning rate to different model parts
            #https://stackoverflow.com/questions/73629330/what-exactly-is-meant-by-param-groups-in-pytorch
            param_groups = []
            param_group_names = []
            for name, parameter in self.model.named_parameters():
                if 'prompt' in name:
                    lr_ = 0.01
                else:
                    lr_ = 0
                param_groups.append({'params': [parameter], 'lr':lr_})
                param_group_names.append(name)

            self.optimizer = optim.Adam(param_groups, lr=self.lr)
            self.scheduler = StepLR(self.optimizer, step_size=self.lr_patience, gamma=self.lr_decay_factor)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = StepLR(self.optimizer, step_size=self.lr_patience, gamma=self.lr_decay_factor)
            

    def train(self):
        print("\n[*] Train on {} samples".format(self.num_train))
        
        # train for each epoch
        for epoch in range(self.epochs):
            print('\nEpoch: {}/{} - base LR: {:.6f}'.format(epoch + 1, self.epochs, self.lr))

            for param_group in self.optimizer.param_groups:
                print('Learning rate: ', param_group['lr'])
            
            self.model.train()
            self.train_one_epoch(self.train_loader)

            # save the model for each epoch
            add_file_name = 'epoch_' + str(epoch)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'schedule_state': self.scheduler.state_dict()
                 }, add=add_file_name
            )
            self.scheduler.step()  # update learning rate


    def train_one_epoch(self, data_loader):

        errors = AverageMeter()

        for i, (input_img, target) in enumerate(data_loader):
            input_var =input_img["face"].cuda()
            target_var = target.cuda()

            # train gaze net with l1 loss and left-right symmetry loss
            with torch.no_grad():
                self.model.eval()
                pred_gaze0 = self.model(torch.flip(input_var, (3,)))
                pred_gaze_ = pred_gaze0.clone()
                pred_gaze_[:,0] = -pred_gaze0[:,0] # flip yaw
            
            self.model.train()
            pred_gaze= self.model(input_var)
        
            gaze_error_batch = np.mean(angular_error(pred_gaze.cpu().data.numpy(), target_var.cpu().data.numpy()))
            errors.update(gaze_error_batch.item(), input_var.size()[0])
            
            # L1 Gaze loss
            loss_gaze = F.l1_loss(pred_gaze, target_var)
            # LR Symmetry loss
            loss_sym = F.l1_loss(pred_gaze, pred_gaze_)
            
            loss_all = loss_gaze + loss_sym
            
            self.optimizer.zero_grad()
            loss_all.backward()
            self.optimizer.step()

            # report information
            if i % self.print_freq == 0 and i != 0:
                print('--------------------------------------------------------------------')
                print('Iteration {} Error: {}'.format(i, errors.avg))


    def evaluation_this_epoch(self, data_loader):
        self.model.eval()
        errors = AverageMeter()

        for i, (input_img, label) in enumerate(data_loader):
            input_var = input_img["face"].cuda()
            pred_gaze = self.model(input_var)
          
            gaze_error_batch = np.mean(angular_error(pred_gaze.cpu().data.numpy(), label.data.numpy()))
            errors.update(gaze_error_batch.item(), input_var.size()[0])

        print('Evaluation results: ', errors.avg)


    def meta_train(self):
        self.model.eval()

        # save original model
        previous_generator = os.path.join(self.ckpt_dir, 'tem_model.pth')
        torch.save(self.model.state_dict(), previous_generator)
        self.inner_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # train for each epoch
        for epoch in range(self.epochs):
            print('\nEpoch: {}/{}'.format(epoch + 1, self.epochs))

            for param_group in self.optimizer.param_groups:
                print('Learning rate: ', param_group['lr'])

            for index, (input_img, label) in enumerate(self.train_loader):
                input_var =input_img["face"].cuda()
                target_var = label.cuda()
                self.train_iter += 1

                if index % 2 ==0:
                    dummy_input = input_var[0].unsqueeze(0)
                    dummy_gt = target_var[0].unsqueeze(0)
                        
                    with torch.no_grad():
                        pred_gaze0 = self.model(torch.flip(input_var, (3,)))
                        pred_gaze_ = pred_gaze0.clone()
                        pred_gaze_[:,0] = -pred_gaze0[:,0]      
                    pred_gaze = self.model(input_var)
                    
                    # LR Loss
                    loss_gaze_lr = F.l1_loss(pred_gaze, pred_gaze_)
                    loss_all = loss_gaze_lr

                    self.inner_optimizer.zero_grad()
                    loss_all.backward()
                    self.inner_optimizer.step()

                else:
                    prevs_weights = torch.load(previous_generator)
                    pred_gaze = self.model(input_var)
                        
                    loss_gaze = F.l1_loss(pred_gaze, target_var)

                    # compute validation gradient
                    model_grads = torch.autograd.grad(loss_gaze, [p for p in self.model.parameters() if p.requires_grad], allow_unused=True)
                    # https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/parameter_update.py
                    # Handles gradients for non-differentiable parameters
                    gradients = []
                    grad_counter = 0
                    for param in self.model.parameters():
                        if param.requires_grad:
                            gradient = model_grads[grad_counter]
                            grad_counter += 1
                        else:
                            gradient = None
                        gradients.append(gradient)

                    model_meta_grads = {name:g for ((name, _), g) in zip(self.model.named_parameters(), gradients)}

                    # meta updates
                    # unpack the list of grad dicts
                    self.model.load_state_dict(prevs_weights)

                    # dummy forward pass
                    pred_gaze = self.model(dummy_input)
                    loss_gaze = F.l1_loss(pred_gaze, dummy_gt)
                    gen_gradients = model_meta_grads

                    self.meta_update_model(self.model, self.optimizer, loss_gaze, gen_gradients)
                    torch.save(self.model.state_dict(), previous_generator)

                    if self.train_iter % 200 == 0:
                        print('saving meta model.')
                        # save the model for each epoch
                        add_file_name = "meta_model_epoch_{}".format(epoch)
                        self.save_checkpoint(
                            {'epoch': epoch + 1,
                             'model_state': self.model.state_dict(),
                             'optim_state': self.optimizer.state_dict(),
                             'schedule_state': self.scheduler.state_dict()
                             }, add=add_file_name
                        )

                        print('We are evaluating epoch {}.'.format(epoch))
                        self.evaluation_this_epoch(self.val_loader)
                        
                        break


    
    def meta_update_model(self, model, optimizer, loss, gradients):
        # register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        # GENERATOR
        hooks = []
        for (k,v) in model.named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            if gradients[k] is not None:
                hooks.append(v.register_hook(get_closure()))

        # compute grads for current step, replace with summed gradients as defined by hook
        optimizer.zero_grad()
        loss.backward()

        # update the net parameters with the accumulated gradient according to optimizer
        optimizer.step()

        # remove the hooks before next training phase
        for h in hooks:
            h.remove()


    def test(self):
        for epoch in range(self.epochs):
            print('Epoch {}'.format(epoch))
            model_name = 'epoch_{}_ckpt.pth.tar'.format(str(epoch))
            ckpt = torch.load(os.path.join(self.pre_trained_model_path, model_name))
            # load variables from checkpoint
            self.model.load_state_dict(ckpt['model_state'], strict=True)
            self.evaluation_this_epoch(self.val_loader)

    
    # for personalization
    def persona(self, subid):
        print(' PERSONA ON SUBJECT ID: ', subid)
        print("\n[*] Persona on {} samples".format(self.num_train))

        # train for each epoch
        best_result = [0.0, 100.0]
        for epoch in range(self.epochs):
            print('\nEpoch: {}/{}'.format(epoch + 1, self.epochs))
            
            for i, (input_img, target) in enumerate(self.train_loader):
                self.model.eval()
                input_var =input_img["face"].cuda()

                with torch.no_grad():
                    pred_gaze0 = self.model(torch.flip(input_var, (3,)))
                    pred_gaze_ = pred_gaze0.clone()
                    pred_gaze_[:,0] = -pred_gaze0[:,0]
                   
                pred_gaze = self.model(input_var)

                loss_gaze = F.l1_loss(pred_gaze, pred_gaze_)    
                loss_all = loss_gaze
                
                self.optimizer.zero_grad()
                loss_all.backward()
                self.optimizer.step()
      
            # evaluate the personalized results
            with torch.no_grad():
                self.model.eval()
                errors = AverageMeter()

                for i, (input_img, label) in enumerate(self.val_loader):
                    input_var = input_img["face"].cuda()

                    pred_gaze0 = self.model(torch.flip(input_var, (3,)))
                    pred_gaze_ = pred_gaze0.clone()
                    pred_gaze_[:,0] = -pred_gaze0[:,0]
                    
                    pred_gaze = self.model(input_var)
                    
                    gaze_error_batch = np.mean(angular_error(pred_gaze.cpu().data.numpy(), label.data.numpy()))
                    errors.update(gaze_error_batch.item(), input_var.size()[0])

                result = [epoch, errors.avg]
                if errors.avg <= best_result[1]:
                    best_result = result
        return best_result
    

    def save_checkpoint(self, state, add=None):
        # save a copy of the model
        if add is not None:
            filename = add + '_ckpt.pth.tar'
        else:
            filename ='ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        print('save file to: ', filename)

    #########################################
    # relating to padding
    #########################################
    def freeze_params(self):
        for c in self.model.children():
            for param in c.parameters():
                param.requires_grad = False

    def make_padding_trainable(self,crop_size=224):
        c_count = 0
        m_count = 0
        for i,m in enumerate(self.model.gaze_network.children()):
            # tune padding on the first 9 convolutional layers
            # i = 1: first layer
            # i = 5: first 5 layers
            # i = 7: first 13 layers
            # i = 8: all 17 layers
            if i<6: 
                if isinstance(m, Conv2d):
                    c_count+=1
                    P = m.padding[0]
                    S = m.stride[0]
                    K = m.kernel_size[0]
                    m.data_crop_size = crop_size
                    m.make_padding_trainable()
                    for k,v in m.state_dict().items():
                        if k in ['prompt_embeddings_tb','prompt_embeddings_lr']:
                            for param in v:
                                param.requires_grad = True
                    crop_size = (crop_size+2*P-K)//S + 1
                elif isinstance(m,torch.nn.MaxPool2d):
                    m_count+=1
                    P = m.padding
                    S = m.stride
                    K = m.kernel_size
                    crop_size = (crop_size+2*P-K)//S + 1
                elif isinstance(m,torch.nn.Sequential):
                    for b in m.children():
                        for c in b.children():
                            if isinstance(c, Conv2d):
                                if c.padding:
                                    c_count+=1
                                    P = c.padding[0]
                                    S = c.stride[0]
                                    K = c.kernel_size[0]
                                    c.data_crop_size = crop_size
                                    c.make_padding_trainable()
                                    for k,v in c.state_dict().items():
                                        if k in ['prompt_embeddings_tb','prompt_embeddings_lr']:
                                            for param in v:
                                                param.requires_grad = True
                                    crop_size = (crop_size+2*P-K)//S + 1