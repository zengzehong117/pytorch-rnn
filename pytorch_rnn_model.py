import os
import os.path
import time

from audio_data import *
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
class maskGRU(nn.Module):
    def __init__(self,
                 hidden_dim,
                 batch_size,
                 input_dim,                 
                 onehot_dim=256,
                 out_classes=256,
                 out_classes_tmp=300,
                 embbed_dim=256,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(maskGRU, self).__init__()

#        self.layers = layers
        self.batch_size = batch_size
        self.out_classes = out_classes
        self.out_classes_tmp = out_classes_tmp
        self.hidden_dim=hidden_dim
        self.input_dim=input_dim
        self.input_classes = 256
#        self.input_c=input_c
#        self.input_f=input_f
        self.onehot_dim=onehot_dim
        self.embbed_dim=embbed_dim
        self.hidden           = Variable(torch.zeros(self.batch_size, self.hidden_dim)).cuda()

        self.hidden2hidden   = nn.Linear(self.hidden_dim,self.hidden_dim*3)
        self.hidden1_toout1  = nn.Linear(int(self.hidden_dim/2),self.out_classes_tmp)
        self.hidden2_toout2  = nn.Linear(int(self.hidden_dim/2),self.out_classes_tmp)
        self.out1_toout1     = nn.Linear(self.out_classes_tmp,self.out_classes)
        self.out2_toout2     = nn.Linear(self.out_classes_tmp,self.out_classes)
               
        self.to_embbed1       = nn.Embedding(self.input_classes,self.embbed_dim)
        self.to_embbed2       = nn.Embedding(self.input_classes,self.embbed_dim)
        self.to_embbed3       = nn.Embedding(self.input_classes,self.embbed_dim)
        
        self.input12_to_hid  = nn.Linear(self.input_dim*2,self.hidden_dim*3)
        self.input3_to_hid   = nn.Linear(self.input_dim,self.hidden_dim*3//2)
        self.hidden2hidden   = nn.Linear(self.hidden_dim,self.hidden_dim*3)
        
    def init_hidden(self):
        self.hidden           = Variable(torch.zeros(self.batch_size, self.hidden_dim)).cuda()
    
    def forward(self, input1,input2,target1):
        input_vec1 = self.to_embbed1(input1)
        input_vec2 = self.to_embbed2(input2)
        target1_vec= self.to_embbed3(target1)
        hidden     = self.hidden
        outs1,outs2= [],[]
        for i in range(len(input1)):
            mini_input1,mini_input2,mini_target1 = input_vec1[i],input_vec2[i],target1_vec[i]
            input12           =torch.cat([mini_input1,mini_input2],1)
            input12_to_hid        =self.input12_to_hid(input12)
            input3_to_hid         =self.input3_to_hid(mini_target1)
            hidden2hidden         =self.hidden2hidden(hidden)
            input12_to_hid[:,self.hidden_dim//2:self.hidden_dim]=input12_to_hid[:,self.hidden_dim//2:self.hidden_dim]+input3_to_hid[:,:self.hidden_dim//2]
            input12_to_hid[:,self.hidden_dim*3//2:self.hidden_dim*2]=input12_to_hid[:,self.hidden_dim*3//2:self.hidden_dim*2]+input3_to_hid[:,self.hidden_dim//2:self.hidden_dim]
            input12_to_hid[:,self.hidden_dim*5//2:self.hidden_dim*3]=input12_to_hid[:,self.hidden_dim*5//2:self.hidden_dim*3]+input3_to_hid[:,self.hidden_dim:self.hidden_dim*3//2]
            gates      = input12_to_hid+hidden2hidden
            gate1      = F.sigmoid(gates[:,:self.hidden_dim])
            gate2      = F.sigmoid(gates[:,self.hidden_dim:self.hidden_dim*2])
            transform0 = gate2*hidden2hidden[:,self.hidden_dim*2:self.hidden_dim*3]
            transform  = F.tanh(transform0+input12_to_hid[:,self.hidden_dim*2:self.hidden_dim*3])
            hidden     = gate1*hidden+(1-gate1)*transform            
            out1       =F.relu(self.hidden1_toout1(hidden[:,0:int(self.hidden_dim/2)]))
            out2       =F.relu(self.hidden2_toout2(hidden[:,int(self.hidden_dim/2):self.hidden_dim]))
            out11      =self.out1_toout1(out1)
            out22      =self.out2_toout2(out2)
            outs1.append(out11.unsqueeze(0))
            outs2.append(out22.unsqueeze(0))
        outs1,outs2  = torch.cat([*outs1]),torch.cat([*outs2]) 
        
        return outs1,outs2,hidden
