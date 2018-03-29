import time
from audio_data import WavenetDataset
from model_logging import *
from scipy.io import wavfile
from wave_rnn_model import maskGRU
import torch
import torch.autograd as autograd
from torch.autograd import Variable
dtype = torch.FloatTensor
ltype = torch.LongTensor
import torch.nn.functional as F


use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
hidden_dim = 896
input_dim  = 50

epochs     =50
batch_size =50
seq_len    =2000
target_len =2000//40
out_classes=256

lr         =0.0001
model = maskGRU(hidden_dim=hidden_dim,
                batch_size=batch_size,
                input_dim = input_dim,
                onehot_dim=256,
                out_classes=256,
                out_classes_tmp=300,
                embbed_dim = 50
                )
data = WavenetDataset(dataset_file='mp3/jhs.npz',
                      item_length=seq_len,
                      target_length=target_len,
                      test_stride=500)
print('the dataset has ' + str(len(data)) + ' items')

print('time:',time.asctime( time.localtime(time.time())).split()[3],'start training...')
model.train()
model.cuda()
dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=2,
                                         pin_memory=False)
optimizer = torch.optim.Adam(model.parameters(), lr)

weight_mask=Variable(torch.ones(input_dim*3,hidden_dim*3)).cuda()
weight_mask[input_dim*2:,0:int(hidden_dim/2)]=0.
weight_mask[input_dim*2:,hidden_dim:hidden_dim+int(hidden_dim/2)]=0.
weight_mask[input_dim*2:,hidden_dim*2:hidden_dim*2+int(hidden_dim/2)]=0.



for current_epoch in range(epochs):
    tic = time.time()
    train_loss = 0.
#    train_loss2 = 0.
    batch_step = 0
    tmp_loss   = 0.
    tmp_loss1 = 0.
    tmp_loss2 = 0.
    for (x1,x2,target1,target2) in iter(dataloader):
#        print(len(x1))
        if len(x1)<batch_size:
            continue
        x1 = Variable(x1).cuda()
        x2 = Variable(x2).cuda()
        
        x1,x2 =x1.t(),x2.t()

        target1,target2 = target1.t(),target2.t()
        target1_forin   = Variable(target1).cuda()
        target1 = Variable(target1.contiguous().view(-1)).cuda()
        target2 = Variable(target2.contiguous().view(-1)).cuda()
        
        optimizer.zero_grad()
        out1,out2,_  = model(x1,x2,target1_forin)

        out1,out2=out1.view(-1,out_classes),out2.view(-1,out_classes)          
        loss1=F.cross_entropy(out1, target1)
        loss2=F.cross_entropy(out2, target2)
        loss = loss1+loss2        
        loss.backward()
        optimizer.step()

        model.init_hidden()
        batch_step += 1
        tmp_loss1   += loss1.data[0]
        tmp_loss2   += loss2.data[0]
#        print(tmp_loss1)
#        tmp_loss1  += loss1.data[0]
        out_num = 200
        if batch_step%out_num == 0 and batch_step!=0:
#            print('current_epoch:',current_epoch,'time:',time.asctime( time.localtime(time.time())).split()[3],'batch_step:',batch_step,'loss:',tmp_loss/out_num,'loss1:',tmp_loss1/out_num,'loss2:',tmp_loss2/out_num)
            print('current_epoch:',current_epoch,'time:',time.asctime( time.localtime(time.time())).split()[3],'batch_step:',batch_step,'loss1:',tmp_loss1/out_num,'loss2:',tmp_loss2/out_num)
            torch.save(model,'model/mask_gru%depoch%dbatch_step.model'%(current_epoch,batch_step)+str(tmp_loss1))
#            tmp_loss = 0.
            tmp_loss1 = 0.
            tmp_loss2 = 0.
    tmp_loss1 = 0.
    tmp_loss2 = 0.
