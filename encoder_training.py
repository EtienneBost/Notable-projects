import torch
from torch.utils.data import Dataset,DataLoader
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import RandomSampler
from torch.nn.functional import normalize
from torch.fft import fft



nb_trams_per_example = 10




def transform_mfcc(mfcc,trams_per_data):          #Takes a mfcc and returns sub mfcc on a few trams
  non_zero_indexes=[]
  sub_spec = []
  p = mfcc.shape[0]
  n = mfcc.shape[1]
      
  #tram_length = ((176-42) // trams_per_data)         #On average, sound starts at sample 42 and ends at 176 over 218 samples
  tram_length = n // trams_per_data
  #print('tram length' , tram_length)
  #j=42
  j=0
  while len(sub_spec)<trams_per_data:                      #Add sub mfcc at tram_length interval

    sub_spec.append(torch.flatten(mfcc[:,j:j+tram_length]).float())
    j+=tram_length

  d = torch.stack(sub_spec)
  #print('d',d.shape)
  return d

def to_IR(label):
        IRduration=0.05   #Duration of the impulse response before it hits 0
        sr = 22050
        T = torch.linspace(0,IRduration,int(IRduration * sr))
        a = label[1:11]
        b = label[11:21]
        w = label[21:31]
     
        H = torch.tensor([])

        for t in T:
            b = b * t
            w = w * t
            w = torch.cos(w)
            s = a - b
            s = s/20
            s = 10 ** s
            s = s * w

            H = torch.cat((H,torch.tensor([torch.sum(s).item()])),0)

        fft_label = fft(H)
        IR = torch.abs(fft_label)
        return IR
        



class scrapingDataset(Dataset):
    def __init__(self, input_dir):
        super(Dataset, self).__init__()
        self.input_dir = input_dir
        
        self.samples = []
        for filename in os.listdir(input_dir):
          index = filename[:-4]                #get the filename without the extention
          if filename[-4:len(filename)] != '.npy':
            self.samples.append([index+'.wav',index+'.npy'])
        
                        
    def __getitem__(self, index):

        y, sr = librosa.load(self.input_dir+self.samples[index][0])
        
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrorgam= librosa.power_to_db(spectrogram, ref=np.max)

        label = torch.tensor(np.load(self.input_dir+self.samples[index][1])).float()
        len_v = (len(label) - 62) //2
        vx = np.array(label[62 : len_v + 62])
        vy = np.array(label[len_v + 62 : len(label)])
        #print('original: ',len(vx) , len(vy))

        vx = librosa.resample(vx,orig_sr = len(vx)/3,target_sr = 100)
        vy = librosa.resample(vy,orig_sr = len(vy)/3,target_sr = 100)
        #print('resampled: ',len(vx) , len(vy))
 
        return {'sound_file': torch.flatten(torch.tensor(spectrogram)) ,
                'label': torch.tensor(np.concatenate((label[:62] , vx , vy)))}
    
    def __len__(self):
        return len(self.samples)



def generate_batch(batch):
  specs=[]
  labels_m=[]
  labels_surface=[]
  labels_ir=[]
  labels_traj=[]
  if len(batch) > 0:
    for sample in batch:
        if len(sample['sound_file']) == 17152 and len(sample['label']) > 300:
          specs.append(sample['sound_file'])
          labels_m.append(sample['label'][0])
          labels_surface.append(sample['label'][1])
          labels_ir.append(sample['label'][2:62])
          labels_traj.append(sample['label'][62:]) 
    return {'sound_batch': torch.stack(specs) , 'label_mass_batch' :torch.stack(labels_m), 'label_surf_batch' :torch.stack(labels_surface), 'label_ir_batch' :torch.stack(labels_ir), 'label_traj_batch' :torch.stack(labels_traj)}



train_directory ='train/'
dev_directory = 'dev/'




train_set  = scrapingDataset(dev_directory)
dev_set  = scrapingDataset(dev_directory)

sampler = RandomSampler(train_set, replacement=True, num_samples=10000)
sampler2 = RandomSampler(dev_set , replacement=True , num_samples = 1000)


test = dev_set.__getitem__(9)            #Display Data

spec = test['sound_file']

print("Spec test shape ",spec.shape)


print("Label SHAPE", np.shape(test['label']))



batch_size = 64
train_loader = DataLoader(train_set , sampler=sampler , batch_size=batch_size, collate_fn=generate_batch , drop_last=False)
valid_loader = DataLoader(dev_set , sampler=sampler2, batch_size = batch_size , collate_fn = generate_batch , drop_last=True)

print('Size of training set:',len(train_set))



import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft

def H(t,a,b,w):        #t:= float
                       #a,b,w := 1-d tensor of same dimension 

  b = b * t
  w = w * t
  w = torch.cos(w)
  s = a - b
  s = s/20
  s = 10 ** s
  s = s * w
  return torch.sum(s)


class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim1,output_dim2):
        super().__init__()
        self.fc1 = nn.Linear( input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.3)
        
        self.traj = nn.Linear(hidden_dim , output_dim2)
        self.ir = nn.Linear(hidden_dim , output_dim1)
        self.surf = nn.Linear(hidden_dim , 1)
        self.mass = nn.Linear(hidden_dim , 1)
    def forward(self, x):

        return self.mass(self.fc3(self.fc2(self.fc1(x)))) , self.surf(self.fc3(self.fc2(self.fc1(x)))) , self.ir(self.fc3(self.fc2(self.fc1(x)))) , self.traj(self.fc3(self.fc2(self.fc1(x))))







input_dim = 17152
hidden_dim = 1024
output_dim1 = 60
output_dim2 = 600
net = Net(input_dim,hidden_dim,output_dim1,output_dim2)
print(net.parameters)
net.load_state_dict(torch.load("Backup model"))






#print(torch.max(d['sound_file']))

class multimodal_loss(nn.Module):
    def __init__(self):
        super(multimodal_loss, self).__init__()
        
    def forward(self, o_mass , o_surf , o_ir ,o_traj , t_mass , t_surf , t_ir , t_traj):

          #print(o_mass.shape , t_mass.shape , o_ir.shape , t_ir.shape , o_traj.shape , t_traj.shape)
       
          m_loss = torch.sqrt(torch.mean((o_mass - t_mass)**2))

          surface_loss = torch.sqrt(torch.mean((o_surf - t_surf)**2))

          ir_loss = torch.sqrt(torch.mean((o_ir  - t_ir)**2))

          traj_loss = torch.sqrt(torch.mean((o_traj  - t_traj)**2))
  

          loss = m_loss + surface_loss + ir_loss / 2000 + traj_loss


          return loss


def custom_MSE(o_mass , o_surf , o_ir ,o_traj , t_mass , t_surf , t_ir , t_traj):
          m_loss = torch.sqrt((o_mass - t_mass)**2)
          
          surface_loss = torch.sqrt((o_surf - t_surf)**2)
          print(o_ir.shape , t_ir.shape)
          ir_loss = torch.sqrt(torch.mean((o_ir  - t_ir)**2))
          traj_loss = torch.sqrt(torch.mean((o_traj  - t_traj)**2))
          loss = m_loss + surface_loss + ir_loss / 2000 + traj_loss
          return loss




import torch.optim as optim
print("Working with GPU : {}".format(torch.cuda.is_available()))

def perf(model,loader,criterion1,criterion2,criterion3,criterion4):
  
  model.eval()
  total_loss1 = total_loss2 = total_loss3 = total_loss4 = num_loss  = 0
  for batch in loader:
    x = batch['sound_batch']
    t_mass=batch['label_mass_batch'].reshape(-1,1)
    t_surf=batch['label_surf_batch'].reshape(-1,1)
    t_ir=batch['label_ir_batch']
    t_traj=batch['label_traj_batch']



    with torch.no_grad():
      o_mass , o_surf , o_ir , o_traj  = model(x)
      
      loss1 = criterion1(o_mass,t_mass)
      #print(o_surf.shape , t_surf.shape)
      loss2 = criterion2(o_surf,t_surf)
      #print(o_ir.shape , t_ir.shape)
      loss3 = criterion3(o_ir,t_ir)
      loss4 = criterion4(o_traj,t_traj)
      #loss = loss1 + loss2 + loss3 / 2000 + loss4

        



      total_loss1 += loss1
      total_loss2 += loss2
      total_loss3 += loss3
      total_loss4 += loss4




      num_loss += len(batch)
  return total_loss1 / num_loss , total_loss2 / num_loss , total_loss3 / num_loss , total_loss4 / num_loss




def fit(model, nb_epochs, train_loader, valid_loader,):
  criterion = multimodal_loss()
  criterion1 = nn.MSELoss()
  criterion2 = nn.MSELoss()
  criterion3 = nn.MSELoss()
  criterion4 = nn.MSELoss()
  #criterion = custom_MSE
  optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)
  print("Epoch    Train loss   Valid loss")
  train_losses1 = []
  dev_losses1 = []
  train_losses2 = []
  dev_losses2 = []
  train_losses3 = []
  dev_losses3 = []
  train_losses4 = []
  dev_losses4 = []
  epochs=[]
  for epoch in range(nb_epochs):
    model.train()
    epoch_loss1 = epoch_loss2 = epoch_loss3 = epoch_loss4 = num =0
    for batch in train_loader:
        X = batch['sound_batch']

        t_mass=batch['label_mass_batch'].reshape(-1,1)
        t_surf=batch['label_surf_batch'].reshape(-1,1)
        t_ir=batch['label_ir_batch']
        t_traj=batch['label_traj_batch']
          
        optimizer.zero_grad()
        o_mass , o_surf , o_ir , o_traj  = model(X)

        loss1 = criterion1(o_mass,t_mass)
        loss2 = criterion2(o_surf,t_surf)
        loss3 = criterion3(o_ir,t_ir)
        loss4 = criterion4(o_traj,t_traj)
        loss = loss1 + loss2 + loss3 / 2000 + loss4

        epoch_loss1 += loss1.item()
        epoch_loss2 += loss2.item()
        epoch_loss3 += loss3.item()
        epoch_loss4 += loss4.item()


        
        #loss.requires_grad = True
        loss.backward()
        optimizer.step()
        #print(loss.item())
        num+=len(batch)
        #print(loss.item())
    train_loss1 = epoch_loss1 /num
    train_loss2 = epoch_loss1 /num
    train_loss3 = epoch_loss1 /num
    train_loss4 = epoch_loss1 /num
    train_losses1.append(train_loss1)
    train_losses2.append(train_loss2)
    train_losses3.append(train_loss3)
    train_losses4.append(train_loss4)

    dev_loss1 , dev_loss2 , dev_loss3 , dev_loss4  = perf(model,valid_loader,criterion1,criterion2,criterion3,criterion4)
    dev_losses1.append(dev_loss1.item())
    dev_losses2.append(dev_loss2.item())
    dev_losses3.append(dev_loss3.item())
    dev_losses4.append(dev_loss4.item())

    dev_loss = dev_loss1 + dev_loss2 + dev_loss3 / 2000 + dev_loss4
    train_loss = train_loss1 + train_loss2 + train_loss3 + train_loss4
    epochs.append(epoch)
    print(epoch , train_loss , dev_loss.item() )
    if epoch>1:
        torch.save(model.state_dict(), "Backup model")
        np.save('train_loss_mass',train_losses1)
        np.save('train_loss_surf',train_losses2)
        np.save('train_loss_ir',train_losses3)
        np.save('train_loss_traj',train_losses4)
        
        np.save('dev_loss_mass',dev_losses1)
        np.save('dev_loss_surf',dev_losses2)
        np.save('dev_loss_ir',dev_losses3)
        np.save('dev_loss_traj',dev_losses4)
        
        np.save('epochs',epochs)
  return epochs,train_losses1 , dev_losses1  ,train_losses2 , dev_losses2  ,train_losses3 , dev_losses3  ,train_losses4 , dev_losses4  




epochs,train_losses1 , dev_losses1  ,train_losses2 , dev_losses2  ,train_losses3 , dev_losses3  ,train_losses4 , dev_losses4  = fit(net , 1000 , train_loader , valid_loader)


np.save('train_loss_mass',train_losses1)
np.save('train_loss_surf',train_losses2)
np.save('train_loss_ir',train_losses3)
np.save('train_loss_traj',train_losses4)

np.save('dev_loss_mass',dev_losses1)
np.save('dev_loss_surf',dev_losses2)
np.save('dev_loss_ir',dev_losses3)
np.save('dev_loss_traj',dev_losses4)

np.save('epochs',epochs)






