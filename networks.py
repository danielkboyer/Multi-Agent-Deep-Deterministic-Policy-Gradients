import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = chkpt_dir
        self.name = name
        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        #self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
      

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        q = self.q(x)

        return q

    def save_checkpoint(self,good,dir):
        if dir != "":
            path = os.path.join(self.chkpt_file,dir)
            if not os.path.isdir(path):
                os.mkdir(path)
            T.save(self.state_dict(), self.chkpt_file+dir+"/"+self.name)
            return
        if good:
            T.save(self.state_dict(), self.chkpt_file+"good/"+self.name)
        else:
            T.save(self.state_dict(), self.chkpt_file+"bad/"+self.name)

    def load_checkpoint(self,good,dir):
        if dir != "":
            self.load_state_dict(T.load(self.chkpt_file+dir+"/"+self.name))
        if good:
            self.load_state_dict(T.load(self.chkpt_file+"good/"+self.name))
        else:
            self.load_state_dict(T.load(self.chkpt_file+"bad/"+self.name))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.name = name
        self.chkpt_file = chkpt_dir
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        #self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #changes the tensor to values between [0,1] scaled appropiately
        pi = T.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self,good,dir):
        if dir != "":
            path = os.path.join(self.chkpt_file,dir)
            if not os.path.isdir(path):
                os.mkdir(path)
            T.save(self.state_dict(), self.chkpt_file+dir+"/"+self.name)
            return
        if good:
            T.save(self.state_dict(), self.chkpt_file+"good/"+self.name)
        else:
            T.save(self.state_dict(), self.chkpt_file+"bad/"+self.name)

    def load_checkpoint(self,good,dir):
        if dir != "":
            self.load_state_dict(T.load(self.chkpt_file+dir+"/"+self.name))
        if good:
            self.load_state_dict(T.load(self.chkpt_file+"good/"+self.name))
        else:
            self.load_state_dict(T.load(self.chkpt_file+"bad/"+self.name))

