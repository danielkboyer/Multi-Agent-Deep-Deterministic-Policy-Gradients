import torch as T
import torch.nn.functional as F
from agent import Agent
import ctypes
import copy
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario+"/" 
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        T.autograd.set_detect_anomaly(True)
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards,dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            #the target actors decision for the new states
            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            #all of the actors choices for the current actor states
            all_agents_new_mu_actions.append(pi)
            #What the agent chose before
            old_agents_actions.append(actions[agent_idx])

        #all the actions of the target_actor for new states
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        #all the actions of the actor for current states
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        #what the agent chose before 
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            #clear gradients
            for param in agent.target_critic.parameters():
                param.grad = None
            for param in agent.critic.parameters():
                param.grad = None
            for param in agent.actor.parameters():
                param.grad = None

            
            
            
            

            #forward passes
            target_critic_value = agent.target_critic.forward(states_,new_actions).flatten()
            critic_value = agent.critic.forward(states,old_actions).flatten()

            actor_value = agent.critic.forward(states, mu).flatten()
            #compute new q_targets
            critic_target = rewards[:,agent_idx] + agent.gamma * target_critic_value
            #compute loss
            critic_loss = F.mse_loss(critic_target,critic_value)
            actor_loss = -T.mean(actor_value)
            #backward pass
            actor_loss.backward()
            critic_loss.backward()
            #optimizer step
            agent.critic.optimizer.step()
            agent.actor.optimizer.step()
            
           

            agent.update_network_parameters()
