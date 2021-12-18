import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from make_env import make_env
import time
import graph as g
import sys
import os
def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state



if __name__ == '__main__':

    
    #scenario = 'simple'
    scenario = 'food_compete'
    env = make_env(scenario)
    n_agents = env.n
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims)
    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.action_space[0].n
    folder_number = sys.argv[1]
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=512, fc2=512,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir=f'C:/Users/dedaq/Documents/MADDPGV2/Multi-Agent-Deep-Deterministic-Policy-Gradients/tmp/maddpg{folder_number}/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=4096)
    PRINT_INTERVAL = 2
    N_GAMES = 50000
    total_steps = 0
    good_score_history = []
    bad_score_history = []
    
    good_best_score = -1000000
    bad_best_score = -1000000
    
    evaluate = True if sys.argv[2] == '1' else False
    load_good = True if sys.argv[3] == '1' else False
    load_check_point = False
    make_movie = True
    

    #print(sys.argv[0])
    #graph variables
    episodeListG = []
    avgGoodScoreListG = []
    avgBadScoreListG = []
    if evaluate:
        maddpg_agents.load_checkpoint(load_good,"")
    if load_check_point:
        maddpg_agents.load_checkpoint(load_good,"")
    if make_movie:
        movie_number = 100
        maddpg_agents.load_checkpoint(False,str(movie_number))
        base_path = f'C:/Users/dedaq/Documents/MADDPGV2/Multi-Agent-Deep-Deterministic-Policy-Gradients/tmp/maddpg{folder_number}/{scenario}/'
        while os.path.isdir(base_path+str(movie_number)):
            maddpg_agents.load_checkpoint(False,str(movie_number))
            obs = env.reset()
            done = [False]*n_agents
            while not any(done):
                env.render(title="Episode "+str(movie_number))
                time.sleep(0.01)

                actions = maddpg_agents.choose_action(obs)
                obs_, reward, done, info = env.step(actions)

                obs = obs_
            movie_number+=200
        exit(0)
    for i in range(N_GAMES):
        obs = env.reset()
        good_score = 0
        bad_score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()
                time.sleep(0.01) # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

           

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            # if episode_step %50 == 0 and episode_step != 0:
            #    done = [True]*n_agents
            obs = obs_
            
            for j,info_d in enumerate(info['n']):
                
                if info_d == True:
                    good_score += reward[j]
                else:
                    bad_score +=reward[j]
            
            total_steps += 1
            episode_step += 1
            
            
        
        good_score_history.append(good_score)
        bad_score_history.append(bad_score)
        good_avg_score = np.mean(good_score_history[-50:])
        bad_avg_score = np.mean(bad_score_history[-50:])
        
        if not evaluate:
            #so we don't store the first bad lucky experiences
            if i > 50:
                if good_avg_score > good_best_score:
                    maddpg_agents.save_checkpoint(True,"")
                    good_best_score = good_avg_score
                if bad_avg_score > bad_best_score:
                    maddpg_agents.save_checkpoint(False,"")
                    bad_best_score = bad_avg_score
            if i % 100 == 0 and i > 0:
                maddpg_agents.save_checkpoint(False,str(i))
        episodeListG.append(i)
        avgGoodScoreListG.append(good_avg_score)
        avgBadScoreListG.append(bad_avg_score)
        if i % PRINT_INTERVAL == 0:
            print('episode', i, 'good average score {:.1f}'.format(good_avg_score))
            print('episode', i, 'bad average score {:.1f}'.format(bad_avg_score))
            
        if i % 50 == 0:
            
            if not evaluate:
                g.save_avg_graph(episodeListG,avgGoodScoreListG,avgBadScoreListG)
