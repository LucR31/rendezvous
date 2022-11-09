import numpy as np
from tqdm import tqdm

def play_game_train(episodes, env, model):

    list_scores=[]
   
    for episode in tqdm(range(1,episodes+1),desc="Progress…"):
    
        obs=env.reset()
        done=False
        score=0
        
        while not done:
            
            env.render()
            action,_=model.predict(obs)
            obs, reward, done, info=env.step(action)
            score+=reward
           
            
        list_scores.append(score)
    return np.array(list_scores).mean(), [int(x) for x in list_scores]
    #print("Episode:{} Score:{}".format(episode, score))
    
    
def play_game(episodes, env):

    list_scores=[]
 
    for episode in tqdm(range(1,episodes+1),desc="Progress…"):
    
        obs=env.reset()
        done=False
        score=0
        
        while not done:
            
            env.render()
            action=env.action_space.sample()
            obs, reward, done, info=env.step(action)
            score+=reward
            
            
        list_scores.append(score)
    return np.array(list_scores).mean(), list_scores
    #print("Episode:{} Score:{}".format(episode, score))