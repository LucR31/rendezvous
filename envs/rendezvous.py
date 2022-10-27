#libraries required for poliastro

from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import OrbitPlotter2D
from poliastro.maneuver import Maneuver

#gym and other required libraries

import logging.config
import math
import random
import gym
import numpy as np
import pkg_resources
from gym import spaces

#==================================================================

""" Goal spaceship data"""
r = [-6045, -3490, 2500] << u.km
v = [-3.457, 6.618, 2.533] << u.km / u.s
nostromo = Orbit.from_vectors(Earth, r, v)

"""Other values"""
STEP=10
PLANET_RADIUS=6371

#==============================class start=========================


class Poliastro_env(gym.Env):
    

    def __init__(self,r_1= [-6045, -3490, 2500]<< u.km,
                      v_1= [-3.457, 6.618, 2.533]<< u.km/u.s):
        
        # Initial orbit
        self.r_1 = r_1
        self.v_1 = v_1
        self.orbit = Orbit.from_vectors(Earth, r_1, v_1)
        
        #Action space 
        self.action_space = spaces.Box(low=np.array([-5, -5, -5]),
                                       high=np.array([5,5,5]),
                                       dtype=np.float16)
        
        #Obaservation space
        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(1, 1, 1),
                                       dtype=np.float16)
       
    def _get_obs(self):
        
        return {"agent": self.orbit.r.value, "target": nostromo.r.value}
    
    def step(self,action):
           
        """Take the action and make time pass"""
        dv = action << (u.m / u.s)
        self.orbit = self.orbit.apply_maneuver(Maneuver.impulse(dv))
        self.orbit=self.orbit.propagate(STEP<<u.min)
        
        """Get reward """
        reward=self.get_reward()
        
        """Get observation"""
        observation = self._get_obs()
        
        """If crash, end the task"""
        done=True if self.ground_check() or self.too_far() or reward>=10 else False
         
        return reward, done, observation
    
    def get_reward(self):
        
        dist=self.check_distance()
        
        if dist<100:
            return int(100-dist)
        return 0
              
    def ground_check(self): 
        
        perigee=self.orbit.a*(1-self.orbit.ecc) 
    
        if perigee <= PLANET_RADIUS<< u.km:
            return True    
        return False
    
    def too_far(self):
        
        return True if self.orbit.a>16000<< u.km else False
    
    def check_distance(self):
        
        """ Get the difference between both orbits
        need a formula for it"""
        
        a=self.orbit.a.value
        b=nostromo.a.value
        e=self.orbit.ecc.value
        d=nostromo.ecc.value
        
        diff=0
        for x in np.linspace(0, 6, 20):
            r_1=a*(1-e**2)/(1+e*np.cos(x))
            r_2=b*(1-d**2)/(1+d*np.cos(x))
            diff+=abs(r_1-r_2)
           
        return diff

    def reset(self):
        self.orbit=Orbit.from_vectors(Earth, self.r_1, self.v_1)
        return None

    def render(self):
        pass