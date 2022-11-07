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
import pygame

#==================================================================

"""Other values"""
STEP=1
PLANET_RADIUS=6371
BOX_LIMIT=np.inf
FUEL=1000
list_points_target=[]
list_points_agent=[]

#==============================class start=========================


class Poliastro_env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self,r_1= [-7045, -2910, 2504],
                      v_1= [-3.357, 5.728, 2.133],
                      fuel=FUEL,
                      render_mode=None, 
                      r = [-6045, -3490, 2500],
                      v = [-3.457, 6.618, 2.533]
                      ):
        r=r<< u.km
        v=v<< u.km/u.s
        self.r = r
        self.v = v
        self.target = Orbit.from_vectors(Earth, r, v)
        
        #render attributes
        self.window_size = 1000  # The size of the PyGame window
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        
        # Initial orbit14820
        r_1=r_1<< u.km
        v_1=v_1<< u.km/u.s
        self.r_1 = r_1
        self.v_1 = v_1
        self.orbit = Orbit.from_vectors(Earth, r_1, v_1)
        self.fuel=fuel
        
        #Action space 
        self.action_space = spaces.Box(low=np.array([-10, -10, -10]),
                                       high=np.array([10,10,10]),
                                       dtype=np.float16)
        
        #Obaservation space
        self.observation_space = spaces.Dict(
            
            { "agent":  spaces.Box(low=-BOX_LIMIT,
                                            high=BOX_LIMIT,
                                            shape=(3,),
                                            dtype=np.float16),
             
              "target":  spaces.Box(low=-BOX_LIMIT,
                                            high=BOX_LIMIT,
                                            shape=(3,),
                                            dtype=np.float16)
            })
        
        
        
       
    def _get_obs(self):
        return {"agent": self.orbit.r.value.astype("float16"), "target": self.target.r.value.astype("float16")}
    
    def _get_info(self):
        apo_diff,peri_diff=self.apo_peri()
        return {"distance": self.check_distance(),
                "perigee_diff": peri_diff,
                "apogee_diff": apo_diff}
    
    def apo_peri(self):
        
        a=self.orbit.a.value
        b=self.target.a.value
        e=self.orbit.ecc.value
        d=self.target.ecc.value
        
        return a*(e+1)-b*(d+1), a*(-e+1)-b*(-d+1)
        
    
    def step(self,action):
        
        dist_initial=self.check_distance()#initial distance
        
        """Take the action and make time pass"""
        dv = action << (u.m / u.s)
        self.orbit = self.orbit.apply_maneuver(Maneuver.impulse(dv))
        self.orbit=self.orbit.propagate(STEP<<u.min)
        self.target=self.target.propagate(STEP<<u.min)
        self.fuel-=1
        
        """Get observation"""
        observation = self._get_obs()
        
        """If crash, end the task"""
        done=True if self.ground_check() or self.fuel<=0 else False
        
        """Get reward """
        reward=self.get_reward(dist_initial)
        
        info=self._get_info()
        
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, done, info
    
    def get_reward(self,dist_initial):
        
        if self.check_distance()<dist_initial and self.check_distance()>10:
            return 1
        elif self.check_distance()<=10:
            return 2
        return -1
              
    def ground_check(self): 
        
        r=self.orbit.r.value
        dist=np.sqrt(r[0]**2+r[1]**2+r[2]**2)
        
        if dist <= PLANET_RADIUS:
            return True    
        return False
    
    def too_far(self):
        
        return True if self.orbit.r>BOX_LIMIT<< u.km else False
    
    
    def check_distance(self):
        
        """ Get the difference between both orbits
        need a formula for it, add inclination!!!"""
        
        a=self.orbit.a.value
        b=self.target.a.value
        e=self.orbit.ecc.value
        d=self.target.ecc.value
        
        diff=0
        for x in np.linspace(0, 6, 20):
            r_1=a*(1-e**2)/(1+e*np.cos(x))
            r_2=b*(1-d**2)/(1+d*np.cos(x))
            diff+=abs(r_1-r_2)
           
        return diff

    def reset(self):
        
        self.orbit=Orbit.from_vectors(Earth, self.r_1, self.v_1)
        self.target = Orbit.from_vectors(Earth, self.r, self.v)
        self.fuel=FUEL
        observation=self._get_obs()
        list_points_target=[]
        list_points_agent=[]
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        #(16000,16000)
        canvas.fill((0, 0, 0))
        
        font = pygame.font.SysFont(None, 24)
        img1 = font.render('Target', True, (255,255,255))
        img2 = font.render('Agent', True, (255,255,255))
        
        dist=round(self.check_distance(),1)
        img3 = font.render(f'Diff: {dist}', True, (255,255,255))
        img4 = font.render(f'Fuel: {self.fuel}', True, (255,255,255))
        img5 = font.render(f'Inc. Diff: {round(abs(self.orbit.inc.value-self.target.inc.value),4)} rad', True, (255,255,255))
        img6 = font.render(f'A. Node. Diff: {round(abs(self.orbit.raan.value-self.target.raan.value),4)} rad', True, (255,255,255))
        
        
        canvas.blit(img3, (50,50))
        canvas.blit(img4, (50,67))
        canvas.blit(img5, (50,87))
        canvas.blit(img6, (50,107))
        
        # planet
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            [500,500],90,
            0,
        )
        
    
        # First we draw the target
        r_target=(self.target.r.value+[8000,8000,0])*0.05
        pygame.draw.circle(
            canvas,
            (0,255,0),
            [int(r_target[0]),int(r_target[1])],10,
            0,
        )
        canvas.blit(img1, [int(r_target[0]),int(r_target[1])])
        
        
        #drawing traceline
        list_points_target.append([int(r_target[0]),int(r_target[1])])
        
        if len(list_points_target)>FUEL//3:
            del list_points_target[0]
            
        for i in range(len(list_points_target)):
            pygame.draw.circle(
            canvas,
            (0,255,0),
            [list_points_target[i][0],list_points_target[i][1]],1,
            0,)
         
        
       
        # Now we draw the agent
        r_agent=(self.orbit.r.value+[8000,8000,0])*0.05
        
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            [int(r_agent[0]),int(r_agent[1])],10,
            0,
        )
        canvas.blit(img2, [int(r_agent[0]),int(r_agent[1])])
        
        #drawing traceline 2
        list_points_agent.append([int(r_agent[0]),int(r_agent[1])])
        if len(list_points_agent)>FUEL//3:
            del list_points_agent[0]
        for i in range(len(list_points_agent)):
            pygame.draw.circle(
            canvas,
            (255,0,0),
            [list_points_agent[i][0],list_points_agent[i][1]],1,
            0,)
        

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
            
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
        
    def close(self):
            if self.window is not None:
                pygame.display.quit()
                pygame.quit()