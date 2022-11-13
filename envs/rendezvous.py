#libraries required for poliastro

from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import OrbitPlotter2D
from poliastro.maneuver import Maneuver
from poliastro.constants import GM_earth
from poliastro.core.elements import eccentricity_vector,rv_pqw

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

k = GM_earth.value
STEP=2 #min
PLANET_RADIUS=6371 #km
BOX_LIMIT=np.inf
TRACE_POINTS=1000
list_points_target=[]
list_points_agent=[]
FUEL=500
DV=10

#==============================class starts========================
def vect_l(vector):
    return np.sqrt(sum([x**2 for x in vector]))

class Poliastro_env(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self,r_1= [-6045, -3490, 2500],#position vector agent (km)
                      v_1= [-3.657, 6.818, 2.633], #velocity vector agent (km/s)
                      r = [-6045, -3490, 2500], #position vector target (km)
                      v = [-3.557, 6.718, 2.633],#velocity vector target (km/s)
                      distance="euclidean",
                      fuel=FUEL,
                      render_mode=None, 
                      ):
        
        # orbit distance
        self.distance=distance
        
        #render attributes
        self.window_size = 1000  # The size of the PyGame window
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # orbit target and agent
        self.r_1=r_1<< u.km
        self.v_1=v_1<< u.km/u.s
        self.r=r<< u.km
        self.v=v<< u.km/u.s
        self.orbit = Orbit.from_vectors(Earth, self.r_1, self.v_1)
        self.target = Orbit.from_vectors(Earth, self.r, self.v)
        self.fuel=fuel
        
        
        #Action space 
        self.action_space = spaces.Box(low=np.array([-DV, -DV, -DV]),
                                       high=np.array([DV,DV,DV]),
                                       dtype=np.float16)
        #Obaservation space
        self.observation_space = spaces.Dict(
            
            { "agent":  spaces.Box(low=-BOX_LIMIT,
                                            high=BOX_LIMIT,
                                            shape=(1,),
                                            dtype=np.float16),
             
              "target":  spaces.Box(low=-BOX_LIMIT,
                                            high=BOX_LIMIT,
                                            shape=(2,),
                                            dtype=np.float16)
            })
        
    def _get_obs(self):
        #return {"agent": self.orbit.r.value.astype("float16"), "target": self.target.r.value.astype("float16")}
        d1,d2=self.check_distance()
        return {"agent":np.array([self.orbit.nu.value]), "target":np.array([d1,d2])}
        
    def _get_info(self):
        apo_diff,peri_diff=self.apo_peri()
        
        if self.distance=="euclidean":
            dist_initial=self.distance_check_2()
        else:
            dist_initial=self.check_distance()
            
        return {"distance": dist_initial,
                "perigee_diff": peri_diff,
                "apogee_diff": apo_diff,
                "fuel":self.fuel,
                "gound_dist": vect_l(self.orbit.r.value)-PLANET_RADIUS}
    
    def apo_peri(self):  
        a,e = self.orbit_params()[0],self.orbit_params()[1]
        apoapsis_diff=a[0]*(e[0]+1)-a[1]*(e[1]+1)
        periapsis_diff=a[0]*(-e[0]+1)-a[1]*(-e[1]+1)
        return apoapsis_diff,periapsis_diff
    
    def orbit_params(self):
        a=(self.orbit.a.value,self.target.a.value)
        ecc=(self.orbit.ecc.value,self.target.ecc.value)
        inc=(self.orbit.inc.value,self.target.inc.value)
        raan=(self.orbit.raan.value,self.target.raan.value)
        argp=(self.orbit.argp.value,self.target.argp.value)
        nu=(self.orbit.nu.value,self.target.nu.value)    
        return [a,ecc,inc,raan,argp,nu]
     
    def step(self,action):
        
        """set inital distance"""
        if self.distance=="euclidean":
            dist_initial=self.distance_check_2()
        else:
            d1,d2=self.check_distance()
            dist_initial=d1+d2
            
        """Take the action and make time pass"""
        dv = action << (u.m / u.s)
        
        self.orbit = self.orbit.apply_maneuver(Maneuver.impulse(dv))
        self.orbit=self.orbit.propagate(STEP<<u.min)
        self.target=self.target.propagate(STEP<<u.min)
        
        """fuel consumption"""
        self.fuel-=vect_l(dv.value)*0.1
        
        """Get observation"""
        observation = self._get_obs()
        
        """If crash or no fuel end the task"""
        #done=True if self.fuel<=0 else False
        done=True if self.ground_check() or self.fuel<=0 or self.get_reward==1 else False
        """Get reward """
        reward=self.get_reward(dist_initial)
       
        """INFO"""
        #info=self._get_info()
        info={}      
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, done, info
    
    def get_reward(self,dist_initial):
        
        if self.distance=="euclidean":
            dist_=self.distance_check_2()
        else:
            d1,d2=self.check_distance()
        
        if d1+d2<dist_initial:
            return 1
        return -1
    
    def get_reward_2(self):
            
        if self.distance=="euclidean":
            dist_=self.distance_check_2()
        else:
            dist_=self.check_distance()
            
        if dist_<=100000:
            return 1
        return 0
              
    def ground_check(self):        
        return True if vect_l(self.orbit.r.value) <= PLANET_RADIUS else False    
       
    def distance_check_2(self):
        return np.sqrt(sum([(x[0]-x[1])**2 for x in self.orbit_params()]))
    
    def check_distance(self):
        
        ecc_t=self.target.ecc
        nu_t=self.target.nu
        p_t=self.target.p
        
        ecc_o=self.orbit.ecc
        nu_o=self.orbit.nu
        p_o=self.orbit.p

        r_t, v_t = rv_pqw(k, p_t, ecc_t, nu_t)
        v_ecc_t=eccentricity_vector(k,r_t,v_t)
        
        r_o, v_o = rv_pqw(k, p_o, ecc_o, nu_o)
        v_ecc_o=eccentricity_vector(k,r_o,v_o)
        
        t_p=self.target.r_p.value
        t_a=self.target.r_a.value
        o_p=self.orbit.r_p.value
        o_a=self.orbit.r_a.value
        
        diff_a=vect_l(t_a*v_ecc_t-o_a*v_ecc_o)
        diff_p=vect_l(t_p*v_ecc_t-o_p*v_ecc_o)
        
        return diff_a, diff_p
        
        
    def dat_stats(self):
        return [x[0]-x[1] for x in self.orbit_params()]
        
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
    
    
#=================RENDER TERRITORY============================

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
        
        d1,d2=self.check_distance()
        dist=round(d1+d2,1)
        img3 = font.render(f'Diff: {dist}', True, (255,255,255))
        img4 = font.render(f'Fuel: {round(self.fuel,2)}', True, (255,255,255))
        img5 = font.render(f'Inc. Diff: {round(abs(self.orbit.inc.value-self.target.inc.value),3)} rad', True, (255,255,255))
        img6 = font.render(f'A. Node. Diff: {round(abs(self.orbit.raan.value-self.target.raan.value),3)} rad', True, (255,255,255))
        
        
        canvas.blit(img3, (50,50))
        canvas.blit(img4, (50,67))
        canvas.blit(img5, (50,87))
        canvas.blit(img6, (50,107))
        
        #================panel two=================
        a_=font.render(f'a: {round(self.orbit.a.value,2)}', True, (255,0,0))
        e_=font.render(f'ecc: {round(self.orbit.ecc.value,2)}', True, (255,0,0))
        i_=font.render(f'inc: {round(self.orbit.inc.value,2)}', True, (255,0,0))
        
        a_t=font.render(f'a: {round(self.target.a.value,2)}', True, (0,255,0))
        e_t=font.render(f'ecc: {round(self.target.ecc.value,2)}', True, (0,255,0))
        i_t=font.render(f'inc: {round(self.target.inc.value,2)}', True, (0,255,0))
        
        canvas.blit(a_, (900,50))
        canvas.blit(e_, (900,67))
        canvas.blit(i_, (900,87))
        
        canvas.blit(a_t, (800,50))
        canvas.blit(e_t, (800,67))
        canvas.blit(i_t, (800,87))
        
        
        # planet
        pygame.draw.circle(
            canvas,
            (51,153,255),
            [500,500],PLANET_RADIUS*0.03,
            0,
        )
       
        # First we draw the target
        r_target=(self.target.r.value+[16700,16700,0])*0.03
        pygame.draw.circle(
            canvas,
            (0,255,0),
            [int(r_target[0]),int(r_target[1])],10,
            0,
        )
        canvas.blit(img1, [int(r_target[0]),int(r_target[1])])
        
        
        #drawing traceline
        list_points_target.append([int(r_target[0]),int(r_target[1])])
        
        if len(list_points_target)>TRACE_POINTS:
            del list_points_target[0]
            
        for i in range(len(list_points_target)):
            pygame.draw.circle(
            canvas,
            (0,255,0),
            [list_points_target[i][0],list_points_target[i][1]],1,
            0,)
         
        
       
        # Now we draw the agent
        r_agent=(self.orbit.r.value+[16700,16700,0])*0.03
        
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            [int(r_agent[0]),int(r_agent[1])],10,
            0,
        )
        canvas.blit(img2, [int(r_agent[0]),int(r_agent[1])])
        
        #drawing traceline 2
        list_points_agent.append([int(r_agent[0]),int(r_agent[1])])
        if len(list_points_agent)>TRACE_POINTS:
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