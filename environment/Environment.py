import cv2
import numpy as np

from environment.grabscreen import grab_screen
from environment.keypress import press, release, W,S,A,D,SPACE # SPACE can be problems

class Environment():
    def __init__(self,screen_region=(0,0,1920,1080)):
        self.REWARD_IN_N_FRAMES = 1
        self.SCREEN_REGION = screen_region
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(  winSize  = (15,15),
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  
    def reset(self):
        self.counter = 0
        self.old_state = self.get_state() # (144,256)
        self.corners = cv2.goodFeaturesToTrack(self.old_state, mask=None, **self.feature_params) # or p0
        self.delta_sum = np.array([0.,0.])
        self.done = False
        self.won = False
        self.reward_sum = 0
        self.error_count = 0
        self.old_keys = None
        return self.old_state, self.done
  
    def step(self,action): # (nA,)
        self.take_action(action)
        self.new_state = self.get_state()
        self.corners, self.delta_sum = self.optical_flow(self.old_state,self.new_state,self.corners,self.delta_sum,self.counter)
        self.reward = self.get_reward(self.delta_sum)
        if self.new_state.max() < 100:
            self.done = True
            self.won = False
        # if reached_end_goal: done=True, won=True
        self.reward_sum += self.reward
        self.old_state = self.new_state.copy()
        return self.new_state,self.reward,self.done
        # state_after_action = self.get_state()
        # return state_after_action,self.reward,self.done
    
    def stop(self):
        for key in [W,S,A,D,SPACE]:
            release(key)
    
    def get_state(self):
        screen = grab_screen(gray=True, region=self.SCREEN_REGION)
        return cv2.resize(screen, (256,144))
  
    def take_action(self,action): # (nA,)
        self.new_keys = self.action_to_keys(action)
        if self.old_keys is not None:
            for key in [W,A,S,D,SPACE]:
                release(key)
        if self.new_keys is not None:
            for key in self.new_keys:
                press(key)
        self.old_keys = self.new_keys
  
    def optical_flow(self,old_screen,screen,p0,delta_sum,counter):
        mask = np.zeros_like(old_screen)
        frame = screen.copy()
        try:
            # Calc optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_screen, screen, p0, None, **self.lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # Draw the tracks + sum_x, sum_y
            delta_coords = []
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                delta_coords.append(old-new) # why (new-old) doesnt give good result?
            delta_sum += np.mean(delta_coords,axis=0)
            # text = 'X: '+ str(np.around(delta_x,4)) +' Y: '+ str(np.around(delta_y,4))
            # _print(text)
            img = cv2.add(frame,mask)
            # Now update the previous frame and previous points
            p0 = good_new.reshape(-1,1,2)
        except Exception as e:
            print(e)
            self.error_count += 1
        # update features every 20 frames *(0.037 + 0) // (everything else + model)
        if counter%20 == 0:
            p0 = cv2.goodFeaturesToTrack(old_screen, mask=None, **self.feature_params)
        return p0, delta_sum
  
    def get_reward(self,delta_sum):
        # every N frames give reward
        reward = 0
        if self.counter%self.REWARD_IN_N_FRAMES == 0:
            # text = 'X: '+ str(np.around(delta_sum[0],4)) +' Y: '+ str(np.around(delta_sum[1],4))
            # print(text)
            if delta_sum[0] > 0.3:
                reward = 1
            elif delta_sum[0] < -0.3:
                reward = -1
            else:
                reward = -0.1
            # reset delta sum x / y
            self.delta_sum = np.array([0.,0.])
        return reward
  
    def action_to_keys(self,action): # (nA,)
        if action[0] == 1:
            return [W]
        elif action[1] == 1:
            return [S]
        elif action[2] == 1:
            return [A]
        elif action[3] == 1:
            return [D]
        elif action[4] == 1:
            return [W,A]
        elif action[5] == 1:
            return [W,D]
        elif action[6] == 1:
            return [SPACE]
        elif action[7] == 1:
            return [SPACE,D]
        elif action[8] == 1:
            return [SPACE,A]
        elif action[9] == 1:
            return None
        else:
            return None