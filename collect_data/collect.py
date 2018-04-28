import cv2
import numpy as np
from time import sleep, time
from keypress import press, release, U,D,L,R,C # C can be problems
from keycheck import key_check
from grabscreen import grab_screen

DATASET_FOLDER = 'E:/Datasets/Limbo/test/'
SCREEN_REGION = (0,0,1920,1080)
IMAGES_IN_ONE_FILE = 2500

errors = 0

class Environment():

    def get_state(self):
        screen = grab_screen(gray=True, region=SCREEN_REGION)
        screen = cv2.resize(screen, (256,144))
        return screen

    def show_screen(self,screen):
        cv2.imshow('window',cv2.resize(screen,(640,360)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            self.paused = True
            print('PAUSED')

    def get_reward(self, delta_sum, counter):
        reward = 0
        # every 1 frames give reward
        if counter%1 == 0:
            # print delta sum
            # text = 'X: '+ str(np.around(delta_sum[0],4)) +' Y: '+ str(np.around(delta_sum[1],4))
            # print(text)
            # print reward (if x_coord > 0 -> +1)
            if delta_sum[0] > 0.3:
                reward = 1
            elif delta_sum[0] < -0.3:
                reward = -1
            else:
                reward = -0.1
            # reset delta sum x / y
            delta_sum = np.array([0.,0.])

        return reward, delta_sum

    def run(self):
        global errors
        # counter + delta_sum_x/y(every 30 frames) + init screen + corners + mask(for drawing)
        counter = 1
        delta_sum = np.array([0.,0.])
        old_screen = self.get_state()
        p0 = cv2.goodFeaturesToTrack(old_screen, mask=None, **feature_params)
        D = []
        file_iter = 1
        times = []
        self.paused = True
        while True:
            ''' 1 loop in (0.0467 + 0) // (everything else (cv2) + model) '''
            if not self.paused:
                s_time = time()
                # Screen
                screen = self.get_state()
                # Output
                current_active_keys = key_check()
                output = keys_to_output(current_active_keys)
                # Reward
                p0, delta_sum = optical_flow(old_screen,screen,p0,delta_sum, counter)
                reward, delta_sum = self.get_reward(delta_sum, counter)
                # if screen is black
                if screen.max() < 100:
                    print('YOU LOST')
                # Save
                D.append([screen,output,reward])
                if len(D) > IMAGES_IN_ONE_FILE-1: # ~2mins
                    file_name = DATASET_FOLDER + 'D{}.npy'.format(file_iter)
                    np.save(file_name,D)
                    D = []
                    file_iter += 1
                # Show screen
                # self.show_screen(screen)
                # Time it
                print(time()-s_time)
                # times.append(time()-s_time)
                # if counter%5 == 0:
                #     print(screen.max())
                
                # times.append(time()-s_time)
                # if counter%400 == 0:
                #     print('DONE')
                #     print(np.max(numbers))
                #     print(np.min(numbers))
                #     break

                old_screen = screen.copy()
                counter += 1
            else:
                sleep(1)
                print('errors -',errors,'counts -',counter)
            self.paused = check_pause(self.paused)

def optical_flow(old_screen, screen, p0, delta_sum, counter):
    global errors
    mask = np.zeros_like(old_screen)
    frame = screen.copy()
    try:
        # Calc optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_screen, screen, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # Draw the tracks + sum_x, sum_y
        delta_coords = []
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            delta_coords.append(old-new) # why (new-old) doesnt give good result?
            # Create some random colors
            color = np.random.randint(0,255,(100,3))
            mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 1)
            frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
        delta_sum += np.mean(delta_coords,axis=0)
        # text = 'X: '+ str(np.around(delta_x,4)) +' Y: '+ str(np.around(delta_y,4))
        # _print(text)
        img = cv2.add(frame,mask)
        # Now update the previous frame and previous points
        p0 = good_new.reshape(-1,1,2)
    except Exception as e:
        print(e)
        errors += 1
    # update features every 20 frames *(0.037 + 0) // (everything else + model)
    if counter%20 == 0:
        p0 = cv2.goodFeaturesToTrack(old_screen, mask=None, **feature_params)

    return p0, delta_sum

def keys_to_output(keys):
    '''
    Convert keys to a one-hot vector
     0  1  2  3  4   5     6       7        8        9
    [W, S, A, D, WA, WD, SPACE, SPACE+D, SPACE+A, NOTHING] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output[4] = 1
    elif 'W' in keys and 'D' in keys:
        output[5] = 1
    elif ' ' in keys and 'D' in keys:
        output[7] = 1
    elif ' ' in keys and 'A' in keys:
        output[8] = 1
    elif 'W' in keys:
        output[0] = 1
    elif 'S' in keys:
        output[1] = 1
    elif 'A' in keys:
        output[2] = 1
    elif 'D' in keys:
        output[3] = 1
    elif ' ' in keys:
        output[6] = 1
    else:
        output[9] = 1
    return np.array(output)

def check_pause(paused):
    if 'P' in key_check():
        if paused:
            print('STARTED')
            sleep(1)
            return False
        else:
            print('PAUSED')
            sleep(1)
            return True
    else:
        return paused

def wait(secs):
    for i in reversed(range(secs)):
        print(i+1)
        sleep(1)

def _print(string):
    print(string, end='')
    print('\b' * len(string), end='', flush=True)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

try:
    env = Environment()
    env.run()
except KeyboardInterrupt:
    print('\n')