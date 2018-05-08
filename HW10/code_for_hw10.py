import pdb
import numpy as np
import matplotlib.pyplot as plt
import dist
import util
import pickle
from mdp10 import MDP, TabularQ, NNQ, value_iteration, Q_learn, Q_learn_batch, greedy, sim_episode, evaluate

class No_Exit(MDP):
    # Like breakout or pong, but one player, no walls to break out, no
    # way to win You can move paddle vertically up or down or stay
    actions = (+1, 0, -1)
    
    def __init__(self, field_size, ball_speed = 1, random_start = True):
        # image space is n by n
        self.q = None
        self.n = field_size
        h = self.n * ball_speed 
        self.discount_factor = (h - 1.0) / h
        self.ball_speed = ball_speed
        # state space is: ball position and velocity, paddle position
        # and velocity
        # - ball position is n by n
        # - ball velocity is one of (-1, -1), (-1, 1), (0, -1), (0, 1),
        #                          (1, -1), (1, 1)
        # - paddle position is n; this is location of bottom of paddle,
        #    can stick "up" out of the screen
        # - paddle velocity is one of 1, 0, -1
        self.states = [((br, bc), (brv, bcv), pp, pv) for \
                         br in range(self.n) for 
                         bc in range(self.n) for
                         brv in (-1, 0, 1) for
                         bcv in (-1, 1) for 
                         pp in range(self.n) for 
                         pv in (-1, 0, 1)]
        self.states.append('over')
        self.start = dist.uniform_dist([((br, 0), (0, 1), 0, 0) \
                                        for br in range(self.n)]) \
                if random_start else  \
                dist.delta_dist(((int(self.n/2), 0), (0, 1), 0, 0))

    ax = None
    def draw_state(self, state = None, pause = False):
        if self.ax is None:
            plt.ion()
            plt.figure(facecolor="white")
            self.ax = plt.subplot()

        if state is None: state = self.state
        ((br, bc), (brv, bcv), pp, pv) = state
        im = np.zeros((self.n, self.n+1))
        im[br, bc] = -1
        im[pp, self.n] = 1
        ims = self.ax.imshow(im, interpolation = 'none',
                           cmap = 'viridis', 
                           extent = [-0.5, self.n+0.5,
                                     -0.5, self.n-0.5])
        ims.set_clim(-1, 1)
        plt.pause(0.0001) 
        if pause: input('go?')
        else: plt.pause(0.1) 

    def state2vec(self, s):
        if s == 'over':
            return np.array([[0, 0, 0, 0, 0, 0, 1]])
        ((br, bc), (brv, bcv), pp, pv) = s
        return np.array([[br, bc, brv, bcv, pp, pv, 0]])

    def terminal(self, state):
        return state == 'over'

    def reward_fn(self, s, a):
        return 0 if s == 'over' else 1
        
    def transition_model(self, s, a, p = 0.4):
        # Only randomness is in brv and brc after a bounce
        # 1- prob of negating nominal velocity
        if s == 'over':
            return dist.delta_dist('over')
        # Current state
        ((br, bc), (brv, bcv), pp, pv) = s
        # Nominal next ball state
        new_br = br + self.ball_speed*brv; new_brv = brv
        new_bc = bc + self.ball_speed*bcv; new_bcv = bcv
        # nominal paddle state, a is action (-1, 0, 1)
        new_pp = max(0, min(self.n-1, pp + a))
        new_pv = a
        new_s = None
        hit_r = hit_c = False
        # bottom, top contacts
        if new_br < 0:
            new_br = 0; new_brv = 1; hit_r = True
        elif new_br >= self.n:
            new_br = self.n - 1; new_brv = -1; hit_r = True
        # back, front contacts
        if new_bc < 0:                  # back bounce
            new_bc = 0; new_bcv = 1; hit_c = True
        elif new_bc >= self.n:
            if self.paddle_hit(pp, new_pp, br, bc, new_br, new_bc):
                new_bc = self.n-1; new_bcv = -1; hit_c = True
            else:
                return dist.delta_dist('over')

        new_s = ((new_br, new_bc), (new_brv, new_bcv), new_pp, new_pv)
        if ((not hit_c) and (not hit_r)):
            return dist.delta_dist(new_s)
        elif hit_c:                     # also hit_c and hit_r
            if abs(new_brv) > 0:
                return dist.DDist({new_s: p,
                                   ((new_br, new_bc), (-new_brv, new_bcv), new_pp, new_pv) : 1-p})
            else:
                return dist.DDist({new_s: p,
                                   ((new_br, new_bc), (-1, new_bcv), new_pp, new_pv) : 0.5*(1-p),
                                   ((new_br, new_bc), (1, new_bcv), new_pp, new_pv) : 0.5*(1-p)})
        elif hit_r:
            return dist.DDist({new_s: p,
                               ((new_br, new_bc), (new_brv, -new_bcv), new_pp, new_pv) : 1-p})


    def paddle_hit(self, pp, new_pp, br, bc, new_br, new_bc):
        # Being generous to paddle, any overlap in row
        prset = set(range(pp, pp+2)).union(set(range(new_pp, new_pp+2)))
        brset = set([br, br+1, new_br, new_br+1])
        return len(prset.intersection(brset)) >= 2

##############################
# Display
##############################
def tidy_plot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
    plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def plot_points(x, y, ax = None, clear = False, 
                  xmin = None, xmax = None, ymin = None, ymax = None,
                  style = 'or-'):

    if ax is None:
        if xmin == None: xmin = np.min(x) - 0.5
        if xmax == None: xmax = np.max(x) + 0.5
        if ymin == None: ymin = np.min(y) - 0.5
        if ymax == None: ymax = np.max(y) + 0.5
        ax = tidy_plot(xmin, xmax, ymin, ymax)

        x_range = xmax - xmin; y_range = ymax - ymin
        if .1 < x_range / y_range < 10:
            plt.axis('equal')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x, y, style, markeredgewidth=0.0)
    # Seems to occasionally mess up the limits
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, which='both')
    return ax

import functools
def toHex(s):
    lst = []
    for ch in s:
        hv = hex(ord(ch)).replace('0x', '')
        if len(hv) == 1:
            hv = '0'+hv
        lst.append(hv)
    
    return functools.reduce(lambda x,y:x+y, lst)

##############################
            
def test_learn_play(d = 6, num_layers = 2, num_units = 100,
                    eps = 0.5, iters = 10000, draw=False,
                    tabular = True, batch=False, batch_epochs=10,
                    num_episodes = 10, episode_length = 100):
    iters_per_value = 1 if iters <= 10 else int(iters / 10.0)
    scores = []
    def interact(q, iter=0):
        if iter % iters_per_value == 0:
            scores.append((iter, evaluate(game, num_episodes, episode_length,
                                          lambda s: greedy(q, s))[0]))
            print('score', scores[-1])
    game = No_Exit(d)
    if tabular:
        q = TabularQ(game.states, game.actions)
    else:
        q = NNQ(game.states, game.actions, game.state2vec, num_layers, num_units,
                epochs=batch_epochs if batch else 1)
    if batch:
        qf = Q_learn_batch(game, q, iters=iters, episode_length = 100, n_episodes=10,
                           interactive_fn=interact)
    else:
        qf = Q_learn(game, q, iters=iters, interactive_fn=interact)
    if scores:
        print('String to upload (incude quotes): "%s"'%toHex(pickle.dumps([tabular, batch, scores], 0).decode()))
        # Plot learning curve
        plot_points(np.array([s[0] for s in scores]),
                    np.array([s[1] for s in scores]))
    for i in range(num_episodes):
        reward, _ = sim_episode(game, (episode_length if d > 5 else episode_length/2),
                                lambda s: greedy(qf, s), draw=draw)
        print('Reward', reward)

def test_solve_play(d = 6, draw=False,
                    num_episodes = 10, episode_length = 100):
    game = No_Exit(d)
    qf = value_iteration(game , TabularQ(game.states, game.actions))
    for i in range(num_episodes):
        reward, _ = sim_episode(game, (episode_length if d > 5 else episode_length/2),
                                lambda s: greedy(qf, s), draw=draw)
        print('Reward', reward)



# Value Iteration
print('hey')
# test_solve_play()
# Q-learn
# test_learn_play(iters=5000000, tabular=True, batch=False)
# Batch Q-learn
# test_learn_play(iters=300, tabular=True, batch=True) # Check: why do we want fewer iterations here?
# NN Q-learn
test_learn_play(iters=40000, tabular=False, batch=False)
# Fitted Q
test_learn_play(iters=18, tabular=False, batch=True)