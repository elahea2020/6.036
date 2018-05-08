import pdb
import random
import numpy as np
from dist import uniform_dist, delta_dist, mixture_dist
from util import argmax_with_val, argmax
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

class MDP:
    # Needs the following attributes:
    # states: list or set of states
    # actions: list or set of actions
    # discount_factor: real, greater than 0, less than or equal to 1
    # start: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states
    # These are functions:
    # transition_model: function from (state, action) into DDist over next state
    # reward_fn: function from (state, action) to real-valued reward

    def __init__(self, states, actions, transition_model, reward_fn, 
                     discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s,a) and new state, drawn from transition.  If a
    # terminal state is encountered, sample next state from initial
    # state distribution
    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())

# Perform value iteration on an MDP, also given an instance of a q
# function.  Terminate when the max-norm distance between two
# successive value function estimates is less than eps.
# interactive_fn is an optional function that takes the q function as
# argument; if it is not None, it will be called once per iteration,
# for visuzalization

# The q function is typically an instance of TabularQ, implemented as a
# dictionary mapping (s, a) pairs into Q values This must be
# initialized before interactive_fn is called the first time.

def value_iteration(mdp, q, eps = 0.01, max_iters = 10000):
    q_old = q.copy()
    q_new = q
    gamma = mdp.discount_factor
    for i in range(max_iters):
        max_dif = 0
        for a in mdp.actions:
            for s in mdp.states:
                transition_model = mdp.transition_model(s, a)  # this is only func of a'
                prev_q = 0
                for a_p in transition_model.support():
                    prev_q += transition_model.prob(a_p) * value(q_old, a_p)
                v = mdp.reward_fn(s, a) + gamma * prev_q
                before = q_old.get(s, a)
                q_new.set(s, a, v)
                max_dif = max(max_dif, abs(before - v))
        if max_dif < eps:
            return q_new
        q_old = q_new
        q_new = q_new.copy()
    return q_new


# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    max_v = None
    for a in q.actions:
        if max_v is None or max_v < q.get(s, a):
            max_v = q.get(s, a)
    return max_v

# Given a state, return the action that is greedy with reespect to the
# current definition of the q function
def greedy(q, s):
    pi = None
    best_a = None
    for a in q.actions:
        if pi is None or pi < q.get(s, a):
            pi = q.get(s, a)
            best_a = a
    return best_a

def epsilon_greedy(q, s, eps = 0.5):
    if random.random() < eps:  # True with prob eps, random action
        actions = q.actions
        indx = int(random.random() * len(actions))
        return q.actions[indx]
    else:
        return greedy(q, s)

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])
    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, a, v):
        self.q[(s,a)] = v
    def get(self, s, a):
        return self.q[(s,a)]
    def update(self, data, lr):
        # Your code here
        for (s, a, t) in data:
            self.q[(s, a)] = self.q[(s, a)] * (1 - lr) + lr * t

# def Q_learn(mdp, q, lr=.1, iters=100, eps = 0.5, interactive_fn=None):
#     s = mdp.init_state()
#     for i in range(iters):
#         # include this line in the iteration, where i is the iteration number
#         if interactive_fn: interactive_fn(q, i)
#         a = epsilon_greedy(q, s, eps)
#         r, s_prime = mdp.sim_transition(s, a)
#         max_Q = value(q, s_prime)
#         if mdp.terminal(s):
#             max_Q = 0
#         gamma = mdp.discount_factor
#         t = r + max_Q * gamma
#         # data.append((s, a, t))
#         q.update([s, a, t], lr)
#         s = s_prime
#     return q

def Q_learn(mdp, q, lr=.1, iters=100, eps=0.5, interactive_fn=None):
    s = mdp.init_state()
    for i in range(iters):
        if interactive_fn: interactive_fn(q, i)
        a = epsilon_greedy(q, s, eps)
        r, s_prime = mdp.sim_transition(s, a)
        future_val = 0 if mdp.terminal(s) else value(q, s_prime)
        q.update([(s, a, (r + mdp.discount_factor * future_val))], lr)
        s = s_prime
    return q

# Simulate an episode (sequence of transitions) of at most
# episode_length, using policy function to select actions.  If we find
# a terminal state, end the episode.  Return accumulated reward a list
# of (s, a, r, s') whee s' is None for transition from terminal state.
def sim_episode(mdp, episode_length, policy, draw=False):
    episode = []
    reward = 0
    s = mdp.init_state()
    for i in range(episode_length):
        a = policy(s)
        (r, s_prime) = mdp.sim_transition(s, a)
        reward += r
        if mdp.terminal(s):
            episode.append((s, a, r, None))
            return reward, episode
        episode.append((s, a, r, s_prime))
        if draw: mdp.draw_state(s)
        s = s_prime
    return reward, episode 

# Return average reward for n_episodes of length episode_length
# while following policy (a function of state) to choose actions.
def evaluate(mdp, n_episodes, episode_length, policy):
    score = 0
    length = 0
    for i in range(n_episodes):
        # Accumulate the episode rewards
        r, e = sim_episode(mdp, episode_length, policy)
        score += r
        length += len(e)
        # print('    ', r, len(e))
    return score/n_episodes, length/n_episodes

def Q_learn_batch(mdp, q, lr=.1, iters=100, eps=0.5,
                  episode_length=10, n_episodes=2,
                  interactive_fn=None):
    # Your code here
    experiences = []
    # s = mdp.init_state
    policy = lambda s: epsilon_greedy(q, s, eps)
    for i in range(iters):
        # include this line in the iteration, where i is the iteration number
        if interactive_fn: interactive_fn(q, i)
        for n in range(n_episodes):
            r, episodes = sim_episode(mdp, episode_length, policy)
            experiences += episodes
        q_targets = []
        for (s, a, r, s_prime) in experiences:
            Q_max = 0 if s_prime is None else value(q, s_prime)
            t = r + mdp.discount_factor * Q_max
            q_targets.append((s, a, t))
        q.update(q_targets, lr)
    return q

def make_nn(state_dim, num_hidden_layers, num_units):
    model = Sequential()
    model.add(Dense(num_units, input_dim = state_dim, activation='relu'))
    for i in range(num_hidden_layers-1):
        model.add(Dense(num_units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model

class NNQ:
    def __init__(self, states, actions, state2vec, num_layers, num_units, epochs=1):
        self.actions = actions
        self.states = states
        self.state2vec = state2vec
        self.epochs = epochs
        state_dim = state2vec(states[0]).shape[1]
        self.set_models(state_dim, num_layers, num_units)

    def set_models(self, state_dim, num_layers, num_units):
        self.models = {}
        for elem in self.actions:
            self.models[elem] = make_nn(state_dim, num_layers, num_units)

    def get(self, s, a):
        model = self.models[a]
        return model.predict(self.state2vec(s))

    def update(self, data, lr, epochs=1):
        # print(data)
        # feed_data = {}  # maping action to x,y tuples
        # for a in self.actions:
        #     feed_data[a] = {'X': [], 'Y': []}
        # for (s, a, t) in data:
        #     feed_data[a]['X'].append(self.state2vec(s))
        #     feed_data[a]['Y'].append([t])
        # print(feed_data)
        # for a in self.actions:
        #     if feed_data[a]['X'] == []:
        #         continue
        #     X = np.vstack(feed_data[a]['X'])
        #     Y = np.array(feed_data[a]['Y'])
        #     self.models[a].fit(X, Y, epochs=self.epochs)
        for a in self.actions:
            if [s for (s, at, t) in data if a == at]:
                X = np.vstack([self.state2vec(s) for (s, at, t) in data if a == at])
                Y = np.vstack([np.array([float(t)]) for (s, at, t) in data if a == at])
                self.models[a].fit(X, Y, epochs=self.epochs, verbose=False)