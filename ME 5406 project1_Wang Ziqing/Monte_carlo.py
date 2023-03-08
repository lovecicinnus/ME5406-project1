import time

import numpy as np
from itertools import product

from matplotlib import pyplot as plt

from Environment import Frozenlake
from Parameters import *


class Monte_carlo(object):
    def __init__(self, env, gamma, epsilon_standered=EPSILON, epsilon_decay=EPSILON_START, e_decay=False):
        """
        initialize the algorithm
        Monte_carlo is a model free reinforcement learning method
        """
        self.env = env
        self.gamma = gamma
        self.e_decay = e_decay
        if self.e_decay == True:
            self.epsilon = epsilon_decay
        else:
            self.epsilon = epsilon_standered

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states

        # create a random policy
        self.policy = {j: np.random.choice(self.n_actions) for j in range(self.n_states)}

        # create a q_table
        self.Q = {j: {i: 0 for i in range(self.n_actions)} for j in range(self.n_states)}

        # initialize return as an empty list
        self.returns = {(state, action): [] for state, action in product(range(self.n_states), range(self.n_actions))}

        # recode performance for comparison
        self.reward_list = []
        self.record_route_length = []
        self.record_goal = []
        self.record_fail = []
        self.count_goal = 0
        self.count_fail = 0

    def choose_action(self, action, epsilon):
        """
        epsilon greedy action selection,with a probability of (1-epsilon) to choose best action
        with a probability of epsilon to random choose action
        """
        self.epsilon = epsilon
        # with a probability of epsilon to randomly choose action
        if self.e_decay == False:
            if np.random.uniform() > self.epsilon:
                # choose random action
                return np.random.choice(self.n_actions)
            else:
                return action
        else:
             if np.random.uniform() < self.epsilon:
                 return np.random.choice(self.n_actions)
             else:
                 return action

    def linear_anneal(self, start, anneal_step=END_DECAY, end=EPSILON_END):
        """from start decay to end with current_episode increase ,unchange after record_route_length"""
        epsilon = start
        epsilon = start - ((start - end) / anneal_step) if epsilon > end else epsilon
        return epsilon

    def calculate_g(self, data_list):
        """
        use G=reward+gamma*G(next state) to calculate G
        Args: data_list: list of state, action, reward of an episode.
        Gamma: reward decay
        Return: list of all states with state, action, G for each state visited.
        """
        G = 0
        G_list = []
        for state, action, reward in reversed(data_list):
            # reverse list in order to calculate G from end to start use g(t) = r(t+1) + gamma*G(t+1)
            G = reward + GAMMA * G
            G_list.append([state, action, G])
        return reversed(G_list)

    def train(self):
        """training process, update Q and policy after every episode ends and find the optimal policy"""
        self.env.render()
        reward_list = []
        record_route_length = []
        record_goal = []
        record_fail = []
        performance_bar = []
        count_goal = 0
        count_fail = 0
        episode_reward = 0
        for episode in range(NUM_EPISODES):
            done = False
            episode_data = []
            route_length = 0
            state = self.env.reset()
            while not done:
                # collect data until episode end
                route_length += 1
                action = self.choose_action(self.policy[state], self.epsilon)

                # use this action to get state_, reward and done info
                state_, reward, done = self.env.step(action)
                episode_reward += reward

                # store the episode_data
                episode_data.append([state, action, reward])

                # swap state
                state = state_

                if done:
                    if reward == 1:
                        # record the num of times that the robot reaches the frisbee
                        count_goal += 1

                        if self.e_decay == True:
                            self.epsilon = self.linear_anneal(start=self.epsilon, anneal_step=END_DECAY, end=EPSILON_END)

                        # record the route_length when the robot reaches the frisbee
                        record_route_length.append(route_length)
                        print("Route length of the robot when reaching the frisbee: {}".format(route_length))

                    if reward == -1:
                        # record the num of times that the robot falls into the hole
                        count_fail += 1

            # record the num of reaching the frisbee and falling into the hole
            performance_bar = [count_goal, count_fail]

            # record the performance
            record_goal.append(count_goal)
            record_fail.append(count_fail)
            reward_list.append((episode_reward / (episode + 1)))

            # each episode set seen_state_action_pairs to empty
            state_action_pair = set()

            # collect the data after run one episode and calculate G
            G_list = self.calculate_g(episode_data)

            # use state, action, G to update Q
            for state, action, G in G_list:
                pair = (state, action)
                if pair not in state_action_pair:
                    # if pair not in set mean it is first visit,just the first pair will be record
                    self.returns[pair].append(G)
                    self.Q[state][action] = np.mean(self.returns[pair])
                    state_action_pair.add(pair)

            for state in self.policy.keys():
                # update the policy with the max Q_value
                self.policy[state] = max(self.Q[state], key=self.Q[state].get)

            if episode % 100 == 0:
                print("episode", episode)

            if self.e_decay:
                if episode % 10000 == 0 and episode != 0:
                    print("current epsilon", self.epsilon)

        plot_results(record_goal, record_fail, record_route_length, reward_list, performance_bar)
        # calculate the success rate of the robot
        P_reach = (count_goal / NUM_EPISODES) * 100
        P_fail = (count_fail / NUM_EPISODES) * 100
        print("\n<---------------------Success rate of the robot--------------------->")
        print('Probability of reaching the frisbee is : {}'.format(P_reach))
        print('Probability of falling into the hole is : {}'.format(P_fail))

        return record_goal, record_fail, record_route_length, reward_list, \
               performance_bar, self.policy, self.Q

    def test(self):
        """test the policy if the robot has learned to reach the frisbee """
        print("\n<---------------------Testing the policy ! :)--------------------->")
        done = False
        state = self.env.reset()
        # env.render()
        time.sleep(0.5)
        while not done:
            next_state, reward, done = self.env.step(self.policy[state])
            self.env.render()
            time.sleep(0.5)
            state = next_state

        # show the final route
        self.env.final()


def plot_results(record_goal, record_fail, record_route_length, reward_list, performance_bar):
    """function to plot the performance of the robot in the figures"""
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.5, hspace=0.7)
    f1 = fig.add_subplot(2, 2, 1)
    f2 = fig.add_subplot(2, 2, 2)
    f3 = fig.add_subplot(2, 2, 3)
    f4 = fig.add_subplot(2, 2, 4)

    # plot the reaching times
    f1.plot(range(len(record_goal)), record_goal, color='red')
    f1.set_title("Reaching times")
    f1.set_xlabel("Number of trained episodes")
    f1.set_ylabel("Times of reaching")

    # plot the failing times
    f2.plot(range(len(record_fail)), record_fail, color='orange')
    f2.set_title("Falling times")
    f2.set_xlabel("Number of trained episodes")
    f2.set_ylabel("Times of falling")

    # plot the route length
    f3.plot(range(len(record_route_length)), record_route_length, color='b')
    f3.set_title("Reaching route length")
    f3.set_xlabel("Number of trained episodes")
    f3.set_ylabel("Route length")

    # plot the episode_reward
    f4.plot(range(len(reward_list)), reward_list, color='yellow')
    f4.set_title("Episode reward")
    f4.set_xlabel("Number of trained episodes")
    f4.set_ylabel("Episode reward")

    plt.figure()
    performance_list = ['Reaching', 'Falling']
    color_list = ['blue', 'red']
    plt.bar(np.arange(len(performance_bar)), performance_bar, tick_label=performance_list, color=color_list)
    plt.title('Bar/Reaching and Falling')
    plt.ylabel('Numbers')

    # show the figures
    plt.show()


# Store the final Q table values
def write_Q_table(file_name, Q):
    # open data file
    filename = open(file_name, 'w')
    # write data
    for k, v in Q.items():
        filename.write(str(k) + ':' + str(v))
        filename.write('\n')
    # close file
    filename.close()


if __name__ == "__main__":

    # Create an environment
    env = Frozenlake(map_size=GRID_SIZE)

    # Create a  Monte_carlo agent
    monte_carlo = Monte_carlo(env, gamma=GAMMA, epsilon_standered=EPSILON, epsilon_decay=EPSILON_START,
                              e_decay=False)

    # start training
    record_goal, record_fail, record_route_length, reward_list, performance_bar, policy, Q = monte_carlo.train()

    # write Q_table
    # write_Q_table(file_name="./Q_table/monte_carlo", Q=Q)

    # test the trained policy
    monte_carlo.test()

    # remain visualized
    env.mainloop()
    
    # polt the results and evaluate robot's performance
    # monte_carlo.plot_results()



























