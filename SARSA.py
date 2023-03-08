import numpy as np
import pandas as pd
from Parameters import *
import matplotlib.pyplot as plt
from Environment import Frozenlake
import time


class Sarsa(object):
    def __init__(self, env, learning_rate, gamma, epsilon):
        """create an empty Q-table and initialize the algorithm"""
        # define the environment
        self.env = env
        # all actions
        self.actions = list(range(self.env.n_actions))
        # set learning rate
        self.lr = learning_rate
        # set gamma (reward decay)
        self.gamma = gamma
        # set epsilon
        self.epsilon = epsilon
        # create an empty q_table
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def add_new_state_to_q_table(self, state):
        """check if the state already in Q table;
        if not,append it to Q table and set Q value of this new state are all 0,
        if already in, pass"""
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, state):
        """use epsilon greedy to explore states
         ensure that all actions are selected infinitely often"""
        self.add_new_state_to_q_table(state)
        # with a probability of epsilon to randomly choose action
        if np.random.uniform() > self.epsilon:
            # choose random action
            action = np.random.choice(self.actions)
        else:
            # chose a set of action with max q_value
            state_action = self.q_table.loc[state, :]
            max_a = state_action[state_action == np.max(state_action)].index
            action = np.random.choice(max_a)
        return action

    def update_q_table(self, state, action, reward, state_, action_):
        # add new state to q_table
        self.add_new_state_to_q_table(state_)

        # get the q_value of the current position
        q_predict = self.q_table.loc[state, action]

        # Sarsa: calculate target q_value: Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
        # q_target = reward + self.gamma * self.q_table.loc[state_, :].max()
        q_target = reward + self.gamma * self.q_table.loc[state_, action_]

        # Update q_table
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

        # return self.q_table.loc[state, action]


def train(policy):
    """main training function for updating q_table"""
    # recode performance for comparison
    reward_list = []
    record_route_length = []
    record_goal = []
    record_fail = []
    count_goal = 0
    count_fail = 0
    episode_reward = 0

    for episode in range(NUM_EPISODES):
        done = False
        route_length = 0
        # initialize the environment and get a state at the beginning at each episode
        state = env.reset()
        while not done:
            route_length += 1

            # RL choose action based on epsilon greedy policy
            action = policy.choose_action(state)

            # take the action in the env and get the next_state and reward
            state_, reward, done = env.step(action)

            # record the episode_reward
            episode_reward += reward

            # choose next_action based on next state
            action_ = policy.choose_action(state_)

            # update q_table
            policy.update_q_table(state, action, reward, state_, action_)

            # swap state
            state = state_

            if done:
                if reward == 1:
                    count_goal += 1
                    record_route_length.append(route_length)
                    print("Route length of the robot when reaching the frisbee: {}".format(route_length))
                if reward == -1:
                    count_fail += 1

        # record the num of reaching the frisbee and falling into the hole
        performance_bar = [count_goal, count_fail]

        # recode the performance after each episode ends
        record_goal.append(count_goal)
        record_fail.append(count_fail)
        reward_list.append((episode_reward / (episode + 1)))
        if episode != 0 and episode % 100 == 0:
            print('episode', episode)

    plot_results(record_goal, record_fail, record_route_length, reward_list, performance_bar)

    # calculate the success rate of the robot
    P_reach = (count_goal / NUM_EPISODES) * 100
    P_fail = (count_fail / NUM_EPISODES) * 100
    print("\n<---------------------Success rate of the robot--------------------->")
    print('Probability of reaching the frisbee is : {}'.format(P_reach))
    print('Probability of falling into the hole is : {}'.format(P_fail))
    return record_goal, record_fail, record_route_length, reward_list, performance_bar, policy


def test(policy):
    """test the policy if the robot has learned to reach the frisbee """
    print("\n<---------------------Testing the policy ! :)--------------------->")
    done = False
    state = env.reset()
    # env.render()
    time.sleep(0.5)
    while not done:
        action = policy.choose_action(state)
        next_state, reward, done = env.step(action)
        env.render()
        time.sleep(0.5)
        state = next_state

    # show the final route
    env.final()


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

    # plot the num of reaching the frisbee and falling into the hole
    plt.figure()
    performance_list = ['Reaching', 'Falling']
    color_list = ['blue', 'red']
    plt.bar(np.arange(len(performance_bar)), performance_bar, tick_label=performance_list, color=color_list)
    plt.title('Bar/Reaching and Falling')
    plt.ylabel('Numbers')

    # show the figures
    plt.show()


if __name__ == "__main__":
    # Create an environment
    env = Frozenlake(map_size=GRID_SIZE)

    # create a Sarsa agent
    policy = Sarsa(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

    # start training
    record_goal, record_fail, record_route_length, reward_list, performance_bar, policy = train(policy)

    print("Q_table\n", policy.q_table)

    # test the trained policy
    test(policy)

    # remain visualized
    env.mainloop()

    # polt the results and evaluate robot's performance
    # plot_results()
