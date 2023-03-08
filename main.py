import time

from matplotlib import pyplot as plt

from Environment import Frozenlake
from Parameters import *
import argparse
from Q_learning import Q_learning
from Monte_carlo import Monte_carlo
from SARSA import Sarsa


def train_Q_learning(policy):
    """main training function for updating q_table"""
    # recode performance for comparison
    global reward
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
        route_length = 0
        # initialize the environment and get a state at the beginning at each episode
        state = env.reset()
        while not done:
            # render the environment
            # env.render()
            route_length += 1

            # RL choose action based on epsilon greedy policy
            action = policy.choose_action(state)

            # take the action in the env and get the next_state and reward
            state_, reward, done = env.step(action)

            # update q_table
            policy.update_q_table(state, action, reward, state_)

            # swap state
            state = state_

            if done:
                if reward == 1:
                    count_goal += 1
                    record_route_length.append(route_length)
                    print("Route length of the robot when reaching the frisbee: {}".format(route_length))
                if reward == -1:
                    count_fail += 1
        # record the episode_reward
        episode_reward += reward
        # reward_list += [episode_reward / (episode + 1)]

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
    # print("Q_table", Q_table)
    return record_goal, record_fail, record_route_length, reward_list, performance_bar, policy

def train_Sarsa(policy):
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
    # for j in range(NUM_STEPS):
    #     action = policy.choose_action(state)
    #     next_state, reward, done = env.step(action)
    #     env.render()
    #     time.sleep(0.5)
    #     y = int(math.floor(next_state / GRID_SIZE)) * PIXELS
    #     x = int(next_state % GRID_SIZE) * PIXELS
    #     env.route_store_dir[j] = [x, y]
    #     state = next_state

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

    plt.figure()
    performance_list = ['Reaching', 'Falling']
    color_list = ['blue', 'red']
    plt.bar(range(len(performance_bar)), performance_bar, tick_label=performance_list, color=color_list)
    plt.title('Bar/Reaching and Falling')
    plt.ylabel('Numbers')

    # plt.figure()
    # plt.plot(np.arange(len(reward_list)), reward_list, 'b')
    # plt.title('Episode via Average rewards')
    # plt.xlabel('Episode')
    # plt.ylabel('Average rewards')


    # show the figures
    plt.show()


if __name__ == "__main__":
    # set the training arguments here
    parser = argparse.ArgumentParser(description="Set the arguments to specify the training")
    parser.add_argument("--algorithm", default="Q_learning")  # training algorithm, can be set to "Q_learning", "SARSA" or "Monte_carlo"
    parser.add_argument("--grid_size", default=4)  # grid_size of the map, can be set to 4 or 10
    parser.add_argument("--learning_rate", default=0.01)  # learning_rate
    parser.add_argument("--gamma", default=0.9)  # gamma (reward decay)
    parser.add_argument("--epsilon", default=0.9)  # epsilon greedy
    args = parser.parse_args()

    if args.algorithm == "Q_learning":
        print("<-----Start to train {0} algorithm in {1} grid_size map!----->".format(args.algorithm, args.grid_size))
        time.sleep(3)
        # Create an environment
        env = Frozenlake(map_size=args.grid_size)

        # Create a  Q_learning agent
        policy = Q_learning(env, learning_rate=args.learning_rate, gamma=args.gamma, epsilon=args.epsilon)

        # start training
        record_goal, record_fail, record_route_length, reward_list, performance_bar, policy = train_Q_learning(policy)

        # write Q_table
        # write_Q_table(file_name="./Q_table/Q_learning", Q=policy.q_table)

        print("Q_table\n", policy.q_table)

        # test the trained policy
        test(policy)

        # remain visualized
        env.mainloop()

        # # polt the results and evaluate robot's performance
        # plot_results()

    if args.algorithm == "SARSA":
        print("<-----Start to train {0} algorithm in {1} grid_size map!----->".format(args.algorithm, args.grid_size))
        time.sleep(3)
        # Create an environment
        env = Frozenlake(map_size=args.grid_size)

        # create a Sarsa agent
        policy = Sarsa(env, learning_rate=args.learning_rate, gamma=args.gamma, epsilon=args.epsilon)

        # start training
        record_goal, record_fail, record_route_length, reward_list, performance_bar, policy = train_Sarsa(policy)

        print("Q_table\n", policy.q_table)

        # test the trained policy
        test(policy)

        # remain visualized
        env.mainloop()

        # polt the results and evaluate robot's performance
        # plot_results()

    if args.algorithm == "Monte_carlo":
        print("<-----Start to train {0} algorithm in {1} grid_size map!----->".format(args.algorithm, args.grid_size))
        time.sleep(3)
        # Create an environment
        env = Frozenlake(map_size=args.grid_size)

        # Create a  Monte_carlo agent
        monte_carlo = Monte_carlo(env, gamma=args.gamma, epsilon_standered=args.epsilon, epsilon_decay=EPSILON_START, e_decay=True)

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






