# ME 5406 Deep Learning for Robotics Project1

# Problem Statement
Consider a frozen lake with (four) holes covered by patches of very thin ice. Suppose that a robot is to glide on the frozen surface from one location(i.e., the top left corner) to another (bottom right corner) in order to pick up a frisbee, as illustrated below.

![](render_images/frozen%20lake%20environment.png)
 
# 3 Model Free RL Algorithms:
1. Q learning
2. SARSA
3. First-visit Monte Carlo control

# Requirements
Use `conda install --file requirements.txt` to install the following requirements:
- matplotlib 
- pillow
- pandas
- tk

(you can refer to requirements.txt)

# How to Run
1. You can simply set the arguments to specify the training process and run the code in main.py. The default arguments are: args.algorithm="Q_learning"; 
args.grid_size=4; args.learning_rate=0.01; arg.gamma=0.9; args.epsilon=0.9. You can also simply run the training process of the 3 algorithms in Q_learning.py, SARSA.py and Monte_carlo.py, respectively, the training parameters can be set in Parameters.py.
2. You can also use the terminal to run this code:
```shell
conda create -n WZQ python==3.6
conda activate WZQ  
conda install --file requirements.txt
python main.py --algorithm "Q_learning" --map_size 4  
```
# Experiment Results (4 × 4 grid map)
- Q learning

![](results/Q_learning/4×4%20grid%20map/bar%20reaching%20and%20falling.png)
![](results/Q_learning/4×4%20grid%20map/all%20evaluatioins.png)

- SARSA

![](results/SARSA/4×4%20grid%20map/bar%20reahcing%20and%20falling.png)
![](results/SARSA/4×4%20grid%20map/all%20evaluations.png)

- Monte Carlo First-visit

![](results/Monte_carlo/4×4%20grid%20map/bar%20reaching%20and%20falling.png)
![](results/Monte_carlo/4×4%20grid%20map/all%20evaluations.png)


- Final Optimal Policy
  * Q learning, SARSA and Monte Carlo, respectively
    
    ![](results/Q_learning/4×4%20grid%20map/final%20route.png)
    ![](results/SARSA/4×4%20grid%20map/final%20route.png)
    ![](results/Monte_carlo/4×4%20grid%20map/final%20route.png)


# Acknowledgement
If my code helps you learn reinforcement learning, please give me a star :)
