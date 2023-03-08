import random
import numpy as np
import tkinter as tk  # build the map
import time

from PIL import Image, ImageTk
from Parameters import *  # set the parameters


class Frozenlake(tk.Tk, object):
    """create a class for the Frozenlake environment"""
    def __init__(self, map_size):
        super(Frozenlake, self).__init__()
        # define map_size
        self.map_size = map_size
        # actions for the agent
        self.action = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.n_actions = len(self.action)
        self.n_states = int(self.map_size) * int(self.map_size)

        self.title('Frozen lake')
        self.geometry('{0}x{1}'.format(map_size * PIXELS, map_size * PIXELS))

        # create the dir to store the final route
        self.route_dir = {}
        self.route_store_dir = {}

        # global variable for dictionary with coordinates for the final route
        self.a = {}

        # define the key of the route_dir
        self.i = 0

        # whether to store the route for the first time the robot get to the frisbee
        self.store_route = True

        # showing the record_route_length for longest found route
        self.longest = 0

        # showing the record_route_length for the shortest route
        self.shortest = 0

        # build the frozen lake map
        self.build_environment()

    def build_environment(self):
        """build the frozen_lake map"""
        # build 4*4 frozen lake map
        if self.map_size == 4:
            self.build_environment_map_size_4()
            print("crate Frozen Lake map(4*4)")

        # build 10*10 frozen lake map
        elif self.map_size == 10:
            self.build_environment_map_size_10()
            print("crate Frozen Lake map(10*10)")
        else:
            print("Please input the correct map_size(4 or 10)")

    def build_environment_map_size_4(self):
        """function to build 4*4 grid map"""
        # create canvas
        self.canvas = tk.Canvas(self, bg='white',
                                       height= 4 * PIXELS,
                                       width= 4 * PIXELS)
        # create grids
        # draw vertical lines
        for column in range(0, 4 * PIXELS, PIXELS):
            x0, y0, x1, y1 = column, 0, column, 4 * PIXELS
            self.canvas.create_line(x0, y0, x1, y1, fill='black')
        # draw horizontal lines
        for row in range(0, 4 * PIXELS, PIXELS):
            x0, y0, x1, y1 = 0, row, 4 * PIXELS, row
            self.canvas.create_line(x0, y0, x1, y1, fill='black')

        # create the render images
        # robot(agent)
        img_robot = Image.open("render_images/robot.png")
        self.robot = ImageTk.PhotoImage(img_robot)
        # hole
        img_hole = Image.open("render_images/hole.png")
        self.hole = ImageTk.PhotoImage(img_hole)
        # frisbee
        img_frisbee = Image.open("render_images/frisbee.png")
        self.frisbee_object = ImageTk.PhotoImage(img_frisbee)

        # create holes in the map
        self.hole_position1 = self.canvas.create_image(PIXELS * 0, PIXELS * 3, anchor='nw',
                                                         image=self.hole)

        self.hole_position2 = self.canvas.create_image(PIXELS * 1, PIXELS * 1, anchor='nw',
                                                         image=self.hole)

        self.hole_position3 = self.canvas.create_image(PIXELS * 3, PIXELS * 1, anchor='nw',
                                                         image=self.hole)

        self.hole_position4 = self.canvas.create_image(PIXELS * 3, PIXELS * 2, anchor='nw',
                                                         image=self.hole)

        # create Frisbee in the map
        self.frisbee = self.canvas.create_image(PIXELS * 3, PIXELS * 3, anchor='nw', image=self.frisbee_object)

        # create robot in the map
        self.agent = self.canvas.create_image(0, 0, anchor='nw', image=self.robot)

        # pack the information
        self.canvas.pack()

        # store the positions of the holes
        self.hole_positions = [self.canvas.coords(self.hole_position1),
                                    self.canvas.coords(self.hole_position2),
                                    self.canvas.coords(self.hole_position3),
                                    self.canvas.coords(self.hole_position4)]

        # store the position of the terminal state (Frisbee)
        self.frisbee_position = self.canvas.coords(self.frisbee)

    def build_environment_map_size_10(self):
        """function to build 10*10 grid map"""
        # create canvas
        self.canvas = tk.Canvas(self, bg='white',
                                       height=10 * PIXELS,
                                       width=10 * PIXELS)
        # create grids
        # draw vertical lines
        for column in range(0, 10 * PIXELS, PIXELS):
            x0, y0, x1, y1 = column, 0, column, 10 * PIXELS
            self.canvas.create_line(x0, y0, x1, y1, fill='black')
        # draw horizontal lines
        for row in range(0, 10 * PIXELS, PIXELS):
            x0, y0, x1, y1 = 0, row, 10 * PIXELS, row
            self.canvas.create_line(x0, y0, x1, y1, fill='black')

        # create the render images
        # robot(agent)
        img_robot = Image.open("render_images/robot.png")
        self.robot = ImageTk.PhotoImage(img_robot)
        # hole
        img_hole = Image.open("render_images/hole.png")
        self.hole = ImageTk.PhotoImage(img_hole)
        # frisbee
        img_frisbee = Image.open("render_images/frisbee.png")
        self.frisbee_object = ImageTk.PhotoImage(img_frisbee)

        # create holes in the map
        self.hole_position1 = self.canvas.create_image(PIXELS * 0, PIXELS * 2, anchor='nw',
                                                         image=self.hole)
        self.hole_position2 = self.canvas.create_image(PIXELS * 0, PIXELS * 1, anchor='nw',
                                                         image=self.hole)
        self.hole_position3 = self.canvas.create_image(PIXELS * 1, PIXELS * 4, anchor='nw',
                                                         image=self.hole)
        self.hole_position4 = self.canvas.create_image(PIXELS * 2, PIXELS * 6, anchor='nw',
                                                         image=self.hole)
        self.hole_position5 = self.canvas.create_image(PIXELS * 2, PIXELS * 9, anchor='nw',
                                                         image=self.hole)
        self.hole_position6 = self.canvas.create_image(PIXELS * 3, PIXELS * 0, anchor='nw',
                                                         image=self.hole)
        self.hole_position7 = self.canvas.create_image(PIXELS * 3, PIXELS * 2, anchor='nw',
                                                         image=self.hole)
        self.hole_position8 = self.canvas.create_image(PIXELS * 3, PIXELS * 5, anchor='nw',
                                                         image=self.hole)
        self.hole_position9 = self.canvas.create_image(PIXELS * 2, PIXELS * 7, anchor='nw',
                                                         image=self.hole)
        self.hole_position10 = self.canvas.create_image(PIXELS * 4, PIXELS * 0, anchor='nw',
                                                          image=self.hole)
        self.hole_position11 = self.canvas.create_image(PIXELS * 4, PIXELS * 3, anchor='nw',
                                                          image=self.hole)
        self.hole_position12 = self.canvas.create_image(PIXELS * 5, PIXELS * 2, anchor='nw',
                                                          image=self.hole)
        self.hole_position13 = self.canvas.create_image(PIXELS * 5, PIXELS * 4, anchor='nw',
                                                          image=self.hole)
        self.hole_position14 = self.canvas.create_image(PIXELS * 5, PIXELS * 6, anchor='nw',
                                                          image=self.hole)
        self.hole_position15 = self.canvas.create_image(PIXELS * 5, PIXELS * 7, anchor='nw',
                                                          image=self.hole)
        self.hole_position16 = self.canvas.create_image(PIXELS * 6, PIXELS * 1, anchor='nw',
                                                          image=self.hole)
        self.hole_position17 = self.canvas.create_image(PIXELS * 6, PIXELS * 3, anchor='nw',
                                                          image=self.hole)
        self.hole_position18 = self.canvas.create_image(PIXELS * 6, PIXELS * 5, anchor='nw',
                                                          image=self.hole)
        self.hole_position19 = self.canvas.create_image(PIXELS * 6, PIXELS * 8, anchor='nw',
                                                          image=self.hole)
        self.hole_position20 = self.canvas.create_image(PIXELS * 7, PIXELS * 0, anchor='nw',
                                                          image=self.hole)
        self.hole_position21 = self.canvas.create_image(PIXELS * 7, PIXELS * 3, anchor='nw',
                                                          image=self.hole)
        self.hole_position22 = self.canvas.create_image(PIXELS * 0, PIXELS * 7, anchor='nw',
                                                          image=self.hole)
        self.hole_position23 = self.canvas.create_image(PIXELS * 8, PIXELS * 1, anchor='nw',
                                                          image=self.hole)
        self.hole_position24 = self.canvas.create_image(PIXELS * 8, PIXELS * 6, anchor='nw',
                                                          image=self.hole)
        self.hole_position25 = self.canvas.create_image(PIXELS * 9, PIXELS * 5, anchor='nw',
                                                          image=self.hole)

        # create Frisbee in the map
        self.frisbee = self.canvas.create_image(PIXELS * 9, PIXELS * 9, anchor='nw', image=self.frisbee_object)

        # create robot in the map
        self.agent = self.canvas.create_image(0, 0, anchor='nw', image=self.robot)

        # pack the information
        self.canvas.pack()

        # store the positions of the holes
        self.hole_positions = [self.canvas.coords(self.hole_position1),
                                    self.canvas.coords(self.hole_position2),
                                    self.canvas.coords(self.hole_position3),
                                    self.canvas.coords(self.hole_position4),
                                    self.canvas.coords(self.hole_position5),
                                    self.canvas.coords(self.hole_position6),
                                    self.canvas.coords(self.hole_position7),
                                    self.canvas.coords(self.hole_position8),
                                    self.canvas.coords(self.hole_position9),
                                    self.canvas.coords(self.hole_position10),
                                    self.canvas.coords(self.hole_position11),
                                    self.canvas.coords(self.hole_position12),
                                    self.canvas.coords(self.hole_position13),
                                    self.canvas.coords(self.hole_position14),
                                    self.canvas.coords(self.hole_position15),
                                    self.canvas.coords(self.hole_position16),
                                    self.canvas.coords(self.hole_position17),
                                    self.canvas.coords(self.hole_position18),
                                    self.canvas.coords(self.hole_position19),
                                    self.canvas.coords(self.hole_position20),
                                    self.canvas.coords(self.hole_position21),
                                    self.canvas.coords(self.hole_position22),
                                    self.canvas.coords(self.hole_position23),
                                    self.canvas.coords(self.hole_position24),
                                    self.canvas.coords(self.hole_position25)]

        # store the position of the terminal state (frisbee)
        self.frisbee_position = self.canvas.coords(self.frisbee)

    def reset(self):
        """After each episode ends, the robot go back to the original point and starts again"""
        self.update()
        # Updating agent
        self.canvas.delete(self.agent)
        # again, create the new robot at the original point
        self.agent = self.canvas.create_image(0, 0, anchor='nw', image=self.robot)

        # clear the route_dir and the key
        self.route_dir = {}
        self.i = 0

        # get the state(coordination) of the agent(robot)
        s = self.canvas.coords(self.agent)
        # print("state before indexed", s)

        # calculate the state of agent in index form
        s = self.position_transition(s[0], s[1])
        # print('state', s)

        return s

    def step(self, action):
        """The agent(robot) interacts with the environment and get the next state, reward and other information"""
        # get the current state of the agent
        state = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])

        # define actions
        # 'UP'
        if action == 0:
            if state[1] >= PIXELS:
                base_action[1] -= PIXELS
        # 'DOWN'
        elif action == 1:
            if state[1] < (self.map_size - 1) * PIXELS:
                base_action[1] += PIXELS
        # 'RIGHT'
        elif action == 2:
            if state[0] < (self.map_size - 1) * PIXELS:
                base_action[0] += PIXELS
        # 'LEFT'
        elif action == 3:
            if state[0] >= PIXELS:
                base_action[0] -= PIXELS

        # move the agent according to the action
        self.canvas.move(self.agent, base_action[0], base_action[1])

        # update the next state
        next_state = self.canvas.coords(self.agent)

        # store the coordinate of the robot (next_state) to create the found route
        self.route_dir[self.i] = next_state

        # Updating key of the route_dir
        self.i += 1

        # calculate the reward and get the done info
        if next_state == self.frisbee_position:
            reward = 1
            done = True

            # store the route in route_dir when the robot get to the friebee for the first time and define the route length
            if self.store_route == True:
                for j in range(len(self.route_dir)):
                    self.route_store_dir[j] = self.route_dir[j]
                self.store_route = False
                self.longest = len(self.route_dir)
                self.shortest = len(self.route_dir)

            # compare the currently found route and the stored route, if shorter, then restore the shorter route
            if len(self.route_dir) < len(self.route_store_dir):
                # get the shortest route length
                self.shortest = len(self.route_dir)
                # clear the route_store_dir
                self.route_store_dir = {}
                # assign the route_store_dir to the shorter route
                for j in range(len(self.route_dir)):
                    self.route_store_dir[j] = self.route_dir[j]

            # compare the currently found route and the stored route, if longer, then get the longest route length
            if len(self.route_dir) > self.longest:
                self.longest = len(self.route_dir)

        elif next_state in self.hole_positions:
            reward = -1
            done = True

            # clear the route_dir and the key
            self.route_dir = {}
            self.i = 0

        else:
            reward = 0
            done = False

        # calculate the state of agent in index form
        next_state = self.position_transition(next_state[0], next_state[1])

        return next_state, reward, done

    def render(self):
        time.sleep(0.05)
        self.update() 

    def final(self):
        """function to show the final route"""
        # delete the robot position when getting the frisbee
        self.canvas.delete(self.agent)

        self.canvas.create_image(0, 0, anchor='nw', image=self.robot)

        # show the longest and shortest route length
        print('The shortest route:', self.shortest)
        print('The longest route:', self.longest)
        print("The shortest route is shown in the red spots")

        # create initial point
        origin = np.array([20, 20])
        self.initial_point = self.canvas.create_oval(
            origin[0] - 5, origin[1] - 5,
            origin[0] + 5, origin[1] + 5,
            fill='red', outline='red')

        # define the final route
        for j in range(len(self.route_store_dir)):
            # get all the coordinates of the final route
            self.track = self.canvas.create_oval(
                self.route_store_dir[j][0] + origin[0] - 5, self.route_store_dir[j][1] + origin[0] - 5,
                self.route_store_dir[j][0] + origin[0] + 5, self.route_store_dir[j][1] + origin[0] + 5,
                fill='red', outline='red')
            # Writing the final route in the global variable a
            self.a[j] = self.route_store_dir[j]

    # Returning the final dictionary with route coordinates
    # Then it will be used in agent_brain.py
    def final_states(self):
        return self.a

    def position_transition(self, x, y):
        """function to calculate the state of agent (coordinate) in index form"""
        # Coordinate transformation: Coordinate-> Indexed number
        state_index = int(x / 40) + int(y / 40 * self.map_size)
        return state_index


# test the environment map
if __name__ == '__main__':
    # build the grid map
    env = Frozenlake(map_size=GRID_SIZE)
    env.reset()
    # remain the environment visualized
    env.mainloop()
