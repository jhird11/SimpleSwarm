#!/usr/bin/env python 

import copy
import time 
import sys
import os
import pickle


import numpy as np
import scipy.spatial
from SimulationRobot import SimulationRobot
from matplotlib import pyplot as plt
import matplotlib.animation as animation

"""
Basic Python swam simulator by Julian Hird - j.hird@bristol.ac.uk
Tested on:
    python                        2.7.17
    Ubuntu                        Ubuntu 18.04.4 LTS
    scipy                         1.2.2  
    numpy                         1.16.4 
    matplotlib                    2.2.4

See the bottom of this script for a demonstration of the simulator
"""

def gen_circle(centre_point,r,num_points = 100):
    """
    Returns an array of points along the perimeter of a circle located at centre with radius r

    Parameters
    ----------
    centre_point : tuple
        Centre of the circle
    r : float
        Radius of the circle
    num_points : int
        Number of samples taken along the circle's perimeter

    Returns 
    ------
    circle_array
        An numpy array of points
    """
    x_c,y_c = centre_point

    x = np.linspace(-r,r,num_points/2)   
    y = np.zeros((num_points))
    y[:int(num_points/2)] = r*np.sin(np.arccos(x/r))

    y[int(num_points/2):] = -r*np.sin(np.arccos(x/r))
    x = np.hstack((x,np.flip(x)))
    x+=np.ones(num_points)*x_c
    y+=np.ones(num_points)*y_c
    return np.array((x,y)).T


class SimulationWorld:

    def __init__(self):

        # Time between each simulation step
        self.dt = 0.01

        #Counts the number of simulation steps executed
        self.t_step = 0

        #List of all robots in the simulation world
        self.robot_list = []

        #Determins the outer bounds of the world area in the form of (lowest x, highest x, lowest y, highest y)
        self.barriers = np.array((-10.0,10.0,-10.0,10.0))

        #Size of bins used to divide up the space for collision detection
        self.collision_bin_size = 1.0

        #How many times should we execute binary search to resolve colisions, high numbers mean higher accuracy but will take longer
        self.max_col_resolve_depth = 3

        #Enable collision detection between robots
        self.robot_collisions = True

        #Enable calculation of neighbours of robots
        self.calculate_neighbours = True

        #How long the simulation will run for in time steps
        self.total_steps_num = 0

        #Log used to store simulation information (currently robot positions and rotations)
        self.data_log = None


    def check_barriers(self,robot_radius,robot_position):
        """
            Checks if a robot is in collision with the world's outer barriers

            Parameters
            ----------
            
            robot_size - float
                Radius of robot
            robot_position - np.array
                x,y position of robot

            Returns
            ------
                bool - True if robot is in collision with outer barriers

        """


        #calculates configuration space where robot can be without being in colision with barriers
        #can be precalculated if all robots are the same size

        self.c_space_top_y = self.barriers[3] - robot_radius
        self.c_space_bottom_y = self.barriers[2] + robot_radius
        self.c_space_right_x = self.barriers[1] - robot_radius 
        self.c_space_left_x = self.barriers[0] + robot_radius

        return (robot_position[0] <= self.c_space_left_x or robot_position[0] >= self.c_space_right_x or robot_position[1] >= self.c_space_top_y or robot_position[1] <= self.c_space_bottom_y) 
    

    def check_robot_collision(self,robot1_pos,robot2_pos,robot1_radius,robot2_radius):
        """
            Checks if robots 1 and 2 are in collision


            Parameters
            ----------
            robot1_pos : np.array
                x,y position of robot 1
            
            robot2_pos : np.array
                x,y position of robot 2

            robot1_radius : float
                Radius of robot 1

            robot2_radius : float
                Radius of robot 2

            Returns
            ------
                True if robots are in collision with each other

        """
        return np.sum(np.power((robot2_pos-robot1_pos),2.0)) < np.power(robot1_radius + robot2_radius,2.0)

    def solve_collison(self,robot1,collision_list,last_free_dt,last_collision_dt,depth = 0):

        """
        Determines the latest time between time steps a robot wasn't in collision

        Parameters
        ----------
        robot1 : SimulationRobot
            The robot we're solving collisions for
        collision_list : list
            The list of robots this robot could be in collision with
        last_free_dt : float
            The latest time we know the robot isn't in colision (starts at 0 ie. end of the previous timestep)
        last_collision_dt : float
            The earliest time we know the robot is in collision (starts at dt ie. end of the current time step)
        depth :float  
            The number of times we have called this function for a given robot, will terminate the binary search after max_col_resolve_depth iterations
    
        Returns
        ------
            float - The latest time the robot wasn't in collision relative to the start of the timestep
        """
        depth+=1
        #Terminate the search if we've reached max number of iterations of the search
        if depth >= self.max_col_resolve_depth:
            return last_free_dt

        #test dt is midway between the times we know the robot isn't in collision and the time we known is in collision
        test_dt = ((last_collision_dt-last_free_dt)/2.0+last_free_dt)

        #previous position is the position of the robot at the start if the time step
        robot1_tpos = robot1.prev_position + test_dt*robot1.velocity


        #check new robot's position if it is in collision with robots or barriers
        robot_check = False
        if self.robot_collisions:
            for robot2_index in collision_list:
                robot_check |= self.check_robot_collision(robot1_tpos,self.robot_list[robot2_index].position,robot1.robot_params["radius"] ,self.robot_list[robot2_index].robot_params["radius"] ) 
        
        if (robot_check or self.check_barriers(robot1.robot_params["radius"] ,robot1_tpos)): 
            last_collision_dt = test_dt 
        else:
            last_free_dt = test_dt

        return self.solve_collison(robot1,collision_list,last_free_dt,last_collision_dt,depth)
        
    def populate(self,num_robots,r_template):
        """
        Adds robots into the world, each robot is a deepcopy of r_template

        Parameters
        ----------
        num_robots : int
            Number of robots to add to the world
        r_template : SimulationRobot
            The robot the world will be populated with
        """
        self.num_robots = num_robots
        robot_index = 0
        for i in range(num_robots):
            r = copy.deepcopy(r_template)
            r.robot_index = robot_index
            self.robot_list.append(r)
            robot_index+=1
    def init_data_log(self,steps_num):
        """
        Initialises data log to length determined by steps_num
        Note:
            Data log is initialised to zero so if not all simulation steps are executed then data log after t_step will not be valid
        Parameters
        ----------
            steps_num : int
                 Number of steps the simulation is going to run for 
        """

        self.total_steps_num = steps_num
        self.data_log = np.zeros((self.num_robots,self.total_steps_num,3))

        if self.calculate_neighbours:
            self.current_robot_poses = np.zeros((self.num_robots,3))
        robot_index = 0
        for r in self.robot_list:
            self.data_log[robot_index,0,:2] = r.position[:2]
            self.data_log[robot_index,0,2] = r.rotation


            if self.calculate_neighbours:
                self.current_robot_poses[robot_index,:2] = r.position[:2]
                self.current_robot_poses[robot_index,2]  = r.rotation
            r.on_sim_start()
            robot_index+=1
        self.t_step=1
        
    def init_physics(self):
        """
        Initialises the collision grid. Should be called after setting bin_size and barriers
        """
        self.bin_layout = np.array((np.ceil((self.barriers[1]-self.barriers[0])/self.bin_size),np.ceil((self.barriers[3]-self.barriers[2])/self.bin_size)),dtype = 'int')
       
    def assign_robot_to_bin(self,robot):
        """
        Assigns a robot to the colision grid. Each robot's position in the colision grid is stored in the robot's bin_index parameter

        Returns
        ------
            tuple(int,int) - The robot's position the colision grid
        """

        bin_num_x = np.floor((robot.position[0]-self.barriers[0])/self.bin_size)
        bin_num_y = np.floor((robot.position[1]-self.barriers[2])/self.bin_size)

        robot.bin_index = np.array((bin_num_x,bin_num_y))
        robot.bin_index = np.clip(robot.bin_index,np.zeros(2),self.bin_layout-1)

        robot.bin_index = (int(robot.bin_index[0]),int(robot.bin_index[1]))
        return robot.bin_index
    def get_robots_collision_list(self,robot):
        """
        Compiles a list of robots the robot could be in collision with
        Parameter
        --------
        robot : SimulationRobot
            Robot to compile a colision list for
        Returns
        -------
            list - List of robot indexes the robot could be in colision with
        """
        collison_list = []
        for bin_x_index in [robot.bin_index[0]-1,robot.bin_index[0],robot.bin_index[0]+1]:#not inclusive
            for bin_y_index in [robot.bin_index[1]-1,robot.bin_index[1],robot.bin_index[1]+1]:
                if (bin_x_index >= 0 and bin_y_index >= 0 and bin_x_index < self.bin_layout[0] and bin_y_index < self.bin_layout[1]):
                    collison_list+=(self.robot_bins[bin_x_index][bin_y_index][:])
        return collison_list
    def arrange(self,settings):
        """
        Arranges the robots into a starting configuration. Should be called just before the first time_step()

        Parameters
        ---------
        settings - tuple
            Determins how the robots are arranged
            ("auto_box", box_position, robot_spacing, rand_amount) will organise the robots in the smallest possible square centered on box_position such that robots are separated by robot_spacing. rand_amount can be used to make the robot arrangement less regular by adding a random offset of magnitude rand_amount to each robot's position
            NOTE:
                This method assumes all robots are the same size based on the first robot in robot list
                Robot separation is the distance between the edges of the robots rather than their centres
                
                Robots have uniformally distrobuted rotations


        """

        mode = settings[0]
        self.robot_radius = self.robot_list[0].robot_params["radius"]  

        self.sep_dist = self.robot_radius*2

  
        if mode == "auto_box":


            boxpos = settings[1]
            robot_spacing = settings[2]
            rand_amount = settings[3]
                
            robot_index = 0

            robot_spacing/=2
            robot_spacing+= self.robot_list[0].robot_params["radius"] 
            robot_spacing*=2

            boxsize = robot_spacing*np.ceil(np.sqrt(self.num_robots))*np.ones(2) + robot_spacing/2

            grid_width = int(np.floor(boxsize[0]/robot_spacing))#Assumes same size of robots
            grid_height = int(np.floor(boxsize[1]/robot_spacing))


            if (grid_width*grid_height) < self.num_robots:
                print("Box not big enough {} spaces {} robots".format((grid_width*grid_height),self.num_robots))

            #print("{} spaces in box of size {}".format(grid_width*grid_height,boxsize))
            grid_points = [ (row,col) for row in range(grid_width) for col in range(grid_height)]
            grid_points = np.array(grid_points)
            extra_space = boxsize - np.array((grid_width-1,grid_height-1),dtype='float')*robot_spacing
            np.random.shuffle(grid_points) #{set(range(grid_width))
            for r in self.robot_list:

                r.rotation = np.random.uniform(-np.pi,np.pi)
                r.position = boxpos - np.array(boxsize)/2.0 + grid_points[robot_index]*robot_spacing +extra_space/2 + np.random.uniform(-rand_amount,rand_amount,(2,))
                robot_index+=1
                
            
        ##################Asign robots to collision bins####################   
        self.robot_bins = [ [ [] for i in range(int(self.bin_layout[1])) ] for i in range(int(self.bin_layout[0])) ]
        robot_index = 0
        
        for r in self.robot_list:
            self.assign_robot_to_bin(r)


            self.robot_bins[r.bin_index[0]][r.bin_index[1]].append(robot_index)
            robot_index+=1

        #Robots must not start off in collision or collision resolution won't work
        in_collision_r = False
        in_collision_b = False
        for r in self.robot_list:
            in_collision_b =  self.check_barriers(r.robot_params["radius"] ,r.position)
            for r2 in self.robot_list:
                if not r is r2: 
                    in_collision_r = in_collision_r or self.check_robot_collision(r.position,r2.position,r.robot_params["radius"],r2.robot_params["radius"]) 
                
            if in_collision_r or in_collision_b:
                break
        if in_collision_r or in_collision_b:
            print("After arranging robots in the world, they are in collision!")
            print("In collision with robots? {} Outside world bounds? {}".format(in_collision_r,in_collision_b))
            # self.init_data_log(1)
            # self.plot_world(0,physics_debug = True)
            # plt.show()
    def time_step(self):
        """
        Executes on time step of the simulation
        """
        if self.t_step >= self.total_steps_num:
            print("t_step > {} too large for data log".format(self.total_steps_num))




        robot_index = 0
        ###############Update position of each robot#################

        #This dictionary could have any data from the world in it when it makes sense to pre-calcated for each robot, rarther than have each robot query the world class during its control update
        self.world_sense_data = {}
        if self.calculate_neighbours:
            self.world_sense_data["current_robot_poses"] = self.current_robot_poses.copy()
            self.world_sense_data["robot_distances"] = scipy.spatial.distance.cdist( self.world_sense_data["current_robot_poses"][:,:2],self.world_sense_data["current_robot_poses"][:,:2])
        for r in self.robot_list:    
            #Move each robot according to velocity
            r.movement_update(self.dt)
            #Update robot logic 


            r.control_update(self.dt,self)


            collision_list = self.get_robots_collision_list(r)
            
            in_collision_b = self.check_barriers(r.robot_params["radius"] ,r.position)
            in_collision_r = False
            if not in_collision_b and self.robot_collisions:
                for i in collision_list:
                    r2 = self.robot_list[i]
                    if not r is r2: 
                        in_collision_r = self.check_robot_collision(r.position,r2.position,r.robot_params["radius"] ,r2.robot_params["radius"])
                    if in_collision_r:
                        break

            
            in_collision = in_collision_b or in_collision_r
            if in_collision:
                solved_dt = self.solve_collison(r,collision_list,0.0,self.dt,depth = 0)
                r.position = r.prev_position + r.velocity*solved_dt
            
            #### Reassign robots to their new colision bin based on new location###
            self.robot_bins[r.bin_index[0]][r.bin_index[1]].remove(robot_index)
            bin_index = self.assign_robot_to_bin(r)
            self.robot_bins[r.bin_index[0]][r.bin_index[1]].append(robot_index)    
            
            ###Log data, could add other measures here such as robot states etc.
            self.data_log[robot_index,self.t_step,:2] = r.position[:2]
            self.data_log[robot_index,self.t_step,2]  = r.rotation
            if self.calculate_neighbours:
                self.current_robot_poses[robot_index,:2] = r.position[:2]
                self.current_robot_poses[robot_index,2]  = r.rotation
            robot_index+=1

        self.t_step+=1
    def plot_world(self,time_step,plot_robot_trajectories = False,bounds = None, physics_debug = False):
        """
        Plots the world for debuging purposes

        Parameters
        ----------
        time_step - int
            time step to plot
        plot_robot_trajectories - bool
            Will plot the position history for each robot
        bounds - tuple 
            The bounds to the plot. Defaults to world's barriers
        physics_debug - Bool 
            If true will plot collision grid and robot's previous positions
        """
        for i in range(self.num_robots):
            if plot_robot_trajectories:    
                plt.plot(self.data_log[i,0:time_step,0],self.data_log[i,0:time_step,1],linewidth = 0.5)
            robot_body_line = gen_circle((0.0,0.0),world.robot_list[i].robot_params["radius"],20)
            plt.plot(self.data_log[i,time_step,0],self.data_log[i,time_step,1],marker='o')
            plt.plot(robot_body_line[:,0] + self.data_log[i,time_step,0],robot_body_line[:,1] + self.data_log[i,time_step,1])
            plt.text(self.data_log[i,time_step,0], self.data_log[i,time_step,1], "r{}".format(i))
            if time_step > 0 and physics_debug:
                  plt.plot(robot_body_line[:,0] + self.data_log[i,time_step-1,0],robot_body_line[:,1] + self.data_log[i,time_step-1,1],linestyle = '--',color ='r')
          
        p1 = np.array((self.barriers[0],self.barriers[2]))
        p2 = np.array((self.barriers[1],self.barriers[2]))
        p3 = np.array((self.barriers[1],self.barriers[3]))
        p4 = np.array((self.barriers[0],self.barriers[3]))

        barrier_line = np.vstack((p1,p2,p3,p4,p1))
        plt.plot(barrier_line[:,0],barrier_line[:,1])
        if physics_debug:
            print(self.bin_layout)
            for x in range(self.bin_layout[0]):
                plt.plot((x*self.bin_size+self.barriers[0],x*self.bin_size+self.barriers[0]),(self.barriers[2],self.barriers[3]),linestyle= '--',linewidth = 0.5,color = 'black')
            for y in range(self.bin_layout[1]):
                plt.plot((self.barriers[0],self.barriers[1]),(y*self.bin_size+self.barriers[2],y*self.bin_size+self.barriers[2]),linestyle= '--',linewidth = 0.5,color = 'black')
        plt.axis('equal')
        if bounds is None:
            plt.xlim((self.barriers[0]*1.1,self.barriers[1]*1.1))
            plt.ylim((self.barriers[2]*1.1,self.barriers[3]*1.1))
        else:
            plt.xlim((bounds[0],bounds[1]))
            plt.ylim((bounds[2],bounds[3]))
    def save(self,file_path):
        """
        Saves the world to a pickle file at the filepath


        Parameters
        ----------
        file_path : str
            The file path to where the world should be saved. Should include the file extension
        """

        with open(file_path, "wb+") as handle:
            pickle.dump(self,handle)

    def load(self,file_path):
        """
        Loads the world from a pickle file at the filepath


        Parameters
        ----------
        file_path : str
            The file path to where the world will be loaded from. Should include the file extension
        """
        with open(file_path, 'rb') as handle:
            return pickle.load(handle)



class WorldAnimation():
    """
    Class for producing animations of simulations

    Press t while animation is running to pause it, press again to resume playback

    Press r and y to skip one timestep backwards when paused

    Animations can also be exported to .mp4 with by passing A "save_path" to start_animation but this will required that the codecs are installed on the machine


    NOTE:
        This could be done a lot nicer with patches, but for simplicity we will stick with lines
    """
    def __init__(self,world):
        """
        Initialises the WorldAnimation class

        Parameters
        ---------
        world - SimulationWorld
            The world you want to animate
        """

        self.world = world
        
        self.figure = plt.figure()
        self.figure.canvas.mpl_connect('key_press_event', self.key_press_handler)
        blank_arr = np.zeros((1,world.num_robots))

        self.robot_lines = plt.plot(blank_arr, blank_arr)
        self.body_lines = []
        for r in world.robot_list:
            self.body_lines.append(gen_circle((0.0,0.0),r.robot_params["radius"],10))

        p1 = np.array((world.barriers[0],world.barriers[2]))
        p2 = np.array((world.barriers[1],world.barriers[2]))
        p3 = np.array((world.barriers[1],world.barriers[3]))
        p4 = np.array((world.barriers[0],world.barriers[3]))
        barrier_line = np.vstack((p1,p2,p3,p4,p1))
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.plot(barrier_line[:,0],barrier_line[:,1])
        plt.xlim((self.world.barriers[0]*1.1,self.world.barriers[1]*1.1))
        plt.ylim((self.world.barriers[2]*1.1,self.world.barriers[3]*1.1))

        self.time_text = plt.text(0.025, 1.01, "t = 0.0s",  transform = ax.transAxes, color = 'black')
        self.pause_toggle = False
        self.pause_offset = 0

        self.internal_time_step = 0
        self.saving_to_file = False

    def key_press_handler(self,event):
        """
        Key Handler call back for the figure
        
        Parameters
        ---------
        event - KeyEvent

        """
        sys.stdout.flush()
        if event.key == 't':
            self.pause_toggle = not self.pause_toggle
            print("pause_toggle {}".format(self.pause_toggle))
        if event.key == 'r':
            if self.pause_toggle:
                self.internal_time_step -= 1   
        if event.key == 'y':
            if self.pause_toggle:
                self.internal_time_step += 1 


    def animation_callback(self,time_step):
        """
        Callback for FuncAnimation
        
        Parameters
        ---------
        time_step - int
            UNUSED - Required argument form func animation, WorldAnimation uses an internal counter to allow for pausing and skipping functionality.

        """

        #Increases internal counter if not paused


        self.time_text.set_text("t = {:4.2f}s".format(self.internal_time_step*world.dt)) 


        #Updates robot's lines by using the body points generated about the origin then offsetting by the robot's position stored in the SimulationWorld's data log
        robot_index = 0
        for r_line in self.robot_lines:
            p = np.tile(self.world.data_log[robot_index,self.internal_time_step,:2],(self.body_lines[robot_index].shape[0],1))+self.body_lines[robot_index]
            r_line.set_data(p.T)
            robot_index+=1
    
        if not self.pause_toggle:
            self.internal_time_step += 1
    
        self.internal_time_step = self.internal_time_step % self.world.t_step
        if self.saving_to_file:
            print("Saving animation... {}/{} = {:4.2f}%".format(self.internal_time_step,self.world.t_step,(float(self.internal_time_step)/self.world.t_step)*100.0))
    def start_animation(self,save_path = None):
        """
        Starts the animation by calling FuncAnimation which then repeatidly calls animation_callback to animate the plot

        Parameters 
        ----------
        save_path - str
            Saves the animation in .mp4 formate to a save path relative to the current python path (sys.path[0]). This should be the folder the script is is in but isn't garaunteed.
        """
        sim_ani = animation.FuncAnimation(self.figure,self.animation_callback, range(0,world.t_step,1), fargs=None,interval=50, blit=False)
        if  save_path is not None: 
            FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
            self.saving_to_file = True 
            sim_ani.save(os.path.join(sys.path[0],save_path), writer = FFwriter)
            self.saving_to_file = False
        print("Starting Animation...")
        print("Press t while animation is running to pause it, press again to resume playback")
        print("Press r and y to skip one timestep forwards/backwards when paused")
        plt.show()


def plot_boid_debug(world,robot_indexes):
    """
        Used for visualisng the different forces acting on a set of boid_flocker robots at the current timestep

        NOTE:
            This function uses the robot's current state, to visulise the forces over time would require this data to be stored in the world's data log
        Parameters
        ----------
        world - SimulationWorld
            The world the robot's are in
        robot_indexes - list
            Indexes of the robot's visualisations should be drawn for

    """
    for i in robot_indexes:
        for nb in world.robot_list[i].neighbour_indexs:
            plt.plot((world.robot_list[i].position[0],world.robot_list[nb].position[0]),(world.robot_list[i].position[1],world.robot_list[nb].position[1]),color='purple',linewidth = 0.5,linestyle = '--')
        plt.arrow(world.robot_list[i].position[0],world.robot_list[i].position[1],world.robot_list[i].cohesion_force[0],world.robot_list[i].cohesion_force[1],color = 'r')
        plt.arrow(world.robot_list[i].position[0],world.robot_list[i].position[1],world.robot_list[i].allignment_force[0],world.robot_list[i].allignment_force[1],color = 'b')
        plt.arrow(world.robot_list[i].position[0],world.robot_list[i].position[1],world.robot_list[i].seperation_force[0],world.robot_list[i].seperation_force[1],color = 'g')
        plt.arrow(world.robot_list[i].position[0],world.robot_list[i].position[1],world.robot_list[i].centre_force[0],world.robot_list[i].centre_force[1],color = 'pink')
        
        plt.arrow(world.robot_list[i].position[0],world.robot_list[i].position[1],world.robot_list[i].target_vect[0],world.robot_list[i].target_vect[1],color = 'yellow')
        plt.arrow(world.robot_list[i].position[0],world.robot_list[i].position[1],world.robot_list[i].velocity[0],world.robot_list[i].velocity[1],color = 'grey',linestyle = '--')
        plt.plot(world.robot_list[i].neighbour_centroid[0],world.robot_list[i].neighbour_centroid[1],marker = 'x')


if __name__ == "__main__":


    #To create a simulation first create a SimulationWorld object
    world = SimulationWorld()




    #You can seed the simulation to ensure it performs the same steps every time and reproduce bugs
    #np.random.seed(0)
    world.barriers = np.array((-30,30,-30,30))

    ########################   Random walking robot parameter dictionary   #########################


    robot_params_rw = {  "algorithm"             : "random_walker",
                      "dir_change_distro"     : ("gaussian",0.0,0.5),
                      "step_len_distro"       : ("gaussian",0.0,0.1),
                      "max_speed"             : 1.0,
                      "radius"                : 0.1
                    }
    
    ########################   Boid flocking robot parameter dictionary   ########################

    robot_params_boid = {  "algorithm"             : "boid_flocker",

                          "neighbourhood_mode"    : "distance",
                          "neighbourhood_distance": 3.0,


                          # "neighbourhood_mode"    : "nearist",
                          # "neighbourhood_size"    : 1.0,

                          "seperation_dist"       : 1.0,

                          "update_period"           : 0.1,
                          
                          "cohesion_coefficient"    : 30.0,
                          "alignment_coefficient"   : 60.0,
                          "seperation_coefficient"  : 20.0,

                          "central_pull_coefficient": 10.0,

                          "rotational_p_control"  : 0.9,


                          "max_speed"             : 1.0,
                          "radius"                : 0.1
                        }

    #Robots are created using the dictionaries above as it allows all the parameters of the robot to view in on data structure
    #r = SimulationRobot(robot_params_rw) # Uncommment to switch to random walking robots
    r = SimulationRobot(robot_params_boid)
    
    #The required simulation time step  and collision bin size are then calculated. These robots currently go at fixed size but if there speed could vary the fasted possible velocity should be used here

    #This ensure a robot can't move through another robot during a single time step
    world.dt = r.robot_params["radius"]/r.robot_params["max_speed"]
    #This ensures no robot can cross into another colision bin during a single timestep
    world.bin_size = (r.robot_params["radius"] *2.1 + (r.robot_params["max_speed"])*world.dt+0.1)
    
    #Now that we've set the bin size we can initialise the collision bins    
    world.robot_collisions = True
    world.init_physics()
 

    #Adds our robot to the world
    world.populate(100,r)

    #Initialises our robot's positions are the start  of the simulation, current this is a the smallest possible square the robots can occuoy with 0.25m inbetween their bodies
    world.arrange(("auto_box",(0.0,0.0),0.25,0.0))


    #The number of steps we will simulate
    #This is the time we want to simulate divided by the amount of time we simulate each time step (dt)
    steps_num = int(1*60.0/world.dt)

    #If true this pauses the simulation every 10% of the time being simulated and and plots the worlds state - useful for debugging
    plot_snap_shots = False
    snap_shot_steps = steps_num/10

    #Execution time stats, how long on average a time_step takes to simulate and the longest time taken to simulate a time step
    exec_times = 0.0
    max_exec_time = 0.0

    world.init_data_log(steps_num)
    for step in range(steps_num-1):
        start_time = time.time()
        world.time_step()
        dt = (time.time() - start_time)
        exec_times+= dt
        if dt > max_exec_time:
            max_exec_time = dt

        if plot_snap_shots and step%snap_shot_steps == 0:
            #set bounds in the plot_world function to the following to focus on robot i 
            #bounds = np.array((-1,1,-1,1))+np.repeat(world.robot_list[i].position,2)
            world.plot_world(step+1,physics_debug = True)
            #Unccoment to see boid force visualisations in the snap shots
            #plot_boid_debug(world,[0,])
            #plt.show()

        print("Simulating... {:4.2f}%".format(float(step)/steps_num*100.0))
    world.save("world.pickle")
    print("Average execution time {}s per timestep. Maximum time per timestep = {}s".format(exec_times/float(steps_num),max_exec_time))

    #Plots the final state of the simulation and animates
    final_state_fig = plt.figure()
    world.plot_world(step,plot_robot_trajectories = False)
    plt.title("Final simulation state")

    world_anim = WorldAnimation(world)
    world_anim.start_animation("Simulation_animation.mp4")

