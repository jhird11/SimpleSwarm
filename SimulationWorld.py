#!/usr/bin/env python 

import copy
import time 
import sys
import os
import pickle
from functools import partial

import numpy as np
import scipy.spatial
from SimulationRobot import SimulationRobot

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
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
        self.max_col_resolve_depth = 2

        #Enable collision detection between robots
        self.robot_collisions = True

        #Enable calculation of neighbours of robots
        self.calculate_neighbours = True


        #How long the simulation will run for in time steps
        self.total_steps_num = 0

        #Log data with this period, should be a multiple of dt
        self.data_log_period = 0.1
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

            Data log takes is of the shape (num_robots, data_log_length,4)

            For a given robot at a given data index eg. data_log[0,0,:] = [robot x, robot y, robot rotation, robot state]
            
            Data is logged every data_log_period  in simulation time. If this is smaller than simulation timesteps (dt) then data_log_period will equal dt
            and will be logged every timestep. You may wish to make data_log_period greater than dt for large numbers of simulation steps or if you are only interested in the final state of the simulaiton

            data_log_period will be rounded down to the neariest multiple of dt and should be set before this function is called
        Parameters
        ----------
            steps_num : int
                 Number of steps the simulation is going to run for 
        """

        self.total_steps_num = steps_num

        self.log_period_steps =  int(np.max((1.0,np.floor(self.data_log_period /self.dt))))

        self.data_log_period = self.log_period_steps*self.dt

        log_length = int(np.ceil(float(self.total_steps_num)/self.log_period_steps)) 

        self.data_log = np.zeros((self.num_robots,log_length,4))

        if self.calculate_neighbours:
            self.current_robot_states = np.zeros((self.num_robots,4))

        robot_index = 0
        for r in self.robot_list:
            self.data_log[robot_index,0,:2] = r.position[:2]
            self.data_log[robot_index,0,2] = r.rotation
            self.data_log[robot_index,0,3] = r.robot_state


            if self.calculate_neighbours:
                self.current_robot_states[robot_index,:2] = r.position[:2]
                self.current_robot_states[robot_index,2]  = r.rotation
                self.current_robot_states[robot_index,3]  = r.robot_state
            r.on_sim_start()
            robot_index+=1
        
        self.data_log_index = 1
        self.t_step=1

    def get_data_log_index_at_time(self,time):
        """
        Returns the index of the data_log closest to "time" 

        NOTE 
            time is rounded down to the nearest time when data was logged. So the data log entry will never occur after the specified time
        Parameters
        ----------
        time - float
            Simulation time in seconds to get the data index for  
        returns
            int - data_log index
        """
        return np.floor(time/self.data_log_period).astype('int')

    def get_data_log_at_time(self,time):
        """
        Returns the data_log entry coresponding to the simulation time specified by time

        NOTE 
        time is rounded down to the nearest time when data was logged. So the data log entry will never occur after the specified time
        Parameters
        ----------
        time - float
            Simulation time in seconds to get the data index for  
        returns
            np.array - data_log entry at "time"
        """
        return self.get_data_log_by_index(self.get_data_log_index_at_time(time),self.get_data_log_index_at_time(time)+1)

    def get_data_log_by_index(self,index_start,index_end):
        """
        Accesses the data log between index_start and index_end (non inclusive). Indexes are clipped so they remain in the simulation period.
        This means that accessing the data log using this function at indexes beyond the end of the simulation will result in accessing the data log at the last data_log entry.
        
        See get_data_log_index_at_time to convert between simulation time and data log indexes

        
        Parameters
        ----------
        index_start -  index of the data log to begin access at 

        index_end -  index of the data log to end access at (not included)

        Returns
        np.array - datalog between these indexes 
        OR
        None if indexes are invalid or equal

        """

        if index_end > index_start:
            clipped_indexes = np.clip(range(index_start,index_end),0,self.data_log.shape[1]-1)
            return self.data_log[:,clipped_indexes,:]
        else:
            return None     
    def init_physics(self,maximum_neighbour_size = 1.0):
        """
        Initialises the collision grid. Should be called after setting bin_size and barriers
        """
        self.neighbour_bin_half_size = maximum_neighbour_size/self.bin_size
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


    def arrange(self,mode = "smallest_square", **kargs):
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



        if mode == "smallest_square" or mode == "uniform_box":


            boxpos = kargs["center_pos"]
            robot_spacing =  kargs["robot_separation"]
            rand_amount = kargs.setdefault("added_noise",0.0)
            

            robot_index = 0

            robot_spacing/=2
            robot_spacing+= self.robot_list[0].robot_params["radius"] 
            robot_spacing*=2
            if mode == "smallest_square":
                boxsize = robot_spacing*np.ceil(np.sqrt(self.num_robots))*np.ones(2) + robot_spacing/2
            elif mode == "uniform_box":
                self.sep_dist = self.robot_list[0].robot_params["radius"]*2
                boxsize = kargs["box_size"]
            
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



        logging_step = self.t_step%self.log_period_steps == 0

        robot_index = 0
        ###############Update position of each robot#################

        #This dictionary could have any data from the world in it when it makes sense to pre-calcated for each robot, rarther than have each robot query the world class during its control update
        self.world_sense_data = {}
        if self.calculate_neighbours:
            current_robot_state = self.current_robot_states.copy()

            self.world_sense_data["current_robot_poses"] =  current_robot_state[:,:3]#We copy this as it is changed in the loop below.

            #calculates the distance matrix where the element i,j represents the distance between robot i and robot j. This matrix is symetrical so that element i,j is equal to j,i
            #eg1. To get this distance between robot 0 and robot 5 this would be self.world_sense_data["robot_distances"][0,5]
            #eg2. To get the distance between robot 0 and all other robots this would be self.world_sense_data["robot_distances"][0,:]
            self.world_sense_data["robot_distances"] = scipy.spatial.distance.cdist( self.world_sense_data["current_robot_poses"][:,:2],self.world_sense_data["current_robot_poses"][:,:2])
            self.world_sense_data["current_robot_states"] = current_robot_state[:,3]        
        for r in self.robot_list:   

            #Move each robot according to velocity
            r.movement_update(self.dt)

            #Update robot logic 
            r.control_update(self.dt,self)

            #collision detection
            in_collision_b = self.check_barriers(r.robot_params["radius"] ,r.position)
            in_collision_r = False

            #create list of robots the robot might be in collision with if collision are enabled

            if self.robot_collisions:
                collision_list = self.get_robots_collision_list(r)
                collision_list.remove(robot_index)
            else:
                collision_list = []
            
            #check list of robots we might be in colision with unless we're already in collison 
            if not in_collision_b and self.robot_collisions:
                
                for i in collision_list:
                    r2 = self.robot_list[i]
                    in_collision_r = self.check_robot_collision(r.position,r2.position,r.robot_params["radius"] ,r2.robot_params["radius"])
                    if in_collision_r:
                        break
            
            in_collision = in_collision_b or in_collision_r
            if in_collision:
                #resolve the collision using binary search
                solved_dt = self.solve_collison(r,collision_list,0.0,self.dt,depth = 0)
                r.position = r.prev_position + r.velocity*solved_dt
            
            #### Reassign robots to their new colision bin based on new location###
            self.robot_bins[r.bin_index[0]][r.bin_index[1]].remove(robot_index)
            bin_index = self.assign_robot_to_bin(r)
            self.robot_bins[r.bin_index[0]][r.bin_index[1]].append(robot_index)    
            
            #Log data, could add other measures here such as robot states etc.
            #The data log doesn't log every time step allowing for smaller filesizes for long simulations
            if logging_step:
                self.data_log[robot_index,self.data_log_index,:2] = r.position[:2]
                self.data_log[robot_index,self.data_log_index,2]  = r.rotation
                self.data_log[robot_index,self.data_log_index,3]  = r.robot_state

            if self.calculate_neighbours:
                self.current_robot_states[robot_index,:2] = r.position[:2]
                self.current_robot_states[robot_index,2]  = r.rotation
                self.current_robot_states[robot_index,3]  = r.robot_state
            robot_index+=1

        if logging_step:
            self.data_log_index+=1      
        self.t_step+=1

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
    def __init__(self,world,**kargs):
        """
        Initialises the WorldAnimation class

        NOTE:
            This class will create its own figure

        Parameters
        ---------
        world - SimulationWorld
            The world you want to animate

        **kargs - misc

        Valid **kargs Keywords

        robot_trail_length - str
            Controls the length of the trail behind each robot in terms of timesteps (default 0)
        robot_trail_width - float
            Controls the width of robot trails (default = 0.1)
        robot_state_cmap - dict
            Maps between robot state (intergers) to valid matplotlib colour codes. dict should be in the form {robot_state : colour}
        "robot_labels" - bool
            If true will label each robot with its robot_index (Default = False)

        fast_plot - bool
            If true will disable robot trails, robot labels and simpify the shape used to represent each robot

        view_collision_bins - bool
            If true will plot the collision bins as dotted lines on the world (Default = False)

        viewing_bounds = tuple
            Sets the viewing window of the plot in world co-ordinates in the form (min_x,max_x,min_y,max_y) defaults to 10% larger than the world's barriers

        """

        self.world = world
        self.pause_toggle = False

        self.internal_time_step = 0
        self.saving_to_file = False

        
        self.figure = plt.figure()
        self.figure.canvas.mpl_connect('key_press_event', self.key_press_handler)
        blank_arr = np.zeros((1,world.num_robots))

        self.robot_artists = []


        #Set values for **kargs if they are not specified

        if "robot_trail_length" in kargs and kargs["robot_trail_length"]!=0:
            self.trail_length = int(kargs["robot_trail_length"])
            self.enable_trails = True
        else:
            self.enable_trails = False
            self.trail_length = 0 

        if "robot_state_cmap" in kargs:
            self.r_state_cmap = kargs["robot_state_cmap"]
        else:
            self.r_state_cmap = { 0 : 'dimgrey',
                                  1 : 'palegreen',
                                  2 : 'lightcoral',
                                  3 : 'blue'}

        self.enable_labels = kargs.setdefault("robot_labels",False)


        #Creates patches and trails for each robot based on if they are enabled or not
        robot_index = 0
        for r in world.robot_list:
           
            if "fast_plot" in kargs and kargs["fast_plot"] == True:
                self.fast_plot = True
                body_patch = patches.CirclePolygon((0,0),world.robot_list[robot_index].robot_params["radius"],resolution = 5,linewidth = kargs.setdefault("robot_body_width",0.1))
                direction_patch = None
                robot_trail = None
                robot_label = None

                self.enable_trails = False
                self.enable_labels = False
            else: 
                self.fast_plot = False   
                body_patch = patches.Circle((0,0),world.robot_list[robot_index].robot_params["radius"],linewidth = kargs.setdefault("robot_body_width",0.1))
                direction_patch = patches.Wedge((0,0), world.robot_list[robot_index].robot_params["radius"], -15, 15,color = 'black')

                if self.enable_trails:
                    robot_trail = plt.plot([], [],linewidth = kargs.setdefault("robot_trail_width",0.1))[0]
                else:
                    robot_trail = None
                
                if self.enable_labels:
                    robot_label = plt.text(0.0, 0.0, "r{}".format(robot_index),clip_on = True)

                else:
                    robot_label = None

            self.robot_artists.append((body_patch,direction_patch,robot_trail,robot_label))
                
            robot_index+=1
    
        self.ax = plt.gca()
        self.ax.set_aspect('equal')

        for artist_group in self.robot_artists:
            self.ax.add_artist(artist_group[0])
            if not artist_group[1] is  None:
                self.ax.add_artist(artist_group[1])
        
        #Plot static elements such as barrier lines and colision bins
        p1 = np.array((world.barriers[0],world.barriers[2]))
        p2 = np.array((world.barriers[1],world.barriers[2]))
        p3 = np.array((world.barriers[1],world.barriers[3]))
        p4 = np.array((world.barriers[0],world.barriers[3]))
        barrier_line = np.array((p1,p2,p3,p4,p1))
        plt.plot(barrier_line[:,0],barrier_line[:,1])

        if "view_collision_bins" in kargs and kargs["view_collision_bins"] == True:
            for x in range(self.world.bin_layout[0]):
                plt.plot((x*self.world.bin_size+self.world.barriers[0],x*self.world.bin_size+world.barriers[0]),(self.world.barriers[2],self.world.barriers[3]),linestyle= '--',linewidth = 0.5,color = 'black')
            for y in range(self.world.bin_layout[1]):
                plt.plot((self.world.barriers[0],self.world.barriers[1]),(y*self.world.bin_size+world.barriers[2],y*self.world.bin_size+self.world.barriers[2]),linestyle= '--',linewidth = 0.5,color = 'black')

        if "viewing_bounds" in kargs:
            plt.xlim((kargs["viewing_bounds"][0],kargs["viewing_bounds"][1]))
            plt.ylim((kargs["viewing_bounds"][2],kargs["viewing_bounds"][3]))
        else:
            plt.xlim((self.world.barriers[0]*1.1,self.world.barriers[1]*1.1))
            plt.ylim((self.world.barriers[2]*1.1,self.world.barriers[3]*1.1))

        self.time_text = plt.text(0.025, 1.01, "t = 0.0s",  transform = self.ax.transAxes, color = 'black')

        self.rendering_stats = False
    def update_robot_patch(self,robot_pose,body_patch,direction_patch):
        """
        Updates the plotting elements representing the robot

        Parameters
        ---------
        robot_pose - np,array
            Robot's pose in the form (x,y,rotation)
        body_patch - matplotlib.patches.Circle or matplotlib.patches.CirclePolygon depending on fast_plot
            Plotting element representing robot's body
        direction_patch - matplotlib.patches.Wedge or None depending on fast_plot
            Plotting element representing robot's direction
        """
        if not self.fast_plot:
            body_patch.center = tuple(robot_pose[:2])
            dir_patch_angle = robot_pose[2]
            dir_patch_pos = robot_pose[:2] + 0.5*body_patch.radius*np.array((np.cos(dir_patch_angle),np.sin(dir_patch_angle)))

            direction_patch.set_center(tuple(dir_patch_pos))
            direction_patch.theta1 = np.rad2deg(dir_patch_angle+np.pi) - 15
            direction_patch.theta2 = np.rad2deg(dir_patch_angle+np.pi) + 15
        
        else:
            body_patch.xy = tuple(robot_pose[:2])

        
    def key_press_handler(self,event):
        """
        Key Handler call back for the figure
        
        Parameters
        ---------
        event - KeyEvent

        """
        sys.stdout.flush()
        if event.key == 'u':
            self.pause_toggle = not self.pause_toggle
            print("pause_toggle {}".format(self.pause_toggle))
        if event.key == 'y':
            if self.pause_toggle:
                self.increase_internal_time(-world.data_log_period)
        if event.key == 'i':
            if self.pause_toggle:
                self.increase_internal_time(world.data_log_period)

              
    def update_plot(self,time):
        """
        Updates the plot elements to reflect the world at time_step
        
        Parameters
        ---------
            time - Time to plot the world at
        """

        self.time_text.set_text("t = {:4.2f}s".format(time)) 


        robot_index = 0
        trail_start_index = np.clip(world.get_data_log_index_at_time(time -  self.trail_length),0,None)
        current_data_log_index = world.get_data_log_index_at_time(time)

        trail_data = self.world.get_data_log_by_index(trail_start_index,current_data_log_index)

        for artist_group in self.robot_artists:
            
            robot_data = self.world.get_data_log_at_time(time)[robot_index][0,:]
            self.update_robot_patch(robot_data[:3],artist_group[0],artist_group[1])
            
            if not self.r_state_cmap is None:
                artist_group[0].set_facecolor(self.r_state_cmap.setdefault(robot_data[3],'red'))

            if self.enable_trails and not trail_data is None:
                artist_group[2].set_data(trail_data[robot_index,:,:2].T)
            
            if self.enable_labels:
                 artist_group[3].set_position(tuple(robot_data[:2]))
            robot_index+=1

    def plot_snapshot(self,time):
        """
        Plots a snapshot of the world at a particular timestep. Equivalent to calling WorldAnimation.update_plot(time_step)
        
        Parameters
        ---------
            time - Time step to plot the world at
        """


        self.update_plot(time)

    def increase_internal_time(self,increase):
        """
        Increases the internal time counter and performs wrapping between start and end times of the animation

        Parameters
        ---------
        increase - float
            Amount to increase the counter by (can be negative)

        """
        self.internal_time_counter += increase
        if self.internal_time_counter > self.final_time:
            self.internal_time_counter = self.start_time
        if self.internal_time_counter < self.start_time:
            self.internal_time_counter = self.final_time
    def animation_callback(self,frame_time):
        """
        Callback for FuncAnimation
        
        Parameters
        ---------
        frame_time - float
            UNUSED - Required argument form func animation, WorldAnimation uses an internal counter to allow for pausing and skipping functionality.

        """

        self.update_plot(self.internal_time_counter)

        #Increases internal counter if not paused
        if not self.pause_toggle:
            self.increase_internal_time(self.time_inc)

        if self.saving_to_file:
            print("Saving animation... {:4.2f}s out of {:4.2f}s = {:4.2f}%".format(self.internal_time_counter,self.final_time,((self.internal_time_counter-self.start_time)/self.time_inc)/self.rendering_times.shape[0]*100.0))
        elif self.rendering_stats:
            print("Animating drawing speed = {:4.2f} * desired_speed".format((time.time()-self.frame_start_time)/self.interval))
            self.frame_start_time = time.time()
    def start_animation(self,start_time = None,final_time = None, time_between_frames = None, speed = 1.0,save_path = None):
        """
        Starts the animation by calling FuncAnimation which then repeatidly calls animation_callback to animate the plot

        Parameters 
        ----------
        start_time  - float
            Start time of the simulation, will default to zero (start if simulation)

        final_time  - float
            End time of the simulation, will default to the end of the simulation


        time_between_frames - float
            The time between each frame rendered (in terms of simulation time) if None will default to the world's data log period. For best results this should be a multiple of world's data log period

        speed - float
            Playback speed of the animation, ie. the period between rendered frames 1.0 will playback at real time, while 2.0 will playback at double speed. 
            NOTE:
                The figure animation may run slower than this due to the large number of plotting elements to update. But when saving to mp4 the animation will playback correctly
                Might be best to reduce the time_between_frames at highers playback speeds 

        save_path - str
            Saves the animation in .mp4 formate to a save path
        """
        if start_time is None:
            self.start_time = 0.0
        else:
            self.start_time = start_time

        if final_time is None:
            self.final_time = self.world.t_step*self.world.dt
        else:
            self.final_time = final_time

        time_invalid = False 
        if self.start_time < 0.0:
            print("Start time less than zero!") 
            time_invalid = True
        if self.final_time < 0.0:
            print("Final time less than zero!")
            time_invalid = True
        if self.start_time > self.final_time:
            print("Start time before final time!")
            time_invalid = True
        
        if time_invalid:
            print("Invalid start or end time for animation! Defaulting to full simulaton period")
            self.start_time = 0.0
            self.final_time = self.world.t_step*self.world.dt



        if time_between_frames is None:
            self.time_inc = world.data_log_period
        else:
            self.time_inc = time_between_frames

        self.internal_time_counter = self.start_time
        
        self.rendering_times = np.arange(self.start_time,self.final_time,self.time_inc)
        


        self.interval = self.time_inc/speed

        init_func = partial(self.update_plot,0.0)

        sim_ani = animation.FuncAnimation(self.figure,self.animation_callback,self.rendering_times, init_func= init_func,fargs=None,interval = self.interval*1000, blit=False)
        if  save_path is not None: 
            FFwriter = animation.FFMpegWriter(fps = speed/self.time_inc,extra_args=['-vcodec', 'libx264'])
            self.saving_to_file = True 
            sim_ani.save(save_path, writer = FFwriter)
            self.saving_to_file = False

        if self.rendering_stats:
            self.frame_start_time = time.time()
        print("Starting Animation...")
        print("Press t while animation is running to pause it, press again to resume playback")
        print("Press r and y to skip one timestep forwards/backwards when paused")
        plt.show()




if __name__ == "__main__":

    #Simulate the world or load from the "world.pickle" file in this scripts directory
    simulate_world = True
    if simulate_world:
        #To create a simulation first create a SimulationWorld object
        world = SimulationWorld()




        #You can seed the pseudo random number generator before simulation to ensure it performs the same steps every time and reproduce bugs
        #np.random.seed(0)


        ########################   Random walking robot parameter dictionary   #########################
        #Random walking is a series of displacements and changes in heading. Each of these can be described with a certain propbablility distrobution leading to
        #different types of random motion such as brownian motion, levy walk and correlated random walks

        robot_params_rw = {  "algorithm"             : "random_walker",
                          "dir_change_distro"     : ("gaussian",0.0,0.5),
                          "step_len_distro"       : ("gaussian",0.0,0.1),
                          "max_speed"             : 1.0,
                          "radius"                : 0.1
                        }
        
        ########################   Boid flocking robot parameter dictionary   ########################
        #Renyolds flocking algorithm uses 3 rules to cause flocking behavour. This implimentation uses an additional rule which makes the robots head towards the centre 
        #of the world based on their distance
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

                              "central_pull_coefficient": 30.0,


                              "rotational_p_control"  : 0.9,
                              "max_speed"             : 1.0,
                              "radius"                : 0.1 
                            }
        ########################   Firefly inspired synchornisation robot parameter dictionary   ########################
        #These robots will change their state with a certain period (flash_period) but will start out of sync. By increasing their activation
        #value when they see nearby flashes they aim to synchronise their flashes (have the same phase)

        robot_params_firefly = {  "algorithm"             : "firefly_sync",

                                  "neighbourhood_mode"    : "distance",
                                  "neighbourhood_distance": 1.0,


                                  # "neighbourhood_mode"    : "nearist",
                                  # "neighbourhood_size"    : 20,
                                 
                                  "update_period"           : 0.1,
                                 
                                  "flash_period"                : 3.0,                          
                                  "flash_on_duration"           : 0.5,
                                  "activation_increase"         : 0.02,

                                  "static"                 : True,

                                  "dir_change_distro"     : ("uniform",-np.pi,np.pi),
                                  "step_len_distro"       : ("gaussian",0.0,0.2),

                                  "max_speed"             : 1.0,
                                  "radius"                : 0.1
                                }
        #Robots are created using the dictionaries above as it allows all the parameters of the robot to viewed in on data structure
        
        #r = SimulationRobot(robot_params_rw) # Uncommment to switch to random walking robots
        #r = SimulationRobot(robot_params_boid)
        
        r = SimulationRobot(robot_params_firefly) #Uncomment to switch to a firefly synchronisation demo

        #The required simulation time step  and collision bin size are then calculated. These robots currently go at fixed size but if there speed could vary the fasted possible velocity should be used here
        world.calculate_neighbours = True
        #This ensure a robot can't move through another robot during a single time step
        world.dt = r.robot_params["radius"]/r.robot_params["max_speed"]
        #This ensures no robot can cross into another colision bin during a single timestep
        world.bin_size = (r.robot_params["radius"]*2.0 + (r.robot_params["max_speed"])*world.dt+0.01)
        
        #Now that we've set the bin size we can initialise the collision bins    
        world.robot_collisions = False
        world.init_physics()
     

        #Adds our robot to the world
        world.populate(100,r)

        #Initialises our robot's positions are the start  of the simulation, currently this is a the smallest possible square the robots can occuoy with 0.25m inbetween their bodies

        world.arrange(mode = "smallest_square",center_pos = (0.0,0.0),robot_separation = 0.1, added_noise = 0.0)
        #world.arrange(mode = "uniform_box",center_pos = (0.0,0.0),robot_separation = 0.1, added_noise = 0.0,box_size = (world.barriers[1]-world.barriers[0],world.barriers[3]-world.barriers[2]))


        #The number of steps we will simulate
        #This is the time we want to simulate divided by the amount of time we simulate each time step (dt)
#        steps_num = int(2*60.0/world.dt)
        steps_num = int(2*60.0/world.dt)

        #If will create a window that shows the simulation's state evert so often (dictated by snap_shot_steps) - useful for debugging
        plot_snap_shots = False
        snap_shot_steps = 0.2/world.dt

        #Execution time stats, how long on average a time_step takes to simulate and the longest time taken to simulate a time step
        exec_times = 0.0
        max_exec_time = 0.0
        sim_start_time = time.time()

        #Pre-allocates memory for data logging. Saving data every timestep can result in very large file sizes so we can opt to only do it every 2 timesteps
        #world.data_log_period = world.dt*2
        world.init_data_log(steps_num)


        if plot_snap_shots:
            world_anim_snapshot = WorldAnimation(world,fast_plot = True)
            plt.show(block = False)
        
        print("Starting simulation")
        print("Swarm Size : {:d} dt = {:4.2f} measurement dt : {:4.2f}".format(world.num_robots,world.dt,world.data_log_period))
        #Main simulation loop
        for step in range(steps_num-1):
            start_time = time.time()
            world.time_step()
            dt = (time.time() - start_time)
            exec_times+= dt
            if dt > max_exec_time:
                max_exec_time = dt

            if step%max((1,steps_num/100))==0:
                print("Simulating... {:4.2f}% ETA {:4.2f} mins".format(float(step)/steps_num*100.0,(steps_num-step-1)*(exec_times/(step+1))/60.0))
            if plot_snap_shots and step%snap_shot_steps == 0:
                world_anim_snapshot.plot_snapshot(step)
                plt.draw()
                plt.pause(0.01)

        print("Average execution time {:4.4f}s per timestep. Maximum time per timestep = {:4.4f}s Time taken {:4.2f}s".format(exec_times/float(steps_num),max_exec_time,time.time()-sim_start_time))

        #Saves the world using pythons pickle format
        world.save("world.pickle")

    world = SimulationWorld().load("world.pickle")
    #Plots the final state of the simulation and animates
    world_anim_snapshot = WorldAnimation(world)
    world_anim_snapshot.plot_snapshot(world.dt*world.t_step)
    plt.title("Final simulation state")

    robot_cmap =      { 0 : 'dimgrey', 
                        1 : 'palegreen',
                        2 : 'lightcoral',
                        3 : 'blue'}   

    
    world_anim_final = WorldAnimation(world,robot_trail_length = 0, robot_trail_width = 0.1,robot_state_cmap = robot_cmap, robot_labels = False, view_collision_bins = False, viewing_bounds = world.barriers,fast_plot = False )

    world_anim_final.start_animation(save_path = "world_animation.mp4",start_time = None,final_time = None,speed = 1.0,time_between_frames = None)

