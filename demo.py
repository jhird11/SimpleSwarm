
import time 

import numpy as np
from matplotlib import pyplot as plt

from SimulationWorld import SimulationWorld,WorldAnimation
from SimulationRobot import SimulationRobot

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
        r = SimulationRobot(robot_params_boid)
        
        #r = SimulationRobot(robot_params_firefly) #Uncomment to switch to a firefly synchronisation demo

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

    world_anim_final.start_animation(start_time = None,final_time = None,speed = 1.0,time_between_frames = None)

    #Simulations can be saved to .mp4 format but this requires certain codecs
    #world_anim_final.start_animation(save_path = "world_animation.mp4",start_time = None,final_time = None,speed = 1.0,time_between_frames = None)

