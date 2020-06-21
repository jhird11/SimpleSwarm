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
