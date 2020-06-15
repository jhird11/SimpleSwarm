
import numpy as np
from scipy.stats import levy,norm,uniform

def sample_distro(distro_tuple):
    """
    Samples a certain probability distrobution function (PDF) described by a tuple of parameters

    Parameters
    ----------
    distro_tuple - tuple (distrobution_name, *args)
        The PDF to sample from
    Returns
    -------
        float - the number generated from the PDF
    """
    distro_type = distro_tuple[0]
    if distro_type == "levy":
        return levy.rvs(loc=distro_tuple[1], scale=distro_tuple[2],size=1)[0]
    elif distro_type == "gaussian":
        return norm.rvs(loc=distro_tuple[1], scale=distro_tuple[2],size=1)[0]
    elif distro_type == "uniform":
        return uniform.rvs(loc=distro_tuple[1], scale=(distro_tuple[2]-distro_tuple[1]),size=1)[0]
class SimulationRobot:
    """
    Class describing a robot in SimulationWorld

    """
    def __init__(self,robot_params):
        """
        Initialises the robot class from a dictionary containing all key robot parameters

        Parameters
        ----------
        robot_params - dict
            Dictionary containing all key robot parameters        
        """

        #Kinematic information
        self.position = np.zeros(2)
        self.rotation = 0.0
        self.velocity = np.zeros(2)
        
        #Timer can be used to limit the rate at which of control loop executes

        self.timer = 0   
        self.robot_params = robot_params

        #These are assigned by the world class
        self.robot_index = None
        self.bin_index = None
    def on_sim_start(self):
        """
        Initialises the robot depending on the alogorithm set in the robot_params dict
        
        Executed during the init position log of the SimulationWorld class, so should be executed just before the first time step
        """

        if self.robot_params["algorithm"] == "boid_flocker":
            self.timer = -np.random.uniform(0.0,self.robot_params["update_period"])
            rand_dir = np.random.uniform(-np.pi,np.pi)
            self.target_vect = np.array((np.cos(rand_dir),np.sin(rand_dir)))
            self.neighbour_indexs = []
    def movement_update(self,dt):

        """
        Updates the robots position based on its current position and velocity
        

        Parameters
        ----------
        dt - float
        The time difference since the last movement update
    
        """
        self.prev_position = self.position
        self.position = self.position + self.velocity*dt 


    def control_update(self,dt,world = None):
        """
        Updates the robot's velocity and rotation according to the alogorithm set in the robot_params dict
        dt - float
            The time difference since the last control update
        world - SimulationWorld
            The world this robot exists within. Used to detect other objects in the world such as other robots and barriers
        """

        self.timer +=dt 

        if self.robot_params["algorithm"] == "random_walker":
            #Random walker consist of picking a new direction randomly at random time intervals. Step length is dictated by the PDF in "step_len_distro" 
            #Direction changes are decided by "dir_change_distro"

            if (self.timer > 0):
                self.timer = -sample_distro(self.robot_params["step_len_distro"])
                self.rotation  += sample_distro(self.robot_params["dir_change_distro"])
                self.velocity = np.array((np.cos(self.rotation),np.sin(self.rotation)))*self.robot_params["max_speed"]
        

        elif self.robot_params["algorithm"] == "boid_flocker":
            #boid flockers consist of three rules
            # Cohesion - Aim for the centroid of your neighbours
            # Alignment - Aim to align with neighbours
            # Seperation - Move alway from neighbours if two close

            #This implimentation contains an additional rule
            #Centre homing - Move towards the centre of the world (0,0)

            if (self.timer > 0):
                self.timer = -self.robot_params["update_period"]

                self.neighbour_indexs =[]

                #There are two ways of defining your neighbourhood, X closest robots to you and all the robots that are within X distance. Both are implimented here and can be changed with the "neighbourhood_mode" key  
                if self.robot_params["neighbourhood_mode"] == "distance":
                    self.neighbour_indexs = np.arange(0,world.num_robots)[world.world_sense_data["robot_distances"][self.robot_index,:] < self.robot_params["neighbourhood_distance"]]
                    self.neighbour_indexs = self.neighbour_indexs[self.neighbour_indexs != self.robot_index]

                elif self.robot_params["neighbourhood_mode"] == "nearist" and self.robot_params["neighbourhood_size"] > 0:
                    self.neighbour_indexs = np.argpartition(world.world_sense_data["robot_distances"][self.robot_index,:],self.robot_params["neighbourhood_size"])
                    self.neighbour_indexs = self.neighbour_indexs[:self.robot_params["neighbourhood_size"]+1]
                    self.neighbour_indexs = self.neighbour_indexs[self.neighbour_indexs!= self.robot_index]


                #If we have neighbours
                if len(self.neighbour_indexs) != 0:
                    
                    #Get neighbour's distances, bearings and calculate their centroid

                    self.neighbour_dists = world.world_sense_data["robot_distances"][self.robot_index][self.neighbour_indexs]
                    self.neighbour_bearings = world.world_sense_data["current_robot_poses"][self.neighbour_indexs,2]
                    self.neighbour_centroid = np.mean(world.world_sense_data["current_robot_poses"][self.neighbour_indexs,:2],axis = 0)
                    
                    #Use these to calculate the forces
                    self.cohesion_force = (self.neighbour_centroid - self.position)

                    self.allignment_force = np.array((np.cos(np.mean(self.neighbour_bearings)),np.sin(np.mean(self.neighbour_bearings))))

                    #We only apply the seperation to those neighbours which are especially close
                    close_neighbours = self.neighbour_indexs[self.neighbour_dists < self.robot_params["seperation_dist"]]                    
                    self.seperation_force = np.sum(np.tile(self.position,(close_neighbours.shape[0],1))-world.world_sense_data["current_robot_poses"][close_neighbours,:2],axis = 0)
                else:
                    #No neighbours these forces are zero
                    self.cohesion_force=np.zeros(2)
                    self.allignment_force=np.zeros(2)
                    self.seperation_force=np.zeros(2)

                #Calculate our distance from the centre
                dist_from_centre = np.sqrt(np.sum(np.power(self.position,2.0)))
                if dist_from_centre > 0: 
                    self.centre_force = -self.position/dist_from_centre
                else:
                    #Avoid a divide by zero
                    self.centre_force = np.zeros(2)

                #The final direction we want the robot to head in is the sum of each of the four force multiplied by their coefficients
                self.cohesion_force*=self.robot_params["cohesion_coefficient"]
                self.allignment_force*=self.robot_params["alignment_coefficient"] 
                self.seperation_force*=self.robot_params["seperation_coefficient"]
                self.centre_force*=self.robot_params["central_pull_coefficient"]

                self.target_vect = self.cohesion_force +  self.allignment_force + self.seperation_force  + self.centre_force

            #Proportional controller to align the robots velocity vector with the desired vector
            self.error_vect = self.target_vect - self.velocity 
            #Angle between the robot's current heading and the traget vector's heading
            #Uses arctan2 on the robot's velocity to ensure this angle will remain with -pi to pi
            angle_error = (np.arctan2(self.target_vect[1],self.target_vect[0]) -np.arctan2( self.velocity[1], self.velocity[0]))

            #Ensures the error angle is between -pi to pi
            angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

            #Proportional control to control our angular velocity
            self.rotation +=  angle_error*self.robot_params["rotational_p_control"]*dt
            self.velocity = np.array((np.cos(self.rotation),np.sin(self.rotation)))*self.robot_params["max_speed"]


