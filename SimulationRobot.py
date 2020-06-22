
import numpy as np
from scipy.stats import levy,norm,uniform

def sample_distro(distro_tuple):
    """
    Samples a certain probability distrobution function (PDF) described by a tuple of parameters

    Parameters
    ----------
    distro_tuple : tuple (distrobution_name, arg1,arg2...)
        The PDF to sample from

    Returns
    ---------
    float
        the number generated from the PDF


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
        robot_params : dict
            Dictionary containing all key robot parameters        
        """

        #Kinematic information
        self.position = np.zeros(2)
        self.rotation = 0.0
        self.velocity = np.zeros(2)
        
        #Timer can be used to limit the rate at which of control loop executes
        self.timer = 0   
        self.robot_state = 0

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

            #Initialises boids so they don't all update at the same time
            self.timer = np.random.uniform(0.0,self.robot_params["update_period"])
            rand_dir = np.random.uniform(-np.pi,np.pi)
            self.target_vect = np.array((np.cos(rand_dir),np.sin(rand_dir)))
            self.neighbour_indexs = []
        if self.robot_params["algorithm"] == "firefly_sync":
            #self.timer = np.random.uniform(0.0,self.robot_params["update_period"])
            self.movement_timer = 0
            self.activation_value = np.random.uniform(0.0,1.0)
            self.led_timer = 0
            self.prev_flashes_detected = 0
    def movement_update(self,dt):

        """
        Updates the robots position based on its current position and velocity
        

        Parameters
        ----------
        dt : float
            The time difference since the last movement update
    
        """
        self.prev_position = self.position
        self.position = self.position + self.velocity*dt 
    def get_neighbours(self,distance_matrix):
        """
        Returns the robot's neighbours via their robot_index

        Nebighbourhood params are dictated by neighbourhood_mode, neighbourhood_size and neighbourhood_distance in robot_params

        Parameters
        ----------
        distance_matrix : np.array
            NxN array for a world with N robots in it. Represents the distance between every robot in the array

        Returns
        ----------
        list
            Neighbour indexes

        """
        neighbour_indexs = []
        #There are two ways of defining your neighbourhood, X closest robots to you and all the robots that are within X distance. Both are implimented here and can be changed with the "neighbourhood_mode" key  
        if self.robot_params["neighbourhood_mode"] == "distance":
            #we select robot indexes if their coressponding distance is less than our neighbourhood distance
            neighbour_indexs = np.arange(0,distance_matrix.shape[0])[distance_matrix[self.robot_index,:] < self.robot_params["neighbourhood_distance"]]
            
        elif self.robot_params["neighbourhood_mode"] == "nearist" and self.robot_params["neighbourhood_size"] > 0:
            #argpartiion sorts the distance matrix in such a way that we are garanteed to have the X closest distances, but avoids sorting the whole thing
            neighbour_indexs = np.argpartition(distance_matrix[self.robot_index,:],self.robot_params["neighbourhood_size"])
            neighbour_indexs = neighbour_indexs[:self.robot_params["neighbourhood_size"]+1]

        neighbour_indexs = neighbour_indexs[neighbour_indexs!= self.robot_index]
        return neighbour_indexs 

    def control_update(self,dt,world = None):
        """
        Updates the robot's velocity and rotation according to the alogorithm set in the robot_params dict

        Parameters
        ----------
        dt : float
            The time difference since the last control update
        world : SimulationWorld
            The world this robot exists within. Used to detect other objects in the world such as other robots and barriers

        """

        self.timer -=dt 

        if self.robot_params["algorithm"] == "random_walker":
            #Random walker consist of picking a new direction randomly at random time intervals. Step length is dictated by the PDF in "step_len_distro" 
            #Direction changes are decided by "dir_change_distro"

            if (self.timer <= 0):
                self.timer = sample_distro(self.robot_params["step_len_distro"])
                self.rotation  += sample_distro(self.robot_params["dir_change_distro"])
                self.velocity = np.array((np.cos(self.rotation),np.sin(self.rotation)))*self.robot_params["max_speed"]
        

        elif self.robot_params["algorithm"] == "boid_flocker":
            #boid flockers consist of three rules
            # Cohesion - Aim for the centroid of your neighbours
            # Alignment - Aim to align with neighbours
            # Seperation - Move alway from neighbours if two close

            #This implimentation contains an additional rule
            #Centre homing - Move towards the centre of the world (0,0)

            if (self.timer <= 0):
                self.timer = self.robot_params["update_period"]

                self.neighbour_indexs = self.get_neighbours(world.world_sense_data["robot_distances"])



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
                if dist_from_centre > 0:  #Avoid dividing by zero
                    self.centre_force = -self.position/dist_from_centre
                else:
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


        elif self.robot_params["algorithm"] == "firefly_sync":
            #Firefly synchronisation is based on each agent moving its phase of flashing forward when its neighbours flash. Here we use robot_state to represent and LED where 0 == LED off and 1 == LED on

            #Each agent increases an internal activation value every time step when this value reaches 1 the agent flashes its led and resets its activation to zero
            #Every update cycle the agent suverys its neighbourhood for flashes. This algorithm takes the difference between the last number of activated neighbours and the current number to only count new flashes. 
            #New flashes increase the agent's activation by an amount dictated by activation_increase known as the coupling constant
            
            #When the led turns on it does so for a certain period which is more realistic for robotic agents due to latency agents

            #Further discussion of the algorithm implimentation can be found in : https://ieeexplore.ieee.org/abstract/document/5166479

            if not self.robot_params["static"]: #Moving can help synchronisation as it changes the neighbour graph, if true activates a random walk
                self.movement_timer -=dt
                if self.movement_timer <= 0:
                    self.movement_timer = sample_distro(self.robot_params["step_len_distro"])
                    self.rotation  += sample_distro(self.robot_params["dir_change_distro"])
                    self.velocity = np.array((np.cos(self.rotation),np.sin(self.rotation)))*self.robot_params["max_speed"]
            new_flashes = 0
            
            self.led_timer -= dt
            if self.robot_state == 1 and self.led_timer <=0: # Turns the robot's 'LED' on and off
                self.robot_state = 0

            if (self.timer <= 0): #Detects new flashes
                self.timer = self.robot_params["update_period"]
                self.neighbour_indexs = self.get_neighbours(world.world_sense_data["robot_distances"])
                flashes_detected = np.count_nonzero(world.world_sense_data["current_robot_states"][self.neighbour_indexs] == 1)
                new_flashes = max(0,flashes_detected - self.prev_flashes_detected)
                self.prev_flashes_detected = flashes_detected

            #Increase activation value according to formulat in https://ieeexplore.ieee.org/abstract/document/5166479
            self.activation_value+= dt/self.robot_params["flash_period"] + self.robot_params["activation_increase"]*new_flashes*self.activation_value

            if (self.activation_value > 1.0):
                self.activation_value = 0.0
                self.robot_state = 1
                self.led_timer   = self.robot_params["flash_on_duration"]
