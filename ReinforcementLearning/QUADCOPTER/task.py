import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1. - .2*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        if reward>1:
            reward = 1
        if reward<-1:
            reward = -1
        
        #reward=0.0
        reward += -0.5 * abs(self.target_pos[0] - self.sim.pose[0])
        reward += -0.5 * abs(self.target_pos[1] - self.sim.pose[1])
        reward += 20.0 * abs(self.sim.v[2])
        done = False
        if self.sim.pose[2] >= self.target_pos[2]: # agent has crossed the target height
            # raise TypeError
            reward += 50.0  # bonus reward
            done = True
            
        return reward, done

      

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim states (pose,velocities) &
            # sets done if episode complete(i.e. runtime>5)
            delta_reward, done_height = self.get_reward()
            reward += delta_reward
            #if done_height:
                #done = done_height
            pose_all.append(self.sim.pose)
            next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state