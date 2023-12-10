import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
import chex
from gymnax.environments.spaces import Box, Discrete
from jaxmarl.environments.marbler.default_params import *
from flax import struct
from typing import Tuple, Optional, Dict
from functools import partial

import matplotlib.pyplot as plt
import matplotlib
from math import copysign

@struct.dataclass
class State:
    """Basic robotarium State"""
    p_pos: chex.Array  # [num_entities, [x, y, theta]]
    c: chex.Array  # communication state [num_agents, [dim_c]]
    done: chex.Array  # bool [num_agents, ]
    step: int  # current step
    goal: int = None 

class BaseRobotarium(MultiAgentEnv):
    def __init__(
            self,
            num_agents,
            agents = None,
            action_type = DISCRETE_ACT,
            observation_spaces = None,
            dt=DT, #In a discrete action space, dt should be at least #TODO: give recommended minimum
            goal_displacement = None,
            max_steps=100,
            color=None,
            dim_c=0,
            **kwargs
    ):
        assert(
            len(num_agents) > 0 and len(num_agents) <= 20
        ), f"Robotarium can only support up to 20 robots"
        self.num_agents = num_agents

        if agents is None:
            self.agents = [f'agent_{i}' for i in range(num_agents)]
        else:
            assert (
                len(agents) == num_agents
            ), f"Number of agents {len(agents)} does not match number of agents {num_agents}"
            self.agents = agents

        self.a_to_i = {a: i for i, a in enumerate(self.agents)}
        #TODO: self.classes

        #MARBLER supports discrete actions and continuous actions
        #Continous actions are angular and linear velocity for each agent
        #   Actual linear velocities on robots will be action*max_lienar_velocity, actual angular will be action*max_angular_velocity
        #Discrete actions move robots a specified distance left, right, up, down, or they stop
        #   Note that robots may not be able to move the specified distance if CBFs are active
        self.action_type = action_type
        if action_type == DISCRETE_ACT:
            self.action_spaces = {i: Discrete(5) for i in self.agents}
        elif action_type == CONTINUOUS_ACT:
            self.action_spaces = {i: Box(-1, 1, (2,)) for i in self.agents}
        
        assert (
            len(observation_spaces.keys()) == num_agents
        ), f"Number of observation spaces {len(observation_spaces.keys())} does not match number of agents {num_agents}"
        self.observation_spaces = observation_spaces

        self.color = (
            color
            if color is not None
            else [AGENT_COLOR] * num_agents
        )
        
        self.dim_c = dim_c  # communication channel dimensionality

        # Environment parameters
        self.max_steps = max_steps
        self.dt = dt
        self.substeps = substeps
        if goal_displacement:
            assert(
                len(goal_displacement) == num_agents
            ), f"Goal displacement array length {len(goal_displacement)} does not match number of agents"
        else:
            goal_displacement = [jnp.full((self.num_agents), 0.2)] #TODO: find a good number for this

        self.rad = jnp.concatenate(
            [jnp.full((self.num_agents), 0.12)]
        )

        if "max_velocity" in kwargs:
            self.max_vel = kwargs["max_velocity"]
            assert (
                len(self.max_vel) == self.num_entities
            ), f"Max speed array length {len(self.max_vel)} does not match number of entities {self.num_entities}"
            assert (
                max(self.max_vel) <= .2
            ), f"Max velocity {len(self.max_vel)} cannot be greater than .2m/s"
        else:
            self.max_vel = [jnp.full((self.num_agents), 0.2)]

        if "max_omega" in kwargs:
            self.max_omega = kwargs["max_omega"]
            assert (
                len(self.max_omega) == self.num_entities
            ), f"Max speed array length {len(self.max_speed)} does not match number of entities {self.num_entities}"
            assert (
                max(self.max_omega) <= 3.6
            ), f"Max omega {len(self.max_omega)} cannot be greater than 3.6"
        else:
            self.max_omega = [jnp.full((self.num_agents), 3.6)]

        #TODO: CBF creation
    
    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        u,c = self.set_actions(actions, state)

        if (
            c.shape[1] < self.dim_c
        ):  # Copying the MPE code for how communication is handled so this is needed
            c = jnp.concatenate(
                [c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))], axis=1
            )

        #No randomness here
        p_pos = self._world_step(state, u)

        done = jnp.full((self.num_agents), state.step >= self.max_steps)
        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        state = state.replace(
            p_pos=p_pos,
            c=c,
            done=done,
            step=state.step + 1,
        )

        reward = self.rewards(state)
        obs = self.get_obs(obs)
        info = {}

        return obs, state, reward, dones, info

    def _world_step(self, state: State, u: chex.array):
        return self._integrate_state(state.p_pos, u)

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def _integrate_state(self, p_pos, u):
        x, y, theta = p_pos
        vel, omega = u

        # Update the state using the kinematic equations
        new_x = x + vel * jnp.cos(theta) * self.dt
        new_y = y + vel * jnp.sin(theta) * self.dt
        new_theta = theta + omega * self.dt

        return jnp.array([new_x, new_y, new_theta])

    def set_actions(self, actions, state):
        actions = jnp.array([actions[i] for i in self.agents]).reshape(
            (self.num_agents, -1)
        )
        return self.action_decoder(self.agent_range, actions, state)

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_discrete_action(
        self, a_idx: int, action: chex.Array, state: State
    ) -> Tuple[chex.Array, chex.Array]:
        if action[0] == 0: #stop
            linear_velocity = 0
            angular_velocity = 0
        else:
            #Gets and bounds linear velocity
            linear_velocity = self.goal_displacement / self.dt
            if abs(linear_velocity) > self.max_vel:
                linear_velocity = self.max_vel * int(copysign(1,linear_velocity))
            
            current_theta = state.p_pos[a_idx][2]
            if action[0] == 1: #Left
                delta_angle = 0 - current_theta
            elif action[0] == 2: #right
                delta_angle = jnp.pi - current_theta
            elif action[0] == 2: #up
                delta_angle = jnp.pi/2 - current_theta
            elif action[0] == 2: #down
                delta_angle = 3*jnp.pi/2 - current_theta
            
            #Bounds the angle
            while delta_angle > jnp.pi:
                delta_angle -= 2 * jnp.pi
            while delta_angle < -jnp.pi:
                delta_angle += 2 * jnp.pi

            #Gets and bounds angular velocity
            angular_velocity = delta_angle / self.dt
            if abs(angular_velocity) > self.max_vel:
                angular_velocity = self.max_omega * int(copysign(1,angular_velocity))

        u = jnp.array([linear_velocity, angular_velocity])    
        c = action[1:]
        return u, c

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_continuous_action(
        self, a_idx: int, action: chex.Array, state: State
    ) -> Tuple[chex.Array, chex.Array]:
        u = jnp.array([action[0]*self.max_vel, action[1]*self.max_omega])
        c = action[2:]
        return u, c

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        #Initialize agent positions, Environment specific
        raise NotImplementedError
    
    def obs(self, state):
        #Scenario specific
        raise NotImplementedError

    def rewards(self, state):
        #Scenario specific
        raise NotImplementedError
    
    def get_info(self, state):
        #Scenario specific
        raise NotImplementedError

    def is_collision(self, a: int, b: int, state: State):
        #check if two agents are colliding
        delta_pos = state.p_pose(a) - state.p_pos(b)
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        return (dist < .24) & (a != b) #Agents in the robotarium have a 12cm radius


    def out_of_bounds(self):
        #TODO: find if agent is out of bounds
        pass