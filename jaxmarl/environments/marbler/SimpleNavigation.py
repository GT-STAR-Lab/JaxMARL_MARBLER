from typing import Dict
import jax
import jax.numpy as jnp
import chex
from functools import partial

from jaxmarl.environments.marbler.base import BaseRobotarium, State
from gymnax.environments.spaces import Box
from jaxmarl.environments.marbler.default_params import *
from jaxmarl.environments.multi_agent_env import State


class SimpleNaviation(BaseRobotarium):
    def __init__(
        self,
        num_agents = 4,
        action_type=DISCRETE_ACT,
        max_steps=50,
        dt=1,
        step_dist = .2,
        start_dist = .3
    ):
        self.start_dist=start_dist
        agents = ["agent_{}".format(i) for i in range(num_agents)]
        step_dists = [jnp.full((num_agents), step_dist)] #Homogeneous agents in simple navigation

        observation_spaces = {i: Box(-1.5,1.5, (5,)) for i in agents}

        super.init(
            num_agents=num_agents,
            max_steps=max_steps,
            agents=agents,
            action_type=action_type, 
            observation_spaces=observation_spaces,
            dt=dt,
            step_dist=step_dists, 
            dim_c=0, 
            num_landmarks=1,
            landmark_rad = .1
        )
    
    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Initialize with random positions"""
        p_pos = self.get_initial_states(self.start_dist, key, N=self.num_entities)
        
        state = State(
            p_pos=p_pos,
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
        )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return dictionary of agent observations"""
        landmark_pos = state.p_pos[self.num_agents]
        @partial(jax.vmap, in_axes=[0, None])
        def _observation(aidx: int, state: State) -> jnp.ndarray:
            """Return observation for agent i."""
            return jnp.concatenate(
                [state.p_pos[aidx].flatten(), landmark_pos.flatten()]
            )

        obs = _observation(self.agent_range, state)
        return {a: obs[i] for i, a in enumerate(self.agents)}

    def rewards(self, state: State) -> Dict[str, float]:
        """Assign rewards for all agents"""

        @partial(jax.vmap, in_axes=[0, None])
        def _reward(aidx: int, state: State):
            return -.1 * jnp.sum(
                jnp.square(state.p_pos[aidx] - state.p_pos[self.num_agents])
            )

        r = _reward(self.agent_range, state)
        return {agent: r[i] for i, agent in enumerate(self.agents)}

    def get_info(self, state):
        return {}
    