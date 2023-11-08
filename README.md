# JaxMARL

TODO python versions etc

[**Installation**](#install) | [**Quick Start**](#start) | [**Citation**](#cite)
---

<div class="collage">
    <div class="row" align="centre">
        <img src="docs/imgs/cramped_room.gif" alt="Overcooked" width="20%">
        <img src="docs/imgs/qmix_MPE_simple_tag_v3.gif" alt="MPE" width="20%">
    </div>
</div>

## MARL environments and algorithms implemented in JAX.

JaxMARL combines ease-of-use with GPU enabled efficiency, and supports a wide range of commonly used MARL environments as well as popular baseline algorithms. Our aim is to have one library which enables thorough evaluation of MARL methods across a wide range of tasks against relevant baselines. We also introduce SMAX, a vectorised, simplifed version of the popular StarCraft Multi-Agent Challenge, which removes the need to run the StarCraft II game engine. 

## Environments

| Environment |  Source & Description | 
| --- | --- | 
| MPE | links todo |
| Overcooked |  |
| Multi-Agent Brax |  | 
| Hanabi |   |
| SMAX |  |
| STORM |  |
| Coin Game |  |
| Switch Riddle |  | 

## Baseline Algorithms

We follow CleanRL's philosophy of providing single file implementations which can be found within the `baselines` directory.

| Algorithm |  Source & Description | 
| --- | --- | 
| IPPO | links todo | 
| MAPPO |  |
| IQL |  |
| VDN |  | 
| QMIX |  |


<h2 name="install" id="install">Installation </h2>
JaxMARL can be installed directly from PyPi:
```
pip install jaxmarl  -- NOTE THIS DOES NOT WORK YET USE: pip install -e .
```
We have tested JaxMARL on Python 3.8 and 3.9 (TODO 3.9). Ensure you have the correct JAX installation for your hardware accelerator, details can be found in the [Jax Documentation](https://github.com/google/jax#installation)

To run our test scripts, some additional dependencies are required (as some scripts compare against existing implementations), these can be installed with:
```
TODO
```

<h2 name="start" id="start">Quick Start </h2>

We take inspiration from the [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and [Gymnax](https://github.com/RobertTLange/gymnax) interfaces. Actions, observations, rewards and done values are passed as dictionaries keyed by agent name, allowing for differing action and observation spaces. The done dictionary contains an additional `"__all__"` key, specifying whether the episode has ended. We follow a parallel structure, with each agent passing an action at each timestep. For ascyhronous games, such as Hanabi, a dummy action is passed for agents not acting at a given timestep.

### Basic JaxMARL API  Usage
```python 
import jax
from jaxmarl import make

key = jax.random.PRNGKey(0)
key, key_reset, key_act, key_step = jax.random.split(rng, 4)

# Initialise environment.
env = make('MPE_simple_world_comm_v3')

# Reset the environment.
obs, state = env.reset(key_reset)

# Sample random actions.
key_act = jax.random.split(key_act, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_act[i]) for i, agent in enumerate(env.agents)}

# Perform the step transition.
n_obs, n_state, reward, done, infos = env.step(key_step, state, actions)
```

## Contributing 
Please contribute! Please take a look at our contributing guide LINK TODO for how to add an environment/algorithm or submit a bug report.

<h2 name="cite" id="cite">Citing JaxMARL </h2>
If you use JaxMARL in your work, please cite us as follows:

```
TODO
```

## See Also
There are a number of other libraries which inspired this work, we encourage you to take a look!
- [Mava](https://github.com/instadeepai/Mava): JAX implementations of IPPO and MAPPO, two popular MARL algorithms.
- [Gymanx](https://github.com/RobertTLange/gymnax): Implementations of classic RL tasks including classic control, bsuite and MinAtar.
- [Jumanji](https://github.com/instadeepai/jumanji): A diverse set of environments ranging from simple games to NP-hard combinatoral problems.
- [Pgx](https://github.com/sotetsuk/pgx): JAX implementations of classic board games, such as Chess, Go and Shogi.
- [Brax](https://github.com/google/brax): A fully differentiable physics engine written in JAX, features continuous control tasks.
