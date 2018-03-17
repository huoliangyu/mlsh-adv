from gym.envs.registration import register

from test_env.envs import *

register(
    id='Fourrooms-small-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'small',
    })

register(
    id='Stochastic-Fourrooms-small-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'small',
        'stochastic': True,
        'slip_away': True,
    })

register(
    id='Fourrooms-medium-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'medium'
    })

register(
    id='Stochastic-Fourrooms-medium-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'medium',
        'stochastic': True,
        'slip_away': True,
    })

register(
    id='Fourrooms-large-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'large',
    })


register(
    id='Stochastic-Fourrooms-large-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'large',
        'stochastic': True,
        'slip_away': True,
    })

register(id='KeyDoor-v1', entry_point='test_env.envs.key_door:KeyDoor', )
