from gym.envs.registration import register

from test_env.envs import *

register(
    id='Fourrooms-small-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'small',
    })

register(
    id='Fourrooms-medium-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'medium'
    })

register(
    id='Fourrooms-large-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'large',
    })

register(id='KeyDoor-v1', entry_point='test_env.envs.key_door:KeyDoor', )
