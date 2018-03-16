from gym.envs.registration import register

from test_env.envs import *

register(
    id='Fourrooms-v1',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': '1',
    })

register(
    id='Fourrooms-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': '0'
    })

register(id='KeyDoor-v1', entry_point='test_env.envs.key_door:KeyDoor', )
