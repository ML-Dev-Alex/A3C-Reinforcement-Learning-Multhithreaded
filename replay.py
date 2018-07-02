import time
from os import listdir
from os.path import isfile, join, isdir

import retro

GAME = 'Pong-Atari2600'


def render(file):
    movie = retro.Movie(file)
    movie.step()

    env = retro.make(game=movie.get_game(), state=retro.STATE_NONE, use_restricted_actions=retro.ACTIONS_ALL)
    env.initial_state = movie.get_state()
    env.reset()
    frame = 0
    framerate = 1
    while movie.step():
        if frame == framerate:
            env.render()
            time.sleep(0.005)
            frame = 0
        else:
            frame += 1

        keys = []
        for i in range(env.NUM_BUTTONS):
            keys.append(movie.get_key(i))
        _obs, _rew, _done, _info = env.step(keys)
    env.close()


if __name__ == '__main__':
    path = './recordings/{}-6/'.format(GAME)
    time.sleep(2.5)

    if isdir(path):
        time.sleep(0.5)
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files.sort()
        i = 0
        for file in files:
            i += 1
            if ".bk2" in file and i > 1080:
                print('playing', file)
                render(path + file)
    else:
        print('playing', path)
        render(path)
