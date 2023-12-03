import numpy as np

class ReplayBuffer:

    def __init__(self, config):
        self._dict = {
            'path_lengths': np.zeros(config.max_n_episodes, dtype=int),
        }
        self._count = 0
        self.max_n_episodes = config.max_n_episodes
        self.max_path_length = config.max_path_length

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])


    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)

    def add_path(self, path):
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length

        if path['terminals'].any():
            assert (path['terminals'][-1] == True) and (not path['terminals'][:-1].any())

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            
            if key == 'instructions':
                if key not in self._dict:
                    self._dict[key] = []
                self._dict[key].append(path[key])
                continue

            array = path[key]
            if key not in self._dict: self._allocate(key, array)
            # print(key, array, array.shape)
            self._dict[key][self._count, :path_length] = array

        ## record path length
        self._dict['path_lengths'][self._count] = path_length

        ## increment path counter
        self._count += 1

    def finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')

