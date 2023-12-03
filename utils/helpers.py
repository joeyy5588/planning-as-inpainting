import torch
from typing import List, Optional, Tuple, Union
import numpy as np
from collections import deque

def apply_conditioning_1d(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, action_dim:, t] = val.clone()
        if t == 0 and val.shape[-1] == 4:
            x[:, -2:] = val[:, -2:].unsqueeze(-1).expand_as(x[:, -2:]).clone()
    return x

def apply_conditioning_2d(x, conditions):
    for t, val in conditions.items():
        # batch, channel, height, width
        x[:, :-1] = val.clone()
    return x

def apply_representation_2d(x, conditions):
    for t, val in conditions.items():
        # batch, channel, height, width
        x[:, :-2] = val.clone()
    return x

def apply_kuka_2d(x, conditions):
    for t, val in conditions.items():
        # batch, channel, height, width
        x[:, 0] = val.clone()
    return x



def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def find_start_end(grid):
    # function to get neighboring points for a given point (i, j)
    def neighbors(i, j):
        for x, y in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 1:
                yield (x, y)
    
    start, end = None, None
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] > 0:
                neighbor_count = sum(1 for _ in neighbors(i, j))
                # A starting or ending point will have only 1 neighbor
                if neighbor_count == 1:
                    if not start:
                        start = (i, j)
                    else:
                        end = (i, j)
                        start = np.array(start)
                        end = np.array(end)
                        return start, end  # Return the pair once both points are found

    if start is None:
        return np.array([0, 0]), np.array([0, 0])
    elif end is None:
        # Start traversing from the start point since start point is not None
        end = start
        stop = False
        visited = set()
        while not stop:
            stop = True
            i, j = end[0], end[1]
            visited.add((i, j))
            for x, y in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                # exclude the previous point
                if (x, y) in visited:
                    continue
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] > 0:
                    end = (x, y)
                    stop = False
                    break

        start = np.array(start)
        end = np.array(end)
        return start, end


def find_end_from_start(grid, start):
    rows, cols = grid.shape[0], grid.shape[1]
    visited = np.zeros((rows, cols))
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # Checking the validity of the next position.
    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols and visited[x][y] == 0 and grid[x][y] == 1

    # BFS function.
    def bfs(sx, sy):
        queue = deque([(sx, sy)])
        visited[sx][sy] = 1
        last_one = None
        
        while queue:
            x, y = queue.popleft()
            # List of possible moves: Up, Down, Left, Right
            if grid[x][y] == 1:
                last_one = (x, y)

            visited[x][y] = 1
            
            for dx, dy in directions:
                if is_valid(x + dx, y + dy):
                    queue.append((x + dx, y + dy))
                    visited[x + dx][y + dy] = 1

        return last_one

    end_point = bfs(start[0], start[1])
    if np.sum(visited) <= (np.sum(grid) - 2):
        for dx, dy in directions:
            new_x, new_y = end_point[0] + dx, end_point[1] + dy
            if is_valid(new_x, new_y):
                grid[new_x][new_y] = 1
            
        end_point = bfs(end_point[0], end_point[1])

    return end_point