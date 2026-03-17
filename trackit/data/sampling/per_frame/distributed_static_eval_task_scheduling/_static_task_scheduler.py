import numpy as np
from typing import Optional, Tuple


class StaticTaskScheduler:
    def __init__(self, tasks: np.ndarray, world_size: int):
        """
        :param tasks: shape: (num_tasks,), containing the number of steps of each task
        :param world_size: number of ranks
        """
        assert np.all(tasks > 0)
        self._tasks = tasks
        self._world_size = world_size

        # Hard assign all tasks to different Ranks during initialization
        # using Round Robin polling to ensure basic load balancing
        self._rank_tasks = {
            rank: [i for i in range(len(tasks)) if i % world_size == rank]
            for rank in range(world_size)
        }

        self.reset()

    def reset(self):
        self._pending_tasks = {
            rank: list(self._rank_tasks[rank])
            for rank in range(self._world_size)
        }

        self._running_tasks = {
            rank: {} for rank in range(self._world_size)
        }

    def get_next_batch(self, rank_id: int, rank_iteration: int, batch_size: int) -> Optional[
        Tuple[Tuple[int, int], ...]]:
        assert rank_id < self._world_size

        rank_running = self._running_tasks[rank_id]
        pending = self._pending_tasks[rank_id]
        batch = []

        for task_index in list(rank_running.keys()):
            sequence_last_iteration, step_index = rank_running[task_index]
            if rank_iteration <= sequence_last_iteration:
                continue

            batch.append((task_index, step_index))

            step_index += 1
            if step_index == self._tasks[task_index]:
                del rank_running[task_index]
            else:
                rank_running[task_index] = (rank_iteration, step_index)

            if len(batch) == batch_size:
                break

        while len(batch) < batch_size:
            if len(pending) == 0:
                break

            new_task_index = pending.pop(0)
            batch.append((new_task_index, 0))

            if self._tasks[new_task_index] > 1:
                rank_running[new_task_index] = (rank_iteration, 1)

        if len(batch) == 0:
            return None

        batch.sort(key=lambda x: x[0])

        return tuple(batch)

    def get_next(self, rank_id: int, rank_iteration: int) -> Optional[Tuple[int, int]]:
        batch = self.get_next_batch(rank_id, rank_iteration, 1)
        if batch is None or len(batch) == 0:
            return None
        else:
            return batch[0]

    def is_done(self):
        # As long as all pending and running queues are cleared, the overall inference is over
        pending_empty = all(len(self._pending_tasks[r]) == 0 for r in range(self._world_size))
        running_empty = all(len(self._running_tasks[r]) == 0 for r in range(self._world_size))
        return pending_empty and running_empty
