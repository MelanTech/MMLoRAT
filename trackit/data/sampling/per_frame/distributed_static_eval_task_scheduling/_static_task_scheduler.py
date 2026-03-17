import numpy as np
from typing import Optional, Tuple, List


class StaticTaskScheduler:
    def __init__(self, step_counts: np.ndarray, world_size: int):
        """
        Deterministic static task scheduler
        Fully eliminates the dynamic cursor based on the request sequence,
        regardless of the sequence of network requests of each rank,
        the assigned batch combination and internal image padding order is always consistent.
        """
        self.step_counts = step_counts
        self.world_size = world_size
        self.num_tasks = len(step_counts)
        self._schedules = {}

    def _build_schedule(self, rank_id: int, batch_size: int) -> List[List[Tuple[int, int]]]:
        tasks_for_rank = [i for i in range(self.num_tasks) if i % self.world_size == rank_id]
        task_queue = list(tasks_for_rank)

        schedule = []
        active_tasks = []

        for _ in range(batch_size):
            if len(task_queue) > 0:
                active_tasks.append([task_queue.pop(0), 0])
            else:
                break

        while len(active_tasks) > 0:
            current_batch = []
            next_active_tasks = []

            for task_idx, current_step in active_tasks:
                current_batch.append((task_idx, current_step))

                if current_step + 1 < self.step_counts[task_idx]:
                    next_active_tasks.append([task_idx, current_step + 1])
                else:
                    if len(task_queue) > 0:
                        next_active_tasks.append([task_queue.pop(0), 0])

            schedule.append(current_batch)
            active_tasks = next_active_tasks

        return schedule

    def reset(self):
        pass

    def get_next_batch(self, rank_id: int, rank_iteration: int, batch_size: int) -> Optional[List[Tuple[int, int]]]:
        if rank_id not in self._schedules:
            self._schedules[rank_id] = self._build_schedule(rank_id, batch_size)

        schedule = self._schedules[rank_id]

        if rank_iteration < len(schedule):
            return schedule[rank_iteration]
        else:
            return None
