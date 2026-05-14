from typing import Dict, List
import time


class TimeTracker:
    """ TimeTracker to track the time of different repeated steps. Example usage:
    ```
    for x in dataset_name():
        tt.start("forward", "training")
        model(x)
        ...
        tt.start("backward", "training")
        loss.backward()
        ...
        tt.start("validation")
        ...
    times = tt.stop()
    ```
    """
    def __init__(self):
        self.times: Dict[str, float] = {}  # category -> time spent
        self._categories: List[str] = []  # current tracked categories
        self._last: float = -1.0  # time of the last change in categories

    def _update(self):
        """ Update the times spent in current categories. """
        dt = time.time() - self._last
        for c in self._categories:
            self.times[c] = self.times.get(c, .0) + dt
        self._last = time.time()

    def get(self, category: str) -> float:
        """ Get the time spent in the given category. """
        self._update()
        if category not in self.times:
            raise ValueError(f"Category {category} is not tracked.")
        return self.times[category]

    def track(self, *categories: str) -> None:
        """ Track the given categories. """
        self._update()
        self._categories = list(categories)

    def reset(self, *categories: str) -> None:
        """ Reset the time spent in the given categories. """
        for c in categories:
            if c not in self.times and c not in self._categories:
                raise ValueError(f"Category {c} is not tracked.")
            self.times.pop(c, None)
            if c in self._categories:
                self._categories.remove(c)

    def stop(self) -> Dict[str, float]:
        """ Returns a dictionary of category -> durations (in seconds). """
        self._update()
        self._categories = []
        return self.times
