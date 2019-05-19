from queue import PriorityQueue


class SavedModelManager:
    def __init__(self, max_to_keep=3):
        self.max_to_keep = max_to_keep
        self.saved_models = PriorityQueue(maxsize=max_to_keep+1)

    def update(self, path, loss):
        ret = None
        # Remove the worst model
        if len(self.saved_models.queue) > self.max_to_keep:
            _, ret = self.saved_models.get()

        # Push the new data
        self.saved_models.put((-loss, path))
        return ret
