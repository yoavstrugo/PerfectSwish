class Phase:
    """
    Represents a phase in the application, it is app-level.
    """

    def __init__(self, name: str):
        self.name = name

    def get_data(self):
        raise NotImplementedError

    def destroy(self):
        raise NotImplementedError