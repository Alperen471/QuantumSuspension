class BaseController:
    def act(self, state):
        raise NotImplementedError("Controller must implement act() method.")