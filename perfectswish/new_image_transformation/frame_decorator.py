class FrameDecorator:
    def __init__(self, frame, **kwargs):
        self._frame = frame
        self.__attribute_update_methods = {}

        for (key, func) in kwargs.items():
            if key.startswith("__set"):
                attribute_name = key[5:]
                self.__attribute_update_methods[attribute_name] = func

    def _update_attributes(self):
        for (key, func) in self.__attribute_update_methods.items():
            setattr(self, key, func(getattr(self._frame, key)))

    def __getattr__(self, item):
        # If the attribute is not found in the decorator, look for it in the frame
        return getattr(self._frame, item)
