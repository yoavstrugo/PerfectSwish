class FrameDecorator:
    def __init__(self, frame, **kwargs):
        self._frame = frame
        self.__bind_shared_data = {}

        for (key, func) in kwargs.items():
            if key.startswith("__bind"):
                attribute_name = key[6:]
                self.__attribute_update_key[attribute_name] = func

    def _update_attributes(self):
        for (attribute, shared_data_key) in self.__bind_shared_data.items():
            setattr(self, attribute, shared_data_key(getattr(self._frame, attribute)))

    def __getattr__(self, item):
        # If the attribute is not found in the decorator, look for it in the frame
        return getattr(self._frame, item)
