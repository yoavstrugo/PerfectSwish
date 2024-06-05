import numpy as np
class FrameBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_frame(self, frame):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(frame)

    def get_average_frame(self):
        if len(self.buffer) == 0:
            return None
        average_frame = np.zeros_like(self.buffer[0], dtype=np.float32)
        for frame in self.buffer:
            average_frame += frame
        average_frame /= len(self.buffer)
        average_frame = average_frame.astype(np.uint8)
        return average_frame

    def get_buffer(self):
        return self.buffer
