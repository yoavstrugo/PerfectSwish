import numpy as np


class Ball_Hit:
    ball_pos: np.array
    ball_hit_vec: np.array
    white_ball_pos: np.array
    white_ball_hit_vec: np.array
    hit_point: np.array

    def __init__(self, ball_pos, ball_hit_vec, white_ball_pos, white_ball_hit_vec, hit_point) -> None:
        self.ball_pos = ball_pos
        self.ball_hit_vec = ball_hit_vec
        self.white_ball_pos = white_ball_pos
        self.white_ball_hit_vec = white_ball_hit_vec
        self.hit_point = hit_point


class Wall_Hit:
    reflection_vector: np.array
    hit_point: np.array

    def __init__(self, reflection_vector, hit_point):
        self.reflection_vector = reflection_vector
        self.hit_point = hit_point