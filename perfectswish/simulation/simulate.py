import dataclasses
import math

import numpy as np
from numpy.linalg import norm

from perfectswish.api.simulation_objects import Ball_Hit, Wall_Hit

PI = math.pi


def generate_path(white_ball, balls, board, max_len):
    path = [white_ball.position]
    moving_ball = dataclasses.replace(white_ball)
    moving_ball.direction = normalize(white_ball.direction)
    ball_hit = None
    while len(path) < max_len and not ball_hit:
        ball_hit = find_ball_hit(moving_ball, balls)
        if not ball_hit:
            wall_hit = find_wall_hit(moving_ball, board)
            path.append(wall_hit.hit_point)
            moving_ball.position = wall_hit.hit_point
            moving_ball.direction = wall_hit.reflection_vector

    if ball_hit:
        path.append(ball_hit.white_ball_pos)

    return path, ball_hit


def find_wall_hit(white_ball, board):
    w = board.width
    h = board.height
    dir_vec = normalize(white_ball.direction)

    v1 = (np.array([-1 * white_ball.position[0] + white_ball.radius, 0]), [-1, 1])
    v2 = (np.array([w - white_ball.position[0] - white_ball.radius, 0]), [-1, 1])
    v3 = (np.array([0, -1 * white_ball.position[1] + white_ball.radius]), [1, -1])
    v4 = (np.array([0, h - white_ball.position[1] - white_ball.radius]), [1, -1])
    walls = [v1, v2, v3, v4]
    dists = [(v[1], distance_from_wall(dir_vec, v[0])) for v in walls]

    hit_point = np.zeros(2)
    new_dir = np.zeros(2)
    min_dist = math.inf
    for dist in dists:
        if dist[1] < min_dist and dist[1] > 0:
            min_dist = dist[1]
            hit_point = white_ball.position + dir_vec * dist[1]
            new_dir = np.array([dir_vec[0] * dist[0][0], dir_vec[1] * dist[0][1]])

    wall_hit = Wall_Hit(new_dir, hit_point)
    return wall_hit


def distance_from_wall(dir_vec, wall_vec):
    if np.inner(dir_vec, wall_vec) == 0:
        return math.inf

    return norm(wall_vec) * norm(wall_vec) / np.inner(dir_vec, wall_vec)


def find_ball_hit(white_ball, balls):
    hits_data = []
    for ball in balls:
        hit_data = get_hit_data(ball, white_ball)
        hits_data.append(hit_data)

    closest_hit = None
    for hit in hits_data:
        minimum_dist = math.inf
        if not hit:
            continue

        if hit[0] < minimum_dist:
            minimum_dist = hit[0]
            closest_hit = hit[1]

    return closest_hit


def rotation_matrix(cos_theta: float) -> np.array:
    sin_theta = math.sqrt(1 - cos_theta * cos_theta)

    rot_matrix = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
    return rot_matrix


def get_hit_data(ball, white_ball):
    radii_sum = ball.radius + white_ball.radius

    rel_vector = ball.position - white_ball.position
    movement_vector = normalize(white_ball.direction)
    projection = np.inner(rel_vector, movement_vector)

    # Check if the ball is in the positive direction of the white ball's movement
    if projection < 0:
        return None

    projection_vector = movement_vector * projection
    distance_vector = projection_vector - rel_vector
    minimal_dist = norm(distance_vector)

    # Check for balls collision
    if minimal_dist > white_ball.radius + ball.radius:
        return None

    d = math.sqrt(radii_sum * radii_sum - minimal_dist * minimal_dist)
    white_hit_pos = white_ball.position + normalize(projection_vector) * (projection - d)
    hit_vector = ball.position - white_hit_pos

    unit_ball_vec = normalize(hit_vector)
    white_ball_vec = movement_vector - unit_ball_vec * np.inner(movement_vector, unit_ball_vec)
    unit_white_ball_vec = normalize(white_ball_vec)

    # white_hit_pos = ball.position - unit_ball_vec*(radii_sum)
    hit_point = ball.position - unit_ball_vec * (ball.radius)

    hit = Ball_Hit(ball.position, unit_ball_vec, white_hit_pos, unit_white_ball_vec, hit_point)
    return norm(projection_vector), hit


def normalize(vec: np.array) -> np.array:
    if norm(vec) == 0:
        return vec
    return vec / norm(vec)
