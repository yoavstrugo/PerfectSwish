import dataclasses
import math

import numpy as np
from numpy.linalg import norm

from perfectswish.utils.simulation_objects import Ball_Hit, Wall_Hit
from perfectswish.settings import REAL_BALL_RADIUS_PIXELS

PI = math.pi

def get_target_ball(balls: np.ndarray, stickend: np.ndarray, back_center: np.ndarray, front_center: np.ndarray):
    direction = normalize(front_center - back_center)
    target_ball = None
    # draw a line from the back center to the front center, starting at stickend,
    # and find the first ball that intersects with the line
    for ball in balls:
        # if ball == stickend:
        #     continue

        # calculate the distance between the ball center and the line
        distance = abs(np.cross(direction, stickend - ball))
        # if the distance is less than the radius of the ball, the ball is intersected by the line
        if distance <= REAL_BALL_RADIUS_PIXELS:
            target_ball = ball
            break

    return target_ball


def generate_path(white_ball, balls, direction, board, max_len):
    """
    Returns the expected path based on geometric calculations in find_wall_hit, find_ball_hit
    :param white_ball:
    :param balls:
    :param board:
    :param max_len:
    :return: path - list of points connected by straight lines.
    ball_hit - data about the expected hit:
    """
    path = [white_ball]
    moving_ball = white_ball.copy()
    direction = normalize(direction)
    ball_hit = None
    while len(path) < max_len and not ball_hit:
        ball_hit = find_ball_hit(moving_ball, balls, direction)
        if not ball_hit:
            wall_hit = find_wall_hit(moving_ball, board, direction)
            path.append(wall_hit.hit_point)
            moving_ball = wall_hit.hit_point
            direction = wall_hit.reflection_vector

    if ball_hit:
        path.append(ball_hit.white_ball_pos)

    return path, ball_hit


def find_wall_hit(white_ball, board, direction):
    w = board.width
    h = board.height
    dir_vec = normalize(direction)

    v1 = (np.array([-1 * white_ball[0] + REAL_BALL_RADIUS_PIXELS, 0]), [-1, 1])
    v2 = (np.array([w - white_ball[0] - REAL_BALL_RADIUS_PIXELS, 0]), [-1, 1])
    v3 = (np.array([0, -1 * white_ball[1] + REAL_BALL_RADIUS_PIXELS]), [1, -1])
    v4 = (np.array([0, h - white_ball[1] - REAL_BALL_RADIUS_PIXELS]), [1, -1])
    walls = [v1, v2, v3, v4]
    dists = [(v[1], distance_from_wall(dir_vec, v[0])) for v in walls]

    hit_point = np.zeros(2)
    new_dir = np.zeros(2)
    min_dist = math.inf
    for dist in dists:
        if min_dist > dist[1] > 0:
            min_dist = dist[1]
            hit_point = white_ball + dir_vec * dist[1]
            new_dir = np.array([dir_vec[0] * dist[0][0], dir_vec[1] * dist[0][1]])

    wall_hit = Wall_Hit(new_dir, hit_point)
    return wall_hit


def distance_from_wall(dir_vec, wall_vec):
    if np.inner(dir_vec, wall_vec) == 0:
        return math.inf

    return norm(wall_vec) * norm(wall_vec) / np.inner(dir_vec, wall_vec)


def find_ball_hit(white_ball, balls, direction):
    hits_data = []
    for ball in balls:
        hit_data = get_hit_data(ball, white_ball, direction)
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


def get_hit_data(ball, white_ball, direction):
    radii_sum = 2 * REAL_BALL_RADIUS_PIXELS

    rel_vector = ball - white_ball
    movement_vector = normalize(direction)
    projection = np.inner(rel_vector, movement_vector)

    # Check if the ball is in the positive direction of the white ball's movement
    if projection < 0:
        return None

    projection_vector = movement_vector * projection
    distance_vector = projection_vector - rel_vector
    minimal_dist = norm(distance_vector)

    # Check for balls collision
    if minimal_dist > 2 * REAL_BALL_RADIUS_PIXELS:
        return None  # The Balls won't hit!

    # Calculate the hit data:
    d = math.sqrt(radii_sum * radii_sum - minimal_dist * minimal_dist)

    ball_pos = ball
    white_hit_pos = white_ball + normalize(projection_vector) * (projection - d)

    hit_vector = ball - white_hit_pos
    unit_ball_vec = normalize(hit_vector)
    white_ball_vec = movement_vector - unit_ball_vec * np.inner(movement_vector, unit_ball_vec)
    unit_white_ball_vec = normalize(white_ball_vec)

    # white_hit_pos = ball - unit_ball_vec*(radii_sum)
    hit_point = ball - unit_ball_vec * REAL_BALL_RADIUS_PIXELS

    # hit = Ball_Hit(ball_pos, unit_ball_vec, white_hit_pos, unit_white_ball_vec, hit_point)
    hit = Ball_Hit(ball_pos, unit_ball_vec, white_hit_pos, white_ball_vec, hit_point)
    return norm(projection_vector), hit


def normalize(vec: np.array) -> np.array:
    if norm(vec) <= 0.001:
        return np.array([0, 0])

    return vec / norm(vec)
