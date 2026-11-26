"""Explicit synthetic clutter; known occupancy, not live camera perception."""
import numpy as np


def clutter_scene():
    return [
        dict(name='central_post', center=[.25, 0., .48], half_size=[.04, .04, .19]),
        dict(name='table', center=[.50, 0., .06], half_size=[.30, .40, .035]),
        dict(name='far_post', center=[.48, .35, .40], half_size=[.035, .035, .28]),
        dict(name='side_crate', center=[.35, -.43, .25], half_size=[.12, .04, .12]),
        dict(name='rear_crate', center=[-.40, .35, .28], half_size=[.09, .06, .15]),
    ]


START = np.array([-1.2, -1.1, -1.4, -.4, .3, 0.])
GOAL = np.array([1.2, -1.1, -1.4, -.4, .3, 0.])
FIELD_ORIGIN = np.array([-.80, -.80, -.15])
FIELD_SHAPE = (65, 65, 57)
FIELD_SPACING = .025
SURFACE_EPSILON = .025
# Cover bound + additional tracking/model reserve. Voxel allowances are added
# separately in field.py and reported; never interpret total as exact clearance.
INFLATION = SURFACE_EPSILON + .010


def build_field(obstacles, spacing=FIELD_SPACING, inflation=INFLATION):
    from .field import PoissonSafetyField
    upper = FIELD_ORIGIN + FIELD_SPACING * (np.array(FIELD_SHAPE)-1)
    shape = tuple(np.ceil((upper-FIELD_ORIGIN)/spacing).astype(int)+1)
    boxes = [(np.array(o['center'])-np.array(o['half_size']),
              np.array(o['center'])+np.array(o['half_size'])) for o in obstacles]
    return PoissonSafetyField.from_boxes(FIELD_ORIGIN, shape, spacing,
                                         boxes, inflation=inflation)
