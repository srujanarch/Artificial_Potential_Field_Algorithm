import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ============================================================
# 3-DRONE APF
# Stable convergence + obstacle avoidance
# ============================================================


# ============================================================
# SIMULATION
# ============================================================

DT = 0.05

MAX_SPEED = 1.6

MAX_FORCE = 8.0

SIMULATION_FRAMES = 2500

ANIMATION_INTERVAL = 35


# ============================================================
# GOAL
# ============================================================

GOAL_REACHED_DISTANCE = 0.30

FINAL_APPROACH_DISTANCE = 2.0

K_GOAL = 2.5

FINAL_K_GOAL = 4.0


# ============================================================
# STATIC OBSTACLES
# ============================================================

K_STATIC_OBSTACLE = 25.0

STATIC_INFLUENCE = 2.8

STATIC_SAFETY = 0.80


# ============================================================
# MOVING OBSTACLES
# ============================================================

K_MOVING_OBSTACLE = 28.0

MOVING_INFLUENCE = 2.5

MOVING_SAFETY = 0.90

PREDICTION_TIME = 1.0


# ============================================================
# DRONE-DRONE AVOIDANCE
#
# IMPORTANT:
# Goals are 1.5 units apart.
#
# Therefore drone repulsion must NOT act at 2.2 units.
# ============================================================

K_DRONE = 14.0

DRONE_INFLUENCE = 1.15

DRONE_SAFETY = 0.65


# ============================================================
# DAMPING
# ============================================================

DAMPING = 0.88


# ============================================================
# WORLD
# ============================================================

X_MIN = -5

X_MAX = 15

Y_MIN = -5

Y_MAX = 15


# ============================================================
# INITIAL DRONE POSITIONS
# ============================================================

drone_positions = np.array([
    [0.0, -1.5],
    [0.0,  0.0],
    [0.0,  1.5]
], dtype=float)


# ============================================================
# VELOCITIES
# ============================================================

drone_velocities = np.zeros(
    (3, 2),
    dtype=float
)


# ============================================================
# GOALS
# ============================================================

goal_positions = np.array([
    [10.0, 8.5],
    [10.0, 10.0],
    [10.0, 11.5]
], dtype=float)


# ============================================================
# STATIC OBSTACLES
# ============================================================

static_obstacles = np.array([
    [3.0, 3.0],
    [4.5, 7.0],
    [6.0, 5.0],
    [7.0, 8.5],
    [8.0, 3.0]
], dtype=float)


# ============================================================
# MOVING OBSTACLES
# ============================================================

moving_obstacles = np.array([
    [5.0, 10.0],
    [7.5, 6.0],
    [2.5, 11.0]
], dtype=float)


# ============================================================
# MOVING OBSTACLE VELOCITIES
# ============================================================

moving_obstacle_velocities = np.array([
    [0.30, 0.00],
    [0.00, 0.25],
    [0.20, -0.15]
], dtype=float)


# ============================================================
# STATUS
# ============================================================

goal_reached = [
    False,
    False,
    False
]

drone_collision = False

obstacle_collision = False


# ============================================================
# PATHS
# ============================================================

paths = [
    [drone_positions[i].copy()]
    for i in range(3)
]


# ============================================================
# FORCE STORAGE
# ============================================================

current_forces = np.zeros(
    (3, 2)
)


# ============================================================
# VECTOR LIMITER
# ============================================================

def limit_vector(
        vector,
        maximum):

    magnitude = np.linalg.norm(vector)

    if magnitude < 1e-9:

        return np.zeros(2)

    if magnitude > maximum:

        return (
            vector
            / magnitude
            * maximum
        )

    return vector


# ============================================================
# GOAL FORCE
# ============================================================

def goal_force(
        position,
        goal,
        final_mode=False):

    difference = goal - position

    distance = np.linalg.norm(
        difference
    )

    if distance < 1e-9:

        return np.zeros(2)

    direction = (
        difference / distance
    )

    if final_mode:

        strength = FINAL_K_GOAL

    else:

        strength = K_GOAL

    force = (
        direction
        * strength
        * min(distance, 3.0)
    )

    return limit_vector(
        force,
        MAX_FORCE
    )


# ============================================================
# STATIC OBSTACLE FORCE
# ============================================================

def static_obstacle_force(
        position):

    total = np.zeros(2)

    for obstacle in static_obstacles:

        difference = (
            position - obstacle
        )

        distance = np.linalg.norm(
            difference
        )

        if distance < 1e-8:

            distance = 1e-8

            difference = np.array([
                1.0,
                0.0
            ])

        direction = (
            difference / distance
        )

        # Emergency
        if distance < STATIC_SAFETY:

            strength = (
                K_STATIC_OBSTACLE
                * 6.0
                * (
                    STATIC_SAFETY
                    / distance
                )
            )

            total += (
                direction
                * strength
            )

        # Normal
        elif distance < STATIC_INFLUENCE:

            strength = (
                K_STATIC_OBSTACLE
                *
                (
                    1.0 / distance
                    -
                    1.0 / STATIC_INFLUENCE
                )
                /
                (distance ** 2)
            )

            total += (
                direction
                * strength
            )

    return limit_vector(
        total,
        MAX_FORCE
    )


# ============================================================
# MOVING OBSTACLE FORCE
# ============================================================

def moving_obstacle_force(
        position,
        final_mode=False):

    total = np.zeros(2)

    for i in range(
        len(moving_obstacles)
    ):

        current = (
            moving_obstacles[i]
        )

        velocity = (
            moving_obstacle_velocities[i]
        )

        predicted = (
            current
            +
            velocity
            * PREDICTION_TIME
        )

        # ----------------------------------------------------
        # In final approach, only consider moving obstacles
        # that are actually close.
        #
        # This prevents them from disturbing convergence
        # from far away.
        # ----------------------------------------------------

        if final_mode:

            influence = 1.7

            safety = 0.85

        else:

            influence = MOVING_INFLUENCE

            safety = MOVING_SAFETY

        for obstacle in [
            current,
            predicted
        ]:

            difference = (
                position - obstacle
            )

            distance = np.linalg.norm(
                difference
            )

            if distance < 1e-8:

                distance = 1e-8

                difference = np.array([
                    1.0,
                    0.0
                ])

            direction = (
                difference / distance
            )

            # Emergency
            if distance < safety:

                strength = (
                    K_MOVING_OBSTACLE
                    * 7.0
                    * (
                        safety
                        / distance
                    )
                )

                total += (
                    direction
                    * strength
                )

            elif distance < influence:

                strength = (
                    K_MOVING_OBSTACLE
                    *
                    (
                        1.0 / distance
                        -
                        1.0 / influence
                    )
                    /
                    (distance ** 2)
                )

                total += (
                    direction
                    * strength
                )

    return limit_vector(
        total,
        MAX_FORCE
    )


# ============================================================
# DRONE-DRONE REPULSION
# ============================================================

def drone_repulsive_force(
        drone_index,
        positions,
        final_mode=False):

    total = np.zeros(2)

    current = (
        positions[drone_index]
    )

    for j in range(3):

        if j == drone_index:

            continue

        difference = (
            current
            - positions[j]
        )

        distance = np.linalg.norm(
            difference
        )

        if distance < 1e-8:

            distance = 1e-8

            difference = np.array([
                1.0,
                0.0
            ])

        direction = (
            difference / distance
        )

        # ----------------------------------------------------
        # In final mode, only emergency separation is active.
        #
        # This is the key to stopping the oscillation between
        # the three closely spaced goals.
        # ----------------------------------------------------

        if final_mode:

            influence = DRONE_SAFETY

        else:

            influence = DRONE_INFLUENCE

        if distance < DRONE_SAFETY:

            strength = (
                K_DRONE
                * 5.0
                * (
                    DRONE_SAFETY
                    / distance
                )
            )

            total += (
                direction
                * strength
            )

        elif distance < influence:

            strength = (
                K_DRONE
                *
                (
                    1.0 / distance
                    -
                    1.0 / influence
                )
                /
                (distance ** 2)
            )

            total += (
                direction
                * strength
            )

    return limit_vector(
        total,
        MAX_FORCE
    )


# ============================================================
# WALL FORCE
# ============================================================

def wall_force(
        position):

    force = np.zeros(2)

    influence = 0.7

    # Left
    d = (
        position[0] - X_MIN
    )

    if d < influence:

        force[0] += (
            K_STATIC_OBSTACLE
            *
            (
                1.0 / max(d, 0.1)
                -
                1.0 / influence
            )
        )

    # Right
    d = (
        X_MAX - position[0]
    )

    if d < influence:

        force[0] -= (
            K_STATIC_OBSTACLE
            *
            (
                1.0 / max(d, 0.1)
                -
                1.0 / influence
            )
        )

    # Bottom
    d = (
        position[1] - Y_MIN
    )

    if d < influence:

        force[1] += (
            K_STATIC_OBSTACLE
            *
            (
                1.0 / max(d, 0.1)
                -
                1.0 / influence
            )
        )

    # Top
    d = (
        Y_MAX - position[1]
    )

    if d < influence:

        force[1] -= (
            K_STATIC_OBSTACLE
            *
            (
                1.0 / max(d, 0.1)
                -
                1.0 / influence
            )
        )

    return limit_vector(
        force,
        MAX_FORCE
    )


# ============================================================
# TOTAL APF
# ============================================================

def calculate_force(
        drone_index):

    position = (
        drone_positions[drone_index]
    )

    goal = (
        goal_positions[drone_index]
    )

    distance_to_goal = np.linalg.norm(
        position - goal
    )

    # --------------------------------------------------------
    # FINAL APPROACH MODE
    # --------------------------------------------------------

    final_mode = (
        distance_to_goal
        < FINAL_APPROACH_DISTANCE
    )

    # Goal
    F_goal = goal_force(
        position,
        goal,
        final_mode
    )

    # Static obstacles
    F_static = static_obstacle_force(
        position
    )

    # Moving obstacles
    F_moving = moving_obstacle_force(
        position,
        final_mode
    )

    # Drone separation
    F_drone = drone_repulsive_force(
        drone_index,
        drone_positions,
        final_mode
    )

    # Walls
    F_wall = wall_force(
        position
    )

    # --------------------------------------------------------
    # NORMAL MODE
    # --------------------------------------------------------

    if not final_mode:

        total = (
            F_goal
            +
            F_static
            +
            F_moving
            +
            F_drone
            +
            F_wall
        )

    # --------------------------------------------------------
    # FINAL APPROACH
    #
    # Give the goal controller much greater priority.
    # --------------------------------------------------------

    else:

        total = (
            F_goal * 1.5
            +
            F_static * 0.5
            +
            F_moving * 0.7
            +
            F_drone
            +
            F_wall
        )

    return limit_vector(
        total,
        MAX_FORCE
    )


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)


# ============================================================
# OBSTACLES
# ============================================================

static_plot = ax.scatter(
    static_obstacles[:, 0],
    static_obstacles[:, 1],
    s=180,
    marker='s',
    label='Static Obstacles'
)


moving_plot = ax.scatter(
    moving_obstacles[:, 0],
    moving_obstacles[:, 1],
    s=180,
    marker='s',
    label='Moving Obstacles'
)


# ============================================================
# GOALS
# ============================================================

goal_plot = ax.scatter(
    goal_positions[:, 0],
    goal_positions[:, 1],
    s=180,
    marker='*',
    label='Goals'
)


# ============================================================
# DRONES
# ============================================================

drone_plot = ax.scatter(
    drone_positions[:, 0],
    drone_positions[:, 1],
    s=130,
    marker='o',
    label='Drones'
)


# ============================================================
# PATHS
# ============================================================

path_lines = []

for i in range(3):

    line, = ax.plot(
        [],
        [],
        linewidth=2
    )

    path_lines.append(line)


# ============================================================
# DRONE LABELS
# ============================================================

labels = []

for i in range(3):

    text = ax.text(
        drone_positions[i, 0] + 0.15,
        drone_positions[i, 1] + 0.15,
        f'D{i + 1}'
    )

    labels.append(text)


# ============================================================
# GRAPH
# ============================================================

ax.set_xlim(
    X_MIN,
    X_MAX
)

ax.set_ylim(
    Y_MIN,
    Y_MAX
)

ax.set_xlabel(
    'X'
)

ax.set_ylabel(
    'Y'
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    loc='upper left'
)

ax.set_title(
    '3-Drone APF'
)


# ============================================================
# INITIALIZE
# ============================================================

def init():

    drone_plot.set_offsets(
        drone_positions
    )

    moving_plot.set_offsets(
        moving_obstacles
    )

    goal_plot.set_offsets(
        goal_positions
    )

    for i in range(3):

        path_lines[i].set_data(
            [],
            []
        )

    return (
        drone_plot,
        moving_plot,
        goal_plot
    )


# ============================================================
# UPDATE
# ============================================================

def update(frame):

    global drone_collision
    global obstacle_collision

    # --------------------------------------------------------
    # Calculate forces
    # --------------------------------------------------------

    for i in range(3):

        if goal_reached[i]:

            current_forces[i] = (
                np.zeros(2)
            )

        else:

            current_forces[i] = (
                calculate_force(i)
            )

    # --------------------------------------------------------
    # UPDATE VELOCITY
    # --------------------------------------------------------

    for i in range(3):

        if goal_reached[i]:

            continue

        # Force -> acceleration
        drone_velocities[i] += (
            current_forces[i]
            * DT
        )

        # Damping
        drone_velocities[i] *= (
            DAMPING
        )

        # ----------------------------------------------------
        # Distance to goal
        # ----------------------------------------------------

        distance = np.linalg.norm(
            drone_positions[i]
            -
            goal_positions[i]
        )

        # ----------------------------------------------------
        # Speed control
        # ----------------------------------------------------

        if distance < FINAL_APPROACH_DISTANCE:

            speed_limit = (
                MAX_SPEED
                *
                distance
                /
                FINAL_APPROACH_DISTANCE
            )

            speed_limit = max(
                speed_limit,
                0.12
            )

        else:

            speed_limit = MAX_SPEED

        drone_velocities[i] = (
            limit_vector(
                drone_velocities[i],
                speed_limit
            )
        )

    # --------------------------------------------------------
    # UPDATE POSITION
    # --------------------------------------------------------

    for i in range(3):

        if goal_reached[i]:

            continue

        drone_positions[i] += (
            drone_velocities[i]
            * DT
        )

    # --------------------------------------------------------
    # MOVE OBSTACLES
    # --------------------------------------------------------

    for i in range(
        len(moving_obstacles)
    ):

        moving_obstacles[i] += (
            moving_obstacle_velocities[i]
            * DT
        )

        # X bounce
        if (
            moving_obstacles[i, 0]
            <= X_MIN
            or
            moving_obstacles[i, 0]
            >= X_MAX
        ):

            moving_obstacle_velocities[
                i, 0
            ] *= -1

        # Y bounce
        if (
            moving_obstacles[i, 1]
            <= Y_MIN
            or
            moving_obstacles[i, 1]
            >= Y_MAX
        ):

            moving_obstacle_velocities[
                i, 1
            ] *= -1

    # --------------------------------------------------------
    # GOAL CAPTURE
    # --------------------------------------------------------

    for i in range(3):

        distance = np.linalg.norm(
            drone_positions[i]
            -
            goal_positions[i]
        )

        if distance < GOAL_REACHED_DISTANCE:

            goal_reached[i] = True

            # EXACTLY place drone on goal
            drone_positions[i] = (
                goal_positions[i].copy()
            )

            drone_velocities[i] = (
                np.zeros(2)
            )

    # --------------------------------------------------------
    # DRONE COLLISION
    # --------------------------------------------------------

    for i in range(3):

        for j in range(i + 1, 3):

            distance = np.linalg.norm(
                drone_positions[i]
                -
                drone_positions[j]
            )

            if distance < 0.45:

                drone_collision = True

    # --------------------------------------------------------
    # STATIC OBSTACLE COLLISION
    # --------------------------------------------------------

    for i in range(3):

        for obstacle in static_obstacles:

            distance = np.linalg.norm(
                drone_positions[i]
                -
                obstacle
            )

            if distance < 0.60:

                obstacle_collision = True

    # --------------------------------------------------------
    # MOVING OBSTACLE COLLISION
    # --------------------------------------------------------

    for i in range(3):

        for obstacle in moving_obstacles:

            distance = np.linalg.norm(
                drone_positions[i]
                -
                obstacle
            )

            if distance < 0.60:

                obstacle_collision = True

    # --------------------------------------------------------
    # STORE PATH
    # --------------------------------------------------------

    for i in range(3):

        paths[i].append(
            drone_positions[i].copy()
        )

    # --------------------------------------------------------
    # DRAW DRONES
    # --------------------------------------------------------

    drone_plot.set_offsets(
        drone_positions
    )

    # --------------------------------------------------------
    # DRAW MOVING OBSTACLES
    # --------------------------------------------------------

    moving_plot.set_offsets(
        moving_obstacles
    )

    # --------------------------------------------------------
    # DRAW PATHS
    # --------------------------------------------------------

    for i in range(3):

        path = np.array(
            paths[i]
        )

        path_lines[i].set_data(
            path[:, 0],
            path[:, 1]
        )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    for i in range(3):

        labels[i].set_position(
            (
                drone_positions[i, 0] + 0.15,
                drone_positions[i, 1] + 0.15
            )
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    reached = sum(
        goal_reached
    )

    if drone_collision:

        ax.set_title(
            'DRONE-DRONE COLLISION',
            fontsize=15
        )

    elif obstacle_collision:

        ax.set_title(
            'DRONE-OBSTACLE COLLISION',
            fontsize=15
        )

    elif reached == 3:

        ax.set_title(
            'SUCCESS — ALL 3 DRONES REACHED GOALS',
            fontsize=15
        )

        ani.event_source.stop()

    else:

        ax.set_title(
            f'3-Drone APF | '
            f'Goals reached: {reached}/3',
            fontsize=15
        )

    return (
        drone_plot,
        moving_plot,
        goal_plot
    )


# ============================================================
# ANIMATION
# ============================================================

ani = animation.FuncAnimation(
    fig,
    update,
    frames=SIMULATION_FRAMES,
    init_func=init,
    interval=ANIMATION_INTERVAL,
    blit=False,
    repeat=False
)


# ============================================================
# RUN
# ============================================================

plt.show()
