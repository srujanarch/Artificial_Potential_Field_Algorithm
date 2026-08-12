# 🧲 Artificial Potential Field for Multi-Drone Navigation

A three-drone simulation implementing the **Artificial Potential Field (APF)** method for obstacle avoidance and target navigation.

## 📌 Overview

Artificial Potential Field navigation treats the environment as a virtual force field.

Two primary forces are generated:

* **Attractive force** → pulls the drone toward its target.
* **Repulsive force** → pushes the drone away from obstacles.

For a multi-drone system, additional repulsive forces can be introduced between drones.

```text
              TARGET
                 ★
                 ↑
                 │
            Attractive
                 │
                 │
          D1 ────┼────►
                 │
       ███       │       ███
       ███   Repulsive   ███
       ███       │       ███
```

## 🧠 Navigation Model

The total force acting on a drone can be represented as:

```text
Total Force =
Attractive Force
+
Obstacle Repulsion
+
Drone Repulsion
```

The resulting force determines the direction of motion.

## 🎯 Objectives

* Navigate three drones toward their targets.
* Avoid static obstacles.
* Prevent drone-drone collisions.
* Generate smooth trajectories.
* Study APF behavior in multi-agent environments.
* Analyze convergence and local-minimum problems.

## 🧲 Attractive Field

The target generates a force pulling the drone toward it.

```text
Drone ───────────────► Target
```

## 🚧 Repulsive Field

Obstacles generate forces pushing drones away.

```text
        █████
        █████
          ↑
       Repulsive
          ↑
         D1
```

## 🛸 Multi-Drone APF

Each drone considers:

```text
Target
  +
Obstacles
  +
Drone 1
  +
Drone 2
```

This allows APF to be extended from single-agent navigation to swarm navigation.

## ⚠️ Known Problems

APF has several important limitations:

### Local Minima

A drone may reach a location where:

```text
Attractive Force ≈ Repulsive Force
```

and stop moving.

### Oscillation

The drone may repeatedly move between competing forces.

### Narrow Passages

Strong repulsive forces can prevent the drone from passing through narrow areas.

### Goal Convergence

The drone may approach the target but fail to converge accurately.

## ▶️ Run

```bash
python apf_three_drones.py
```

## 📊 Evaluation

* Collision count
* Minimum obstacle clearance
* Minimum drone separation
* Path length
* Time to target
* Final goal error
* Convergence behavior

## 🔬 Future Improvements

* Local-minimum escape mechanism
* Adaptive potential functions
* Dynamic obstacle avoidance
* ORCA integration
* Waypoint guidance
* Formation constraints
* ROS 2 implementation
* PX4/Gazebo integration

## 🔗 Possible Hybrid Architecture

```text
Waypoint Planner
       ↓
      APF
       ↓
Obstacle Avoidance
       ↓
     ORCA
       ↓
Inter-Drone Collision Avoidance
       ↓
   Drone Motion
```
