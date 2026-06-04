# Sensor Placement Research Extensions for Barrier Coverage Systems

## Overview

This repository contains two research-oriented extensions of the paper:

**"Near-optimal Sensor Placement for Detecting Stochastic Target Trajectories in Barrier Coverage Systems"**
by Kim et al. (2025)

The goal of these projects is to explore new directions in sensor placement and optimization for target detection under practical deployment constraints.

Both projects were developed as independent research explorations and toy simulations in Python.

---

## Research Motivation

The original paper focuses on optimizing sensor locations to maximize target detection probability in barrier coverage systems.

While studying the paper, I explored two possible extensions:

1. Sensor placement under budget constraints using heterogeneous sensors.
2. Advanced sensor placement strategies for improving detection performance in dense target regions.

These extensions aim to investigate how practical deployment limitations affect sensor selection and placement decisions.

---

# Project 1: Heterogeneous Sensor Placement Under Budget Constraints

## Problem

The original paper assumes all sensors are identical.

In real-world deployments, sensors often have:

* Different sensing ranges
* Different costs
* Different detection capabilities

This project investigates:

> Given a fixed budget, how should we choose sensor types and sensor locations to maximize target detection?

---

## Sensor Types

| Sensor Type | Detection Range | Cost    |
| ----------- | --------------- | ------- |
| Type A      | 4 km            | 3 units |
| Type B      | 2 km            | 1 unit  |

---

## Proposed Method

### Density-Aware Greedy Placement

At each step:

1. Evaluate all candidate locations.
2. Evaluate both sensor types.
3. Compute how many previously undetected targets would be covered.
4. Select the sensor-location pair with the highest benefit per unit cost.
5. Continue until the budget is exhausted.

---

## Baseline Method

### Uniform Placement

* Uses only cheap sensors.
* Places sensors evenly across the area.
* Does not consider target density.

---

## Results

| Method               | Detection Rate |
| -------------------- | -------------- |
| Uniform Placement    | 60%            |
| Density-Aware Greedy | 87%            |

### Key Observation

Strategically placing a few powerful sensors near dense target regions performs significantly better than uniformly distributing many low-cost sensors.

---

# Project 2: Advanced Barrier Coverage Extension

## Motivation

The original framework provides near-optimal sensor placement for stochastic target trajectories.

This extension investigates whether sensor placement decisions can be further improved by incorporating:

* Target density information
* Adaptive placement strategies
* Enhanced optimization procedures
* More realistic deployment assumptions

---

## Main Idea

Instead of treating all locations equally, the algorithm prioritizes areas with higher target concentration.

The objective is to maximize overall coverage while reducing wasted sensor resources.

---

## Methodology

The proposed framework:

* Models target movement patterns.
* Identifies high-density target regions.
* Allocates sensing resources based on target concentration.
* Compares the proposed strategy against simpler placement approaches.

---

## Findings

The experiments indicate that sensor placement decisions informed by target density consistently outperform naive deployment strategies.

This suggests that incorporating environmental information into placement decisions can significantly improve system effectiveness.

---

# Technologies Used

* Python
* NumPy
* Matplotlib
* Optimization Concepts
* Sensor Network Simulation
* Greedy Algorithms

---

# Repository Structure

```text
sensor-placement-research/
│
├── Project-1-Heterogeneous-Sensor-Placement/
│   ├── report.pdf
│   ├── simulation.py
│   └── figures/
│
├── Project-2-Advanced-Barrier-Coverage/
│   ├── report.pdf
│   ├── simulation.py
│   └── figures/
│
└── README.md
```

---

# Future Work

Possible future directions include:

* Real AIS ship traffic data integration
* Multi-UAV sensor deployment
* Reinforcement Learning based placement
* Dynamic sensor relocation
* Multi-objective optimization
* Real-time adaptive coverage systems

---

# Author

**Imon Hossain**

Department of Computer Science and Engineering (CSE)

Dhaka University of Engineering and Technology (DUET)

Bangladesh

Research Interests:

* Machine Learning
* Medical Image Analysis
* Intelligent Systems
* Sensor Networks
* Optimization
* Robotics and Autonomous Systems

---

# Reference

Kim, M., Stilwell, D. J., Yetkin, H., & Jimenez, J.

"Near-optimal Sensor Placement for Detecting Stochastic Target Trajectories in Barrier Coverage Systems"

IEEE International Systems Conference (SysCon), 2025.
