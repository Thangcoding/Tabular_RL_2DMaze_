# Tabular_RL_2DMaze

## Overview

**Tabular_RL_2DMaze** is a small educational project that provides an **introductory overview of Reinforcement Learning (RL)** and demonstrates how classical **tabular RL algorithms** learn in a simple **2D Maze environment**.

The project includes:

- A **tutorial-style report** summarizing the basic concepts of reinforcement learning.
- Implementations of several **classical tabular RL algorithms**.
- Experiments showing the **learning and convergence behavior** of these algorithms in a 2D grid-based maze.

The goal of this project is to help beginners understand **how reinforcement learning works in practice**.

---

# Demo

Below is an example of an agent solving the maze after training.

<p align="center">
  <img src="maze_demo.gif" width="500">
</p>

The agent learns a policy that navigates from the start position to the goal while maximizing cumulative reward.

---

# Reinforcement Learning Tutorial

The first part of this project focuses on studying and summarizing the **fundamental concepts of Reinforcement Learning**.

A slide-based tutorial report was created based on the following materials:

- *Reinforcement Learning: An Introduction* — Sutton & Barto  
- *Algorithms for Reinforcement Learning* — Csaba Szepesvári  
- Reinforcement Learning lecture slides from **Google DeepMind**

### Topics Covered

The tutorial introduces the main concepts of RL, including:

- Introduction to Reinforcement Learning  
- Agent–Environment interaction  
- Markov Decision Process (MDP)  
- Value functions  
- Bellman equations  

It also explains several classical learning methods:

- Dynamic Programming  
- Monte Carlo methods  
- Temporal Difference (TD) learning  
- Importance Sampling  
- N-step bootstrapping  
- Eligibility traces  
- Exploration vs Exploitation  
- Basic ideas of planning in RL  

The tutorial aims to provide a **clear overview of the main ideas behind reinforcement learning algorithms**.

---

# 2D Maze Environment

To demonstrate how RL algorithms work, the project uses a **simple 2D Maze environment**.

In this environment:

- The maze is represented as a **grid**
- Each grid cell corresponds to a **state**
- The agent can move in four directions:
  - Up
  - Down
  - Left
  - Right
- The goal of the agent is to **reach the target position** while maximizing cumulative reward

This simple environment makes it easier to observe how different RL algorithms **learn and improve their policies over time**.

---

# Implemented Algorithms

Several classical tabular reinforcement learning algorithms are implemented:

### Q-Learning
### Double Q-Learning
### SARSA
### Monte Carlo Control
### N-step learning
### λ-trace algorithms (e.g. SARSA(λ))
### Monte Carlo Tree Search (MCTS)

These implementations help illustrate how different algorithms update value estimates and gradually learn better policies.

---

# Experiments

Experiments are performed to observe the **learning behavior and convergence of different algorithms** in the maze environment.

Typical observations include:

- How quickly each algorithm learns
- Stability of learning curves
- Differences between Monte Carlo, TD, and planning methods

The results provide an intuitive understanding of **how classical RL algorithms behave in simple environments**.

---

# Project Structure
```text
Tabular_RL_2DMaze
│
├── Agent
│ ├── DoubleQlearning.py
│ ├── LambdaTrace.py
│ ├── MCTS.py
│ ├── MonteCarlo.py
│ ├── Nstep.py
│ ├── Qlearning.py
│ └── Sarsa.py
│
├── Env
│ └── maze_space.py # 2D Maze environment
│
├── Random_Maze
│ ├── DFS.py # Maze generation using DFS
│ ├── Prime.py # Maze generation using Prim's algorithm
│ └── maze_generator.py # Maze generator interface
│
└── main.py # GUI program to run and experiment with algorithms
```
