# SimpleSwarm

A simple particle based simulator of swarms coded in Python and Scipy packages. This simulator aims to provide a base implimentation for agent based simulations of robotic swarms.

Features
===========
- Kinematic physics and collision detection
- Neighbourhood calculation
- Simulation animation and plotting

Documentation
==============
Read the docs! https://simpleswarm.readthedocs.io/en/latest/ and run the demo (demo.py)

Requirements
==============
Requires:
    python                        2.7.17 (will be moved to python3 in the next release)
    
    scipy                         >= 1.2.2 
    
    numpy                         >= 1.16.4 
    
    matplotlib                    >= 2.2.4
    
Tested on

    Ubuntu                        Ubuntu 18.04.4 LTS




Examples:
Random walking

![Random Walkers Demo](https://i.imgur.com/FRZGdR6.gif)

Reynolds Flocking

![Flocking Demo](https://i.imgur.com/nKHXBAW.gif)

Firefly synchronisation

![Sync Demo](https://i.imgur.com/fMhaoQ0.gif)

Notes on simulator speed
========================
This simulator sacrafices speed for generalisability so while the code has tried to optimise for speed where possible more efficient methods are available if you assume certain conditions (no collisions etc.)  

Currently neighbours are calculated using scipy.spatial.distance.cdist which computes every pair of distances between the robots, but this could be optimised using the collision box structure. This call is actually faster than you think despite large numbers of robots (>1000). 

Planned features
===============
 - Manipulatable world objects (food, pheromones)
 - In world obsticles beyond outer arena walls
