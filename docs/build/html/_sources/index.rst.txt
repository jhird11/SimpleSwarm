.. SimpleSwarm documentation master file, created by
   sphinx-quickstart on Mon Jun 22 10:24:18 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SimpleSwarm's documentation!
=======================================
SimpleSwarm is a python based simulator for modelling swarms of robots. It models each robot as disks with kinematic physics.

The simulation world and data logging are contained within the SimulationWorld class. This is then populated by instances of SimulationRobot which represent individual robots and their local rules

To see how to setup the SimulationWorld with data logging and then animate the result see demo.py

SimulationWorld
===================
.. automodule:: SimulationWorld
   :members:

SimulationRobot
=====================
.. automodule:: SimulationRobot
   :members:
