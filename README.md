# pyswarm

A simple particle based simulator of swarms coded in Python and Scipy packages

Requires:
    python*                        2.7.17
    
    scipy                         >= 1.2.2 
    
    numpy                         >= 1.16.4 
    
    matplotlib                    >= 2.2.4
    
Tested on

    Ubuntu                        Ubuntu 18.04.4 LTS

*Will be changing to python 3 soon
Run the demo run SimulationWorld from the teriminal like so:

python SimulationWorld

Examples:
Random walking

![Random Walkers demo](https://i.imgur.com/FRZGdR6.gif)

Reynolds Flocking

Firefly synchronisation

Notes

Currently neighbours are calculated using scipy.spatial.distance.cdist which computes every pair of distances between the robots, but this could be optimised using the collision box structure. But this call remains fairly fast despite large numbers of robots (>1000)





