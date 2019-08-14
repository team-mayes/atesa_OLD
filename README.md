# atesa
Python program for setting up, automating, and analyzing aimless shooting with Amber

This code is designed to promote Transition Path Sampling studies by automating aimless shooting, as well parts of setup and analysis. An excellent primer on the method can be found in [Beckham and Peters, 2010](https://pubs.acs.org/doi/abs/10.1021/bk-2010-1052.ch013). ATESA interacts directly with a batch system (either PBS or Slurm are supported) to dynamically submit, interpret, and track shooting moves based on one or more initial structures provided to it. It employs a variable length approach that periodically checks simulations for commitment to user-defined reactant and product states in order to maximize the acceptance ratio and minimize wasted computational resources.
