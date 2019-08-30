Aimless Transision Ensemble Sampling and Analysis (ATESA)
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/team-mayes/atesa.png)](https://travis-ci.com/team-mayes/atesa)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/project/tuckerburgin/atesa/branch/master?svg=true)](https://ci.appveyor.com/project/tuckerburgin/atesa/branch/master)
[![codecov](https://codecov.io/gh/team-mayes/atesa/branch/master/graph/badge.svg)](https://codecov.io/gh/team-mayes/atesa/branch/master)

Python program for automating transition path sampling with aimless shooting using Amber

ATESA automates a particular Transition Path Sampling (TPS) workflow that uses the flexible length aimless shooting algorithm of [Mullen *et al.*](http://doi.org/10.1021/acs.jctc.5b00032) as its main workhorse. TPS is an approach to obtaining and analyzing a reaction coordinate that describes a given chemical transformation on a computationally tractable timescale. An excellent primer on the method can be found in [Beckham and Peters, 2010](https://pubs.acs.org/doi/abs/10.1021/bk-2010-1052.ch013). ATESA interacts directly with a batch system (either TORQUE or Slurm are supported) to dynamically submit, track, and interpret various simulation and analysis jobs based on one or more initial structures provided to it. It employs a variable length approach that periodically checks simulations for commitment to user-defined reactant and product states in order to maximize the acceptance ratio and minimize wasted computational resources.

ATESA contains scripts to automate flexible length aimless shooting, intertial likelihood maximization, committor analysis, and equilibrium path sampling. In combination, these components constitute a near-complete automation of the workflow between producing one or more transition path guess structures, and obtaining, validating, and analyzing a *bona fide* reaction coordinate. For more complete documentation, the user is referred to: [team-mayes/atesa/docs/atesa_docs.pdf](https://github.com/team-mayes/ATESA/blob/master/docs/atesa_docs.pdf)

### Copyright

Copyright (c) 2019, Tucker Burgin
