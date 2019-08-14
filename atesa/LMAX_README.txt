Code for performing likelihood maximization (optionally, inertial likelihood maximization) on aimless shooting data in order to produce a suitable reaction coordinate.

Usage:

python Burgin_LMAX.py -i input_file
	[-q include_velocities]
	[-k num_dimensions [-f fixed_dimensions] | --running num_dims]
	[--output_file output_file]
	[-h | --help]

-i: name of input file containing aimless shooting data. Each line represents a unique shooting point and must formatted as follows:

	[A/B] <- CV0 CV1 CV2 ... 

If velocity information is included, it should immediately follow the positional information and be in the same order. For example, the following represents two observations (one that commits to basin A and one to basin B) with two CVs and corresponding velocities:

	B <- 1.23 3.22 -0.12 0.87
	A <- 1.11 3.24 0.22 0.433

That the columns are not aligned and that not every number has the same precision is of no consequence.

-q: True or False. Choose True if your input_file contains velocity information as described above, or False otherwise.

-k: dimensionality of final RC. If '-q True', the RC will include k CVs AND k corresponding velocities.

-f: indices of CVs to require in the final RC. Note that though the outputs from this script 1-index CVs (i.e., the first column of data corresponds to "CV1"), the -f option uses 0-indexing (so to require e.g. CV1 and CV3, give '-f 0 2') (sorry about that). If '-q True', you must explicitly include the corresponding velocity terms (so e.g., if you want to require CV1 and its corresponding velocity is in the 12th position, you would give '-f 0 11') (again, sorry).

--running: an integer corresponding to the number of dimensions to "work up to" in a sequential manner. This option is very useful if you have a large amount of data and want to obtain a high-dimensional RC (e.g., three or more) quickly. It works by obtaining the best 1-dimensional RC, then requiring that dimension in the best 2-dimensional RC, and so on to the best 'running'-dimensional RC. If '-q True', the final RC will contain 'running' CVs AND 'running' corresponding velocities. If this option is given, any options supplied for -k and -f are ignored.

--output_file: name of output file to write to. Default behavior is to write directly to the terminal instead.

-h: shows help text and quits.

---NOTES---

There are two options listed in the help text but not here: -m and --bootstrap. The former should never need to be specified and the latter is broken, so you're best off ignoring them entirely.

I've included an example input file, 'as.out'. If you were to want to obtain a 3-dimensional RC from this data, you might call Burgin_LMAX.py as such:

	python Burgin_LMAX.py -i as.out -q True --running 3

Note that because as.out contains velocity data, you CANNOT use it with '-q False'. If you try, the velocity terms will be treated as if they were unique CVs rather than as velocities.

If you don't use the --running option, you will obtain a few output files. The 'committor_probability' file contains data for the model's fit to the error function sigmoid. Plotting this can give you a good sense of how cleanly your model fits the putative RC-commitment-probability mapping that underlies LMAX.