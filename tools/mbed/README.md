# CLI for CLAM-MBED

This is a CLI tool for running the dimensionality reduction algorithm `CLAM-MBED` on a dataset.

## Summary of the Algorithm

`CLAM-MBED` is a dimensionality reduction algorithm that uses a mass-spring model to embed arbitrary data into a Euclidean vector space.

We start by building a binary cluster tree over the data using `CLAM`.
The tree may or may not be balanced, depending on the user's choice.
We will eventually use the clusters from this tree and add springs between them to embed the data in the reduced space.

We represent each item in the original data as a unit mass particle in the reduced space.
We add springs connecting pairs of masses that correspond to centroids of different clusters in the tree.
Its rest length is the distance in the original space between the centroids of the clusters it connects in the original dataset.
Its actual length is the distance between the corresponding particles in the reduced space.
We use a damped harmonic oscillator model to simulate the system of particles and springs:

$$m \ddot{x} = -k(l - l_0) - \beta \dot{x}$$

where:

- $m$ is the mass of the particle (equal to the cardinality of the cluster),
- $k$ is the spring constant (chosen by the user),
- $l$ is the actual length of the spring (the Euclidean distance between the corresponding particles in the reduced space),
- $l_0$ is the rest length of the spring (the distance between the centroids of the clusters in the original space),
- $\beta$ is the damping factor (chosen by the user),
- $x$ is the position vector of the particle in the reduced space,
- $\dot{x}$ is the velocity vector of the particle in the reduced space, and
- $\ddot{x}$ is the acceleration vector of the particle in the reduced space.

The simulation proceeds for some number of *major* and *minor* steps.

For each *minor* step, we update the position of each particle as dictated by their velocities and the forces acting on them due to the springs.
Each time step is updated using the following equations:

$$
\begin{align*}
    \ddot{x} &= -\frac{k}{m}(l - l_0) - \beta \dot{x} \\
    \dot{x} &\leftarrow \dot{x} + \ddot{x} \cdot \delta t \\
    x &\leftarrow x + \dot{x} \cdot \delta t
\end{align*}
$$

where $\delta t$ is the time step size (chosen by the user).
At the end of each *minor* step, we record the total kinetic energy of the masses and the total potential energy of the springs.

We repeat the *minor* step between `patience` and `max-steps` times (both chosen by the user) or until the system reaches a "stable" state.
We define the *instability* of the system as the sum of the [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation) of the kinetic and potential energies of the system over the previous `patience` steps.
If the instability is less than a `target` threshold (chosen by the user), we consider the system to be stable.

For each *major* step, we start by calculating the *displacement ratio* of each spring, defined as $\frac{\|l - l_0\|}{l_0}$ where $\|.\|$ denotes the absolute value.
We then find the `f` fraction (chosen by the user) of springs with the highest displacement ratios and collect all clusters connected by these springs.
We then remove all springs connecting to these clusters and add springs between the children of these clusters.
We ensure that each child cluster thus added inherits the springs of its parent cluster.
Thus, each cluster that is replaced can be thought of as having been replaced by a triangle of springs connecting it to its children.
Each inherited spring will have its spring constant multiplied by a factor `dk` (chosen by the user) to make it weaker than the primary springs.
We then remove all springs whose spring constant is too small, as dictated by the `retention-depth` parameter (chosen by the user).
The threshold is $k \cdot (\Delta k) ^ d$ where $\Delta k$ is `dk` and $d$ is the `retention-depth`.
We then repeat the *minor* step for the new system until it reaches a stable state.

We repeat the *major* step until all clusters in the system are leaf clusters, i.e., they have no children.
At this point, we have a reduced representation of the original data.

The output data consist of two files:

- `<dataset-name>-reduced.npy`: A 2d array of `f32` containing the reduced representation of the input data.
- `<dataset-name>-stack.npy`: A 3d array of `f32` containing the stacked positions of the particles in the simulation at the end of each *major* step.

## Current Limitations

- The dimensionality of the reduced data is currently fixed at 3. I will add a new input parameter to allow the user to choose the dimensionality in future versions, after I have tested and optimized the current implementation on a variety of datasets.
- The input data must be in the form of a `npy` file containing a 2D array of `f32` or `f64` values. I will extend this to other formats in future versions.
- The choice of distance metrics for the original data is currently limited to Euclidean and Cosine distances. It is fairly straightforward to extend this to other metrics, and I will do so at the same time as I extend the input data formats.

## Usage

See the help message for usage instructions:

```bash
cargo run --release --package clam-mbed -- --help
```

There are two subcommands: `build` and `measure`.
These are used to perform the dimensionality reduction and measure the quality of the reduced representation, respectively.

Both subcommands require the following arguments:

- `-i, --inp-dir`: (Path) The directory containing the input data.
- `-o, --out-dir`: (Path) The directory to write the output data.
- `-n, --dataset-name`: (String) The name of the dataset.
- `-s, --seed`: (Integer) The random seed to use. Default: `42`.
- `-m, --metric`:(String) The distance metric to use over the original data. Possible values:
  - `euclidean`: Euclidean distance.
  - `cosine`: Cosine distance.

### Creating the Reduced Representation

```bash
cargo run --release --package clam-mbed -- build --help
```

The `build` subcommand requires the following additional arguments:

- `-b, --balanced`: (Bool) Whether to use a balanced clustering. Default: false.
- `-B, --beta`: (Float) The damping factor for the harmonic oscillator model representing springs. Default: `0.99`.
- `-k, --k`: (Float) The spring constant for the primary springs in the simulation. Default: `1.0`.
- `-K, --dk`: (Float) The factor by which to multiply the spring constant for the secondary springs in the simulation. Default: `0.5`.
- `-f, --f`: (Float) The fraction of springs with the highest displacement ratios whose connected clusters will be replaced by their children. Default: `0.5`.
- `-R, --retention-depth`: (Integer) The number of times a spring can be loosened before it is removed. Increasing this will exponentially increase the number of springs in the simulation. Default: `4`.
- `-t, --dt`: (Float) The time step size for the simulation. Default: `0.01`.
- `-p, --patience`: (Integer) The number of *minor* steps to wait before checking for stability. Default: `100`.
- `-M, --max-steps`: (Integer) The maximum number of *minor* steps to run the simulation for between each *major* step. Default: `10000`.
- `-T, --target`: (Float) The target instability threshold for the system. Default: `0.001`.

### Measuring the Quality of the Reduced Representation

```bash
cargo run --release --package clam-mbed -- measure --help
```

The `measure` subcommand requires the following additional arguments:

- `-q, --quality-measures`: (String) A comma-separated list of quality measures to compute. Possible values:
  - `pairwise`: The mean distortion of distances between the same pairs of points in the original and reduced spaces.
  - `triangle-inequality`: The fraction of triplets of points for which the triangle inequality holds in the original space but is violated in the reduced space.
  - `angle`: The mean of the distortion of angles between triplets of points in the original and reduced spaces.
- `-e, --exhaustive`: (Bool) Whether to compute the quality measures for all possible combinations of points or for a random subset. Default: false.

For now, I have only implemented the `pairwise` quality measure.
I will add the other two in future versions.
I will also add more quality measures in future versions.
