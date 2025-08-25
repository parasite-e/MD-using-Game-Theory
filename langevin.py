import numpy as np

# Define physical constants (in appropriate units)
# Boltzmann constant (J/K) - You'll need to use consistent units
kB = 1.380649e-23
# or define reduced units for your simulation. For simplicity,
# let's assume we're in a system where these values make sense.
T = 300            # Temperature (K)
gamma = 1.0        # Friction coefficient (example value)
# Mass of a single Po atom (example value, could be atomic mass unit)
m = 1.0
dt = 0.001         # Time step (s)

# Number of particles and dimensions
N_atoms = 100
dimensions = 3

# For the random force, you'll calculate the 'amplitude' or 'scaling factor'
# based on the fluctuation-dissipation theorem:
# sqrt(2 * gamma * kB * T / m) * sqrt(dt) for velocity increment
# Or, if you're thinking of the random force itself:
# R_force_amplitude = np.sqrt(2 * gamma * kB * T / dt) # This is for the force directly

# When implementing, it's often more convenient to think about the *velocity increment*
# due to the random force over a time step.
# The term to add to velocity update is (R_i(t) / m) * dt
# So, (sqrt(2 * gamma * kB * T * dt) * R_i_unit_normal) / m
# Let's define the coefficient for the velocity increment:
random_velocity_coefficient = np.sqrt(2 * gamma * kB * T * dt) / m

# Inside your simulation loop (e.g., in a velocity update step):

# 1. Generate the 'Wi' (or standard normal random numbers)
#    For N_atoms particles, each having 3 dimensions (x, y, z)
#    You'll need a random number for each dimension of each particle.
#    numpy.random.normal(loc=0.0, scale=1.0, size=(N_atoms, dimensions))
#    loc=0.0 means mean is 0
#    scale=1.0 means standard deviation is 1
#    size=(N_atoms, dimensions) creates an array of the correct shape

# Example in a loop:
# Assuming you have velocities stored in a numpy array, e.g., 'velocities' of shape (N_atoms, 3)

# velocities = np.random.rand(N_atoms, dimensions) # Initialize for example

# --- Inside your time-stepping loop ---

# Calculate the random velocity increment for the current time step
random_velocity_increment = random_velocity_coefficient * \
    np.random.normal(loc=0.0, scale=1.0, size=(N_atoms, dimensions))

# Now, when you update your velocities (e.g., using a numerical integration scheme):
# For a simple Euler scheme (not recommended for MD, but illustrates the point):
# new_velocities = old_velocities + (F_LJ / m) * dt - (gamma / m) * old_velocities * dt + random_velocity_increment

# For a more robust integrator like Velocity Verlet with a Langevin thermostat,
# the terms are integrated slightly differently, but the generation of the
# random numbers remains the same. The `random_velocity_increment`
# term is what introduces the stochasticity.

print(f"Random velocity coefficient: {random_velocity_coefficient:.2e}")
print(
    f"Shape of generated random_velocity_increment: {random_velocity_increment.shape}")
print(
    f"Example random velocity increment for first atom:\n{random_velocity_increment[0]}")

# Important Considerations for a Full Simulation:

# 1.  **Units:** Molecular dynamics simulations often use "reduced units" to avoid dealing with extremely small numbers like `kB` directly. This involves choosing base units for mass, length, and energy. For Lennard-Jones, common choices are:
#     * Length: $\sigma$ (e.g., 1.0)
#     * Energy: $\epsilon$ (e.g., 1.0)
#     * Mass: $m$ (e.g., 1.0 for the atom's mass)
#     This then defines derived units for time, temperature, force, etc. If you use reduced units, `kB` might effectively become `1.0` or some other simple constant related to your temperature scaling.

# 2.  **Numerical Integrator:**
#     * **Euler-Maruyama:** Simplest to implement, but generally not stable or accurate enough for molecular dynamics simulations, especially for energy conservation over long times.
#     * **Velocity Verlet with Langevin Thermostat:** This is a much more common and stable integrator for NVT (constant Number, Volume, Temperature) simulations with Langevin dynamics. It correctly combines the deterministic forces, friction, and random kicks.
#         The equations for Velocity Verlet with Langevin are typically:
#         $$ \mathbf{v}(t + \Delta t/2) = \mathbf{v}(t) + \frac{\mathbf{F}(t)}{2m} \Delta t $$
#         $$ \mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t + \Delta t/2) \Delta t $$
#         $$ \mathbf{v}(t + \Delta t) = c_1 \mathbf{v}(t + \Delta t/2) + c_2 \boldsymbol{\xi} + \frac{\mathbf{F}(t + \Delta t)}{2m} \Delta t $$
#         Where $c_1 = e^{-\gamma \Delta t}$ and $c_2 = \sqrt{\frac{k_B T}{m}(1-e^{-2\gamma \Delta t})}$. $\boldsymbol{\xi}$ is a vector of standard normal random numbers. (There are variations in these coefficients depending on the exact derivation, but the principle is the same).
#         The random part comes in the form of $c_2 \boldsymbol{\xi}$.

# 3.  **Lennard-Jones Force Calculation:** You'll need a function that, given the positions of all atoms, calculates the Lennard-Jones force on each atom due to all other atoms. This typically involves:
#     * Iterating through all unique pairs of atoms.
#     * Calculating the distance $r_{ij}$.
#     * Calculating the force magnitude and direction: $\mathbf{F}_{ij} = -\nabla U^{LJ}(r_{ij})$.
#     * Summing these pairwise forces to get the total force on each atom.
#     * **Periodic Boundary Conditions:** For bulk systems, you'll almost certainly need to implement periodic boundary conditions to avoid surface effects.

# 4.  **Energy Calculation:**
#     * **Kinetic Energy:** $KE = \sum_i \frac{1}{2} m_i |\mathbf{v}_i|^2$
#     * **Potential Energy (Lennard-Jones):** $PE = \sum_{i<j} U^{LJ}(r_{ij})$ (sum over unique pairs)
#     * **Total Energy:** $E_{total} = KE + PE$

# Here's a conceptual outline of how a minimal Python simulation loop would look (highly simplified, not a production-ready code):

# ```python

# --- Parameters (use reduced units for simplicity in MD) ---
sigma = 1.0
epsilon = 1.0
mass = 1.0
kB_reduced = 1.0  # In reduced units, kB is often set to 1.0
temperature = 1.0  # Target temperature in reduced units
gamma = 0.1       # Friction coefficient
dt = 0.005        # Time step

N_atoms = 64
box_size = (N_atoms / 0.8)**(1/3)  # Example: to get a density of ~0.8
# Adjust to place particles reasonably

# --- Initial Conditions ---
# positions = initialize_positions_on_lattice(N_atoms, box_size)
positions = box_size * np.random.rand(N_atoms, 3)  # Random initial positions
velocities = np.sqrt(kB_reduced * temperature / mass) * \
    np.random.normal(size=(N_atoms, 3))
# Center of mass velocity should be zero
velocities -= np.mean(velocities, axis=0)

# --- Simulation Loop ---
num_steps = 10000

# Store energies (optional)
kinetic_energies = []
potential_energies = []
total_energies = []


def calculate_lj_force(positions, sigma, epsilon, box_size):
    # This is a placeholder. A proper implementation needs PBC and cutoff.
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    potential_energy = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            # Apply Periodic Boundary Conditions (simplified)
            r_vec -= box_size * np.round(r_vec / box_size)

            r_sq = np.sum(r_vec**2)
            r = np.sqrt(r_sq)

            # Avoid division by zero if particles are on top of each other
            if r < 0.001:  # Small epsilon to prevent singularity
                continue

            r_inv = 1.0 / r
            r6 = r_inv**6
            r12 = r6**2

            # Lennard-Jones potential energy
            potential_energy += 4 * epsilon * (r12 - r6)

            # Lennard-Jones force (negative gradient of potential)
            # F = 24 * epsilon * (2 * sigma^12 / r^13 - sigma^6 / r^7) * (r_vec / r)
            # Or in terms of r6, r12
            force_magnitude_factor = 24 * epsilon * \
                (2 * r12 * r_inv**2 - r6 * r_inv**2)
            force_ij = force_magnitude_factor * r_vec

            forces[i] += force_ij
            forces[j] -= force_ij  # Newton's third law

    return forces, potential_energy

# Velocity Verlet with Langevin thermostat (simplified)


def update_langevin_vv(positions, velocities, forces, mass, gamma, kB_reduced, temperature, dt, box_size):

    # Calculate coefficients for Langevin thermostat
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt(kB_reduced * temperature / mass * (1 - c1**2))

    # Half-step velocity update (deterministic part)
    velocities_half = velocities + (forces / mass) * (dt / 2.0)

    # Position update
    positions += velocities_half * dt
    # Apply periodic boundary conditions to positions
    positions = positions % box_size

    # Recalculate forces at new positions
    new_forces, current_potential_energy = calculate_lj_force(
        positions, sigma, epsilon, box_size)

    # Add random force and complete velocity update
    # Wi or xi: standard normal random numbers
    random_numbers = np.random.normal(
        loc=0.0, scale=1.0, size=velocities.shape)

    velocities = c1 * velocities_half + c2 * \
        random_numbers + (new_forces / mass) * (dt / 2.0)

    return positions, velocities, new_forces, current_potential_energy


print("\nStarting simulation...")
for step in range(num_steps):
    # Calculate forces at current positions (for the first step or before first half-velocity update)
    if step == 0:
        forces, current_potential_energy = calculate_lj_force(
            positions, sigma, epsilon, box_size)

    # Update positions and velocities using Langevin Velocity Verlet
    positions, velocities, forces, current_potential_energy = \
        update_langevin_vv(positions, velocities, forces,
                           mass, gamma, kB_reduced, temperature, dt, box_size)

    # Calculate energies
    current_kinetic_energy = 0.5 * mass * np.sum(velocities**2)
    current_total_energy = current_kinetic_energy + current_potential_energy

    # Store for analysis
    kinetic_energies.append(current_kinetic_energy)
    potential_energies.append(current_potential_energy)
    total_energies.append(current_total_energy)

    if step % (num_steps // 10) == 0:
        print(f"Step {step}/{num_steps}: KE={current_kinetic_energy:.2f}, PE={current_potential_energy:.2f}, TE={current_total_energy:.2f}")

print("\nSimulation finished.")

# You can then analyze the energy arrays, plot them, etc.
# import matplotlib.pyplot as plt
# plt.plot(kinetic_energies, label='Kinetic Energy')
# plt.plot(potential_energies, label='Potential Energy')
# plt.plot(total_energies, label='Total Energy')
# plt.legend()
# plt.xlabel('Time Step')
# plt.ylabel('Energy')
# plt.show()
