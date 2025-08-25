import numpy as np

# --- 1. Define Global Constants and Simulation Parameters ---

# Conversion factors for readability/user input (Angstroms to meters)
ANGSTROM_TO_METER = 1e-10  # 1 Angstrom = 10^-10 meters

# Physical Constants (SI Units)
KB = 1.380649e-23     # Boltzmann constant (J/K)
PO_MASS_KG = 3.47e-25  # Polonium atom mass (kg)

# Lennard-Jones Parameters for Po-Po interaction (in base Angstroms for user, converted internally)
PO_LJ_EPSILON_JOULES = 2.4e-20  # Joules (energy depth)
PO_LJ_SIGMA_ANGSTROM = 2.37   # Angstroms (effective diameter)

# Simulation Parameters (user-friendly units where applicable)
N_ATOMS = 10                  # Number of Po atoms
BOX_SIZE_ANGSTROM = 20.0      # Simulation box size (Angstroms)
TEMPERATURE_K = 300           # Target Temperature (Kelvin)
# Damping coefficient (kg/s) - Adjusted to 1e-12 as you tried
GAMMA_DAMPING = 1e-12
DT_SECONDS = 2e-15            # Timestep (seconds)
ITERATIONS = 10000            # Total simulation steps

# Initialization specific parameters
# Minimum initial distance between atoms (Angstroms)
MIN_INIT_DISTANCE_ANGSTROM = 2.5
MAX_INIT_ATTEMPTS = 100       # Retry limit for placing atoms randomly

# Lennard-Jones Cutoff (for efficiency, in Angstroms)
LJ_CUTOFF_ANGSTROM = 3.0 * PO_LJ_SIGMA_ANGSTROM  # Standard practice for LJ

# Dimensions of the simulation
DIMENSIONS = 3

# Define strategy-dependent epsilon values for the game
epsilon_params = {
    'AA': 2.4e-20,  # e.g., default strong bond
    'BB': 1.8e-20,  # e.g., a weaker bond
    'AB': 0.5e-20   # e.g., a repulsive or very weak bond
}

# --- 2. PoloniumAtom Class Definition ---


class PoloniumAtom:
    def __init__(self, id, position, velocity, atom_type='Po'):
        self.id = id
        self.strategy = 'A'
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.type = atom_type
        self.mass = PO_MASS_KG

        self.epsilon = PO_LJ_EPSILON_JOULES
        self.sigma = PO_LJ_SIGMA_ANGSTROM * ANGSTROM_TO_METER

        self.bonds = []
        self.neighbors = set()
        self.state = 'free'
        self.energy = 0.0
        self.payoff_matrix = {}

        # Store the total force acting on this atom
        self.force = np.zeros(DIMENSIONS)

# --- 3. Centralized Force Calculation Function ---


def calculate_all_forces(all_atoms, box_size_m, lj_cutoff_m, epsilon_params):
    num_atoms = len(all_atoms)

    current_forces = {atom.id: np.zeros(DIMENSIONS) for atom in all_atoms}
    total_potential_energy = 0.0

    for i in range(num_atoms):
        atom_i = all_atoms[i]
        for j in range(i + 1, num_atoms):
            atom_j = all_atoms[j]

            r_vec = atom_i.position - atom_j.position
            r_vec -= box_size_m * np.round(r_vec / box_size_m)  # MIC

            r_sq = np.sum(r_vec**2)
            r = np.sqrt(r_sq)

            if r < 1e-12:  # Avoid division by zero
                continue

            if r < lj_cutoff_m:
                if atom_i.strategy == 'A' and atom_j.strategy == 'A':
                    epsilon_pair = epsilon_params['AA']
                elif atom_i.strategy == 'B' and atom_j.strategy == 'B':
                    epsilon_pair = epsilon_params['BB']
                else:  # i.e., AB or BA
                    epsilon_pair = epsilon_params['AB']
                sigma_pair = PO_LJ_SIGMA_ANGSTROM * ANGSTROM_TO_METER

                sigma_over_r = sigma_pair / r
                sr6 = sigma_over_r**6
                sr12 = sr6**2

                total_potential_energy += 4 * epsilon_pair * (sr12 - sr6)

                force_magnitude_factor = 24 * \
                    epsilon_pair * (2 * sr12 - sr6) / r_sq
                force_ij = force_magnitude_factor * r_vec

                current_forces[atom_i.id] += force_ij
                current_forces[atom_j.id] -= force_ij

    return current_forces, total_potential_energy

# --- 4. Main Simulation Execution ---


def initialize_atoms(num_atoms, box_size_m, min_init_distance_m, total_system_mass_kg, temperature_k):
    atoms = []
    all_initial_positions_m = []

    print(f"Initializing {num_atoms} atoms...")
    for i in range(num_atoms):
        placed = False
        for attempt in range(MAX_INIT_ATTEMPTS):
            pos_candidate_m = np.random.uniform(
                min_init_distance_m, box_size_m - min_init_distance_m, DIMENSIONS)
            overlap = any(
                np.linalg.norm(pos_candidate_m -
                               existing_pos_m) < min_init_distance_m
                for existing_pos_m in all_initial_positions_m
            )
            if not overlap:
                velocity_m_per_s = np.random.normal(0, np.sqrt(
                    KB * temperature_k / PO_MASS_KG), DIMENSIONS)
                atoms.append(PoloniumAtom(
                    i, pos_candidate_m, velocity_m_per_s))
                all_initial_positions_m.append(pos_candidate_m)
                placed = True
                break
        if not placed:
            raise ValueError("Failed to place atoms without overlap.")

    # Remove Center of Mass Velocity
    total_momentum_m_per_s = sum(atom.mass * atom.velocity for atom in atoms)
    center_of_mass_velocity_m_per_s = total_momentum_m_per_s / total_system_mass_kg
    for atom in atoms:
        atom.velocity -= center_of_mass_velocity_m_per_s

    return atoms


def simulate_md(atoms, epsilon_params, box_size_m, lj_cutoff_m, ITERATIONS):
    # box_size_m = BOX_SIZE_ANGSTROM * ANGSTROM_TO_METER
    # lj_cutoff_m = LJ_CUTOFF_ANGSTROM * ANGSTROM_TO_METER
    # min_init_distance_m = MIN_INIT_DISTANCE_ANGSTROM * ANGSTROM_TO_METER

    # atoms = []
    # all_initial_positions_m = []

    # print(
    #     f"Initializing {N_ATOMS} atoms in a {BOX_SIZE_ANGSTROM} Å cubic box...")
    # for i in range(N_ATOMS):
    #     placed = False
    #     for attempt in range(MAX_INIT_ATTEMPTS):
    #         pos_candidate_m = np.random.uniform(
    #             min_init_distance_m, box_size_m - min_init_distance_m, DIMENSIONS)

    #         overlap = False
    #         for existing_pos_m in all_initial_positions_m:
    #             distance_m = np.linalg.norm(pos_candidate_m - existing_pos_m)
    #             if distance_m < min_init_distance_m:
    #                 overlap = True
    #                 break

    #         if not overlap:
    #             velocity_m_per_s = np.random.normal(0, np.sqrt(
    #                 KB * TEMPERATURE_K / PO_MASS_KG), DIMENSIONS)

    #             atoms.append(PoloniumAtom(
    #                 i, pos_candidate_m, velocity_m_per_s))
    #             all_initial_positions_m.append(pos_candidate_m)
    #             placed = True
    #             break

    #     if not placed:
    #         raise ValueError(
    #             f"Failed to place atom {i} without overlap after {MAX_INIT_ATTEMPTS} attempts. "
    #             "Consider a lattice initialization, a larger box, or fewer atoms."
    #         )
    # print("All atoms randomly placed (checked for initial overlap).")

    # --- Remove Center of Mass Velocity (Initial) ---
    total_momentum_m_per_s = np.zeros(DIMENSIONS)
    total_system_mass_kg = 0.0
    for atom in atoms:
        total_momentum_m_per_s += atom.mass * atom.velocity
        total_system_mass_kg += atom.mass

    center_of_mass_velocity_m_per_s = total_momentum_m_per_s / total_system_mass_kg

    for atom in atoms:
        atom.velocity -= center_of_mass_velocity_m_per_s

    print(
        f"Initial center of mass velocity (before correction): {total_momentum_m_per_s / total_system_mass_kg} m/s")
    print(
        f"Final center of mass velocity: {np.mean([atom.velocity for atom in atoms], axis=0)} m/s (should be near zero)")

    print("\nStarting simulation...")

    kinetic_energies = []
    potential_energies = []
    total_energies = []
    temperatures = []

    # Initial force calculation (at t=0)
    current_forces_dict, current_potential_energy = calculate_all_forces(
        atoms, box_size_m, lj_cutoff_m, epsilon_params)
    for atom in atoms:
        atom.force = current_forces_dict[atom.id]

    for step in range(ITERATIONS):
        # --- Euler-Langevin Dynamics Integration (Corrected) ---

        # 1. Generate all random velocity changes and correct them for zero net momentum
        all_random_velocity_changes = []
        for atom in atoms:
            random_velocity_kick_std = np.sqrt(
                2 * GAMMA_DAMPING * KB * TEMPERATURE_K * DT_SECONDS) / atom.mass
            random_velocity_change = random_velocity_kick_std * \
                np.random.normal(loc=0.0, scale=1.0, size=DIMENSIONS)
            all_random_velocity_changes.append(random_velocity_change)

        # Calculate the total momentum introduced by these random changes
        total_random_momentum_this_step = np.zeros(DIMENSIONS)
        for i, atom in enumerate(atoms):
            total_random_momentum_this_step += atom.mass * \
                all_random_velocity_changes[i]

        # Distribute this momentum correction among atoms
        # This ensures sum(m_i * dv_random_i) = 0 for the entire system at this step
        # Corrected random change = raw_random_change - (total_random_momentum / total_mass_system)
        for i, atom in enumerate(atoms):
            all_random_velocity_changes[i] -= (
                total_random_momentum_this_step / total_system_mass_kg)

        # 2. Update velocities (LJ Force + Friction + Corrected Random Kick)
        for i, atom in enumerate(atoms):
            friction_force = -GAMMA_DAMPING * atom.velocity

            # This is the Euler update for velocity: v(t+dt) = v(t) + (F_total/m)*dt
            atom.velocity += (atom.force + friction_force) / \
                atom.mass * DT_SECONDS  # Deterministic forces
            # Add the corrected random kick
            atom.velocity += all_random_velocity_changes[i]

        # 3. Update positions
        for atom in atoms:
            atom.position += atom.velocity * DT_SECONDS
            atom.position = atom.position % box_size_m  # Apply Periodic Boundary Conditions

        # 4. Recalculate forces for the new positions
        current_forces_dict, current_potential_energy = calculate_all_forces(
            atoms, box_size_m, lj_cutoff_m, epsilon_params)
        for atom in atoms:
            atom.force = current_forces_dict[atom.id]

        # --- Energy and Temperature Calculation ---
        current_kinetic_energy = 0.5 * \
            sum(atom.mass * np.sum(atom.velocity**2) for atom in atoms)
        current_total_energy = current_kinetic_energy + current_potential_energy
        current_temp_instantaneous = (
            2 * current_kinetic_energy) / (DIMENSIONS * N_ATOMS * KB)

        kinetic_energies.append(current_kinetic_energy)
        potential_energies.append(current_potential_energy)
        total_energies.append(current_total_energy)
        temperatures.append(current_temp_instantaneous)

        # --- Output / Monitoring ---
        if step % (ITERATIONS // 10) == 0 or step == ITERATIONS - 1:
            pos_display_0 = atoms[0].position / ANGSTROM_TO_METER
            vel_display_0 = atoms[0].velocity

            # Recalculate avg system vel for printing to confirm it's near zero
            avg_vel_check = np.mean([a.velocity for a in atoms], axis=0)

            print(
                f"--- Step {step}/{ITERATIONS} (Time: {step * DT_SECONDS:.2e} s) ---")
            print(f"Atom 0 Pos: {pos_display_0} Å, Vel: {vel_display_0} m/s")
            print(
                f"System KE: {current_kinetic_energy:.2e} J, PE: {current_potential_energy:.2e} J, TE: {current_total_energy:.2e} J")
            print(
                f"Instantaneous Temp: {current_temp_instantaneous:.2f} K (Target: {TEMPERATURE_K} K)")
            print(
                f"Avg System Pos: {np.mean([a.position for a in atoms], axis=0) / ANGSTROM_TO_METER} Å, Avg System Vel: {avg_vel_check} m/s (should be near zero)")

    print("\nSimulation Finished.")
    return atoms, kinetic_energies, potential_energies, total_energies, temperatures

    # --- Post-Simulation Analysis (Optional) ---


def calculate_rdf_and_r_star(atoms, box_size_m, num_bins=100):
    positions = np.array([atom.position for atom in atoms])
    num_atoms = len(positions)

    # Calculate histogram of distances
    bins = np.linspace(0, box_size_m / 2, num_bins)
    distances = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            r_vec = positions[i] - positions[j]
            r_vec -= box_size_m * np.round(r_vec / box_size_m)
            distances.append(np.linalg.norm(r_vec))

    hist, _ = np.histogram(distances, bins=bins, density=False)

    # Normalize the RDF
    dr = bins[1] - bins[0]
    volume = box_size_m**3
    rho = num_atoms / volume

    rdf = np.zeros(num_bins - 1)
    for i in range(num_bins - 1):
        r = (bins[i] + bins[i+1]) / 2
        # Ideal gas reference density
        ideal_gas_ref = 4 * np.pi * r**2 * dr * rho * num_atoms
        if ideal_gas_ref > 0:
            rdf[i] = hist[i] / ideal_gas_ref

    # Find the position of the first peak
    peak_idx = np.argmax(rdf)
    r_star = (bins[peak_idx] + bins[peak_idx+1]) / 2

    return rdf, bins[:-1], r_star

# Function to calculate LJ potential energy for a given distance and epsilon


def plot_rdf(rdf, bins, title):
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        # The 'bins' array is one element longer than 'rdf', so we plot against the bin centers
        plt.plot(bins, rdf, label='g(r)')
        plt.xlabel('Distance r (m)')
        plt.ylabel('g(r)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    except ImportError:
        print("Matplotlib not found. Skipping RDF plot.")


def calculate_lj_potential(r, epsilon, sigma):
    if r < 1e-12:
        return np.inf
    sigma_over_r = sigma / r
    sr6 = sigma_over_r**6
    sr12 = sr6**2
    return 4 * epsilon * (sr12 - sr6)


def solve_game(r_star, epsilon_params, sigma_m):
    # Define strategy-dependent epsilon values
    epsilon_AA = epsilon_params['AA']   # From your original code
    epsilon_BB = epsilon_params['BB']  # Example: a weaker bond
    # Example: a very weak, nearly repulsive bond
    epsilon_AB = epsilon_params['AB']

    # Use r_star from the previous step to define the payoff matrix
    # We assume P_AB = P_BA for simplicity
    U_AA = calculate_lj_potential(
        r_star, epsilon_AA, sigma_m)
    U_BB = calculate_lj_potential(
        r_star, epsilon_BB, sigma_m)
    U_AB = calculate_lj_potential(
        r_star, epsilon_AB, sigma_m)

    P_AA = -0.5 * U_AA
    P_BB = -0.5 * U_BB
    P_AB = -0.5 * U_AB
    P_BA = P_AB

    # Solve for the mixed strategy Nash Equilibrium
    # We need to find p_A where expected payoffs are equal:
    # p_A * P_AA + (1 - p_A) * P_AB = p_A * P_BA + (1 - p_A) * P_BB
    # p_A * (P_AA - P_AB - P_BA + P_BB) = P_BB - P_AB
    denominator = (P_AA - P_AB - P_BA + P_BB)
    if abs(denominator) < 1e-15:  # Avoid division by zero
        p_A_star = 0.5  # Or handle based on game theory cases
    else:
        p_A_star = (P_BB - P_AB) / denominator
        # Ensure probability is between 0 and 1
        p_A_star = np.clip(p_A_star, 0.0, 1.0)

    print(
        f"The solved mixed-strategy Nash equilibrium p_A* is: {p_A_star:.4f}")
    return p_A_star


# --- New Main Execution Block for Experiment 1 ---
if __name__ == "__main__":
    box_size_m = BOX_SIZE_ANGSTROM * ANGSTROM_TO_METER
    lj_cutoff_m = LJ_CUTOFF_ANGSTROM * ANGSTROM_TO_METER
    min_init_distance_m = MIN_INIT_DISTANCE_ANGSTROM * ANGSTROM_TO_METER
    total_system_mass_kg = N_ATOMS * PO_MASS_KG

    # Define strategy-dependent epsilon values for the game
    epsilon_params = {
        'AA': 2.4e-20,  # e.g., default strong bond
        'BB': 1.8e-20,  # e.g., a weaker bond
        'AB': 0.5e-20   # e.g., a repulsive or very weak bond
    }

    # =========================================================
    # PART 1: Run Pure MD Simulation to find r*
    # =========================================================
    print("--- Running Pure MD Simulation (All 'A' Strategy) ---")
    atoms_pure_md = initialize_atoms(
        N_ATOMS, box_size_m, min_init_distance_m, total_system_mass_kg, TEMPERATURE_K)
    # (Your existing code to initialize N_ATOMS for pure MD goes here)
    # Be sure to set atom.strategy = 'A' for all of them

    # ...
    # Initialize the atoms list and remove CoM velocity, etc.
    # ...

    atoms_pure_md, ke_pure, pe_pure, te_pure, temp_pure = simulate_md(
        atoms_pure_md,
        {'AA': epsilon_params['AA'], 'BB': 0.0,
            'AB': 0.0},  # Use only epsilon_AA
        box_size_m, lj_cutoff_m, ITERATIONS
    )

    # Calculate r* from the final state of the Pure MD run
    # This function needs to be defined
    rdf_pure, bins_pure, r_star_pure = calculate_rdf_and_r_star(
        atoms_pure_md, box_size_m)
    print(f"Equilibrated mean neighbor distance (r*): {r_star_pure:.4e} m")
    plot_rdf(rdf_pure, bins_pure, "RDF for Pure MD Simulation")
    # _, _, r_star = calculate_rdf_and_r_star(atoms_pure_md, box_size_m)
    # print(f"Equilibrated mean neighbor distance (r*): {r_star:.4e} m")

    # =========================================================
    # PART 2: Solve the Game
    # =========================================================
    print("\n--- Solving for Mixed-Strategy Nash Equilibrium ---")
    sigma_m = PO_LJ_SIGMA_ANGSTROM * ANGSTROM_TO_METER
    p_A_star = solve_game(r_star_pure, epsilon_params, sigma_m)
    print(
        f"The solved mixed-strategy Nash equilibrium p_A* is: {p_A_star:.4f}")

    # =========================================================
    # PART 3: Run Game-Theoretic MD
    # =========================================================
    print("\n--- Running Game-Theoretic MD Simulation ---")
    atoms_gt_md = initialize_atoms(
        N_ATOMS, box_size_m, min_init_distance_m, total_system_mass_kg, TEMPERATURE_K)
    # (Your existing code to initialize N_ATOMS goes here)
    # This time, assign strategies based on p_A_star:
    for i in range(N_ATOMS):
        if np.random.rand() < p_A_star:
            atoms_gt_md[i].strategy = 'A'
        else:
            atoms_gt_md[i].strategy = 'B'

    # ...
    # Remove CoM velocity for this new set of atoms
    # ...

    atoms_gt_md, ke_gt, pe_gt, te_gt, temp_gt = simulate_md(
        atoms_gt_md,
        epsilon_params,  # Use the full epsilon matrix now
        box_size_m, lj_cutoff_m, ITERATIONS
    )

    # =========================================================
    # PART 4: Analysis and Comparison
    # =========================================================
    print("\n--- Experiment 1 Complete. Comparing Results ---")
    # You will need to write the code to compare the metrics
    # e.g., plot the energy and temperature evolutions on the same graph
    # and compare the final RDFs

    # (Your existing plotting code can be adapted here to plot both sets of data)

    # Example comparison:
    avg_pe_pure = np.mean(pe_pure[-ITERATIONS//10:])
    avg_pe_gt = np.mean(pe_gt[-ITERATIONS//10:])
    print(f"Final Average Potential Energy (Pure MD): {avg_pe_pure:.2e} J")
    print(f"Final Average Potential Energy (GT MD): {avg_pe_gt:.2e} J")

    rdf_pure, _, _ = calculate_rdf_and_r_star(atoms_pure_md, box_size_m)
    rdf_gt, _, _ = calculate_rdf_and_r_star(atoms_gt_md, box_size_m)
    # ... (code to plot both RDFs on the same graph) ...
    # ... (after both simulate_md calls have completed) ...

    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # --- Plot 1: Energy Evolution Comparison ---
        # Plotting for the Pure MD run
        axs[0].plot(np.arange(ITERATIONS) * DT_SECONDS,
                    np.array(pe_pure) * 1e20, label='Pure MD Potential Energy')
        # Plotting for the Game-Theoretic MD run
        axs[0].plot(np.arange(ITERATIONS) * DT_SECONDS,
                    np.array(pe_gt) * 1e20, label='GT MD Potential Energy')

        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Energy (J)')
        axs[0].set_title('Potential Energy Evolution Comparison')
        axs[0].legend()
        axs[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # ... (You would similarly modify the other plots) ...

        # --- Plot 2: Temperature Evolution Comparison ---
        axs[1].plot(np.arange(ITERATIONS) * DT_SECONDS,
                    temp_pure, label='Pure MD Temperature')
        axs[1].plot(np.arange(ITERATIONS) * DT_SECONDS,
                    temp_gt, label='GT MD Temperature')
        axs[1].axhline(y=TEMPERATURE_K, color='r',
                       linestyle='--', label='Target Temperature')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Temperature (K)')
        axs[1].set_title('Temperature Evolution Comparison')
        axs[1].legend()
        axs[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # --- Plot 3: Final Atom Positions Comparison ---
        # (This is trickier to plot directly on one graph, but you could do two separate scatter plots)
        final_positions_pure = np.array(
            [a.position for a in atoms_pure_md]) / ANGSTROM_TO_METER
        final_positions_gt = np.array(
            [a.position for a in atoms_gt_md]) / ANGSTROM_TO_METER

        axs[2].scatter(final_positions_pure[:, 0], final_positions_pure[:,
                       1], s=50, c='blue', label='Pure MD Final Positions')
        # Use a different color or marker for the game-theoretic run if you wish
        axs[2].scatter(final_positions_gt[:, 0], final_positions_gt[:, 1],
                       s=50, c='red', alpha=0.5, label='GT MD Final Positions')

        axs[2].set_xlim(0, BOX_SIZE_ANGSTROM)
        axs[2].set_ylim(0, BOX_SIZE_ANGSTROM)
        axs[2].set_xlabel('X Position (Å)')
        axs[2].set_ylabel('Y Position (Å)')
        axs[2].set_title('Final Atom Positions (XY Plane)')
        axs[2].set_aspect('equal', adjustable='box')
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not found. Skipping plot generation.")
