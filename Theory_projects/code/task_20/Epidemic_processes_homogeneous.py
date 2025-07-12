import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
import csv
from scipy.integrate import solve_ivp

# Counts nodes in a given state
def count_states(G, state):
    count = 0
    for node in G.nodes:
        if G.nodes[node]["state"] == state:
            count += 1
    return count


# Custom color palette
state_colors = {
    "S": "slateblue",
    "E": "orangered",
    "I": "firebrick",
    "R": "seagreen"
}

def color_node_state(state):
    return state_colors.get(state, "gray")

# Transition function for the network
# This function simulates the spread of an epidemic on a network using a specified model (SI, SIS, SIR, SEIR).
def transition_network(G, model, params, possible_states, t_steps, plot=True):
    beta = params["infection_rate"]
    infected_history = np.zeros(t_steps)
    new_infections_history = np.zeros(t_steps)
    total_initial_infections = 0

    state_counts = {state: [] for state in possible_states}

    # Prepare to store snapshots
    snapshot_times = [0, t_steps // 2, t_steps - 1] if plot else []
    snapshots = {}

    for t in range(t_steps):
        new_states = {}
        new_infections = 0

        # Count current states
        for state in possible_states:
            state_counts[state].append(count_states(G, state))

        # Save snapshots
        if t in snapshot_times:
            snapshots[t] = {node: G.nodes[node]["state"] for node in G.nodes}

        for node in G.nodes:
            current_state = G.nodes[node]["state"]

            if model != "SEIR":
                # If the current node is susceptible, check for infections
                if current_state == "S":
                    I_neighbors = sum(1 for neighbor in G.neighbors(node) if G.nodes[neighbor]["state"] == "I")
                    p_infected = np.random.rand()
                    if p_infected < 1 - (1 - beta) ** I_neighbors:
                        new_states[node] = "I"
                        new_infections += 1
                        if t == 0:
                            total_initial_infections += 1

            if model == "SIS":
                mu = params["recovery_rate"]
                # If the current node is infected, check for "recovery" = susceptible again
                if current_state == "I":
                    p_recovered = np.random.rand()
                    if p_recovered < mu:
                        new_states[node] = "S"

            elif model == "SIR":
                mu = params["recovery_rate"]
                # If the current node is infected, check for recovery
                if current_state == "I":
                    p_recovered = np.random.rand()
                    if p_recovered < mu:
                        new_states[node] = "R"

            elif model == "SEIR":
                mu = params["recovery_rate"]
                epsilon = params["exposed_rate"]
                if current_state == "S":
                    I_neighbors = sum(1 for neighbor in G.neighbors(node) if G.nodes[neighbor]["state"] == "I")
                    p_exposed = np.random.rand()
                    if p_exposed < 1 - (1 - beta) ** I_neighbors:
                        new_states[node] = "E"
                        new_infections += 1
                        if t == 0:
                            total_initial_infections += 1
                # If the current node is susceptible, check for exposure
                elif current_state == "E":
                    p_infected = np.random.rand()
                    if p_infected < epsilon:
                        new_states[node] = "I"
                # If the current node is infected, check for recovery
                elif current_state == "I":
                    p_recovered = np.random.rand()
                    if p_recovered < mu:
                        new_states[node] = "R"

        for node, new_state in new_states.items():
            G.nodes[node]["state"] = new_state

        infected_history[t] = count_states(G, "I")
        new_infections_history[t] = new_infections

    # Compute R0 and Rt
    R0 = total_initial_infections
    Rt = np.zeros(t_steps)
    for t in range(1, t_steps):
        if infected_history[t - 1] > 0:
            Rt[t] = new_infections_history[t] / infected_history[t - 1]

    # If needed, plot the snapshots of the network at specified times
    if plot:
        pos = nx.spring_layout(G, seed=42)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, t_snap in enumerate(snapshot_times):
            nx.draw(
                G,
                pos,
                ax=axes[i],
                node_color=[color_node_state(snapshots[t_snap][n]) for n in G.nodes],
                with_labels=False,
                node_size=10,
                edge_color="lightgrey",  # Make edges light and less visible
                width=0.5                # Optional: thinner lines
            )
            axes[i].set_title(f"Time step {t_snap}", size= 20)
        plt.tight_layout()
        plt.savefig(f"Homogeneous_snapshots_{model}.png", dpi=300)
        plt.close()

    return state_counts, R0, Rt


# Function to load a homogeneous graph based on the specified model
# Available models: "ER" (Erdős-Rényi), "watts_strogatz" (Watts-Strogatz), "regular_graph" (Random Regular Graph)
def load_graph_homogeneous(graph_name, network_params):
    if graph_name == "ER":
        # All N nodes are connected to all other nodes
        N = network_params["N"]
        p = network_params["p"]
        return nx.erdos_renyi_graph(N, p)
    elif graph_name == "watts_strogatz":
        # All N nodes are connected to k/2 other nodes (Wattz-Strogatz with rewiring probability 0)
        N = network_params["N"]
        k = network_params["k"]
        p = network_params["p"]
        return nx.watts_strogatz_graph(N, k, p)
    elif graph_name == "regular_graph":
        N = network_params["N"]
        k = network_params["k"]
        # All N nodes are connected to k other nodes
        return nx.random_regular_graph(k, N)
    else:
        raise ValueError(f"Unknown graph model: {graph_name}")

# Function to save the edges of the graph to a CSV file
def save_graph_edges(G, graph_name):
    filename = f"Homogeneous_{graph_name}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["node_from", "node_to", "weight"])
        for u, v in G.edges():
            writer.writerow([u, v, 1])

# Function to load the model based on the specified name and parameters
# This function returns a dictionary containing the compartments and parameters for the model.
def load_model(model_name, params):
    if model_name == "SI":
        return {
            "compartments": ["S", "I"],
            "params": params
        }
    if model_name == "SIR":
        return {
            "compartments": ["S", "I", "R"],
            "params": params
        }
    elif model_name == "SIS":
        return {
            "compartments": ["S", "I"],
            "params": params
        }
    elif model_name == "SEIR":
        return {
            "compartments": ["S", "E", "I", "R"],
            "params": params
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Function to scan the epidemic threshold for a range of beta values
# This function runs multiple simulations for each beta value and returns the average final infected or recovered fraction
def scan_threshold(beta_values, G, model_name, fixed_params, network_params, t_steps, runs_per_beta=10):
    results = []
    for beta in beta_values:
        infected_final = []
        for _ in range(runs_per_beta):
            # Reset graph
            for node in G.nodes:
                G.nodes[node]["state"] = "S"
            G.nodes[np.random.choice(list(G.nodes))]["state"] = "I"

            # Update model parameters
            params = fixed_params.copy()
            params["infection_rate"] = beta
            model = load_model(model_name, params)

            # Run simulation
            state_counts, _, _ = transition_network(G, model_name, model["params"], model["compartments"], t_steps, plot=False)

            # Measure final infected or recovered (SIS → I, SIR → R)
            if model_name == "SIS" or model_name == "SI":
                infected = state_counts["I"][-1]
            elif model_name in ("SIR", "SEIR"):
                infected = state_counts["R"][-1]
            infected_final.append(infected / network_params["N"])
        
        avg_final = np.mean(infected_final)
        results.append((beta, avg_final))
    return results


# Model name
infection_model = "SIS"  # Available models: "SI", "SIS", "SIR", "SEIR"
# Model parameters
epidemic_params = {
    "infection_rate": 0.2, # Infection rate for all models
    "recovery_rate": 0.1,  # Recovery rate for SIR, SIS and SEIR models
    "exposed_rate": 0.2  # Exposed rate for SEIR model
}

# Define network parameters
network_params = {
    "N": 10000,
    "p": 0.002,
}
k = network_params["p"] * (network_params["N"] - 1)  # k = p * (N - 1)
graph_name = "ER"  # Change to "watts_strogatz" or "regular_graph" as needed
# Load the graph
G = load_graph_homogeneous(graph_name, network_params)
print(G)

# Save the graph edges to a CSV file
save_graph_edges(G, "ER")

# Model definition
# Model name
infection_model = "SEIR"  # Available models: "SI", "SIS", "SIR", "SEIR"
# Model parameters
epidemic_params = {
    "infection_rate": 0.2, # Infection rate for all models
    "recovery_rate": 0.1,  # Recovery rate for SIR, SIS and SEIR models
    "exposed_rate": 0.2  # Exposed rate for SEIR model
}

# Load the model 
model = load_model(infection_model, epidemic_params)

# Use the transition function to simulate the model
t_steps = 70
time = np.arange(t_steps)
plot = True

# Initialize all nodes with state "S"
for node in G.nodes:
    G.nodes[node]["state"] = "S"

# Infect a random node
G.nodes[np.random.choice(list(G.nodes))]["state"] = "I"
k_average = np.mean([degree for node, degree in G.degree()])
# Run the simulation
state_counts, R0, Rt = transition_network(G, infection_model, model["params"], model["compartments"], t_steps, plot)

# Convert to arrays
# S and I are present in all models, so we compute them always
state_counts_S = np.array(state_counts["S"])
state_counts_I = np.array(state_counts["I"])
time = np.arange(len(state_counts_S))
t_span = (time[0], time[-1])
t_eval = time

# Parameters
i0 = state_counts_I[0] / network_params["N"]
beta = model["params"]["infection_rate"]
N = network_params["N"]

# Prepare figure
plt.figure(figsize=(10, 6))

# Plot simulation
plt.plot(time, state_counts_S / N, label="Susceptible (sim)", color=state_colors["S"])
plt.plot(time, state_counts_I / N, label="Infected (sim)", color=state_colors["I"])

if infection_model == "SIS":
    mu = model["params"]["recovery_rate"]
    # Plot ODE solution for SIS model
    def sis_rhs(t, rho):
        return -mu * rho + beta * k_average * (1 - rho) * rho

    sol = solve_ivp(sis_rhs, t_span, [i0], t_eval=t_eval)
    plt.plot(time, sol.y[0], 'k--', label="Infected (ODE)", linewidth=2)

elif infection_model == "SI":
    # Plot ODE solution for SI model
    def si_rhs(t, rho):
        return beta * k_average * (1 - rho) * rho

    sol = solve_ivp(si_rhs, t_span, [i0], t_eval=t_eval)
    plt.plot(time, sol.y[0], 'k--', label="Infected (ODE)", linewidth=2)

elif infection_model == "SIR":
    mu = model["params"]["recovery_rate"]
    # Plot ODE solution for SIR model
    def sir_rhs(t, y):
        S, I, R = y
        dS = -beta * k_average * S * I
        dI = beta * k_average * S * I - mu * I
        dR = mu * I
        return [dS, dI, dR]

    S0 = 1 - i0
    I0 = i0
    R0 = 0.0
    sol = solve_ivp(sir_rhs, t_span, [S0, I0, R0], t_eval=t_eval)
    plt.plot(time, sol.y[1], 'k--', label="Infected (ODE)", linewidth=2)
    state_counts_R = np.array(state_counts["R"])
    plt.plot(time, state_counts_R / N, label="Recovered (sim)", color=state_colors["R"])

elif infection_model == "SEIR":
    mu = model["params"]["recovery_rate"]
    epsilon = model["params"]["exposed_rate"]
    # Plot ODE solution for SEIR model
    def seir_rhs(t, y):
        S, E, I, R = y
        dS = -beta * k_average * S * I
        dE = beta * k_average * S * I - epsilon * E
        dI = epsilon * E - mu * I
        dR = mu * I
        return [dS, dE, dI, dR]

    S0 = 1 - i0
    E0 = 0.0
    I0 = i0
    R0 = 0.0
    sol = solve_ivp(seir_rhs, t_span, [S0, E0, I0, R0], t_eval=t_eval)
    plt.plot(time, sol.y[2], 'k--', label="Infected (ODE)", linewidth=2)
    state_counts_E = np.array(state_counts["E"])
    state_counts_R = np.array(state_counts["R"])
    plt.plot(time, state_counts_E / N, label="Exposed (sim)", color=state_colors["E"])
    plt.plot(time, state_counts_R / N, label="Recovered (sim)", color=state_colors["R"])

plt.xlabel("Time", size=14)
plt.ylabel("Density of Nodes", size=14)
plt.title(f"{infection_model} model – Simulation vs ODE", size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f"Homogeneous_{infection_model}_densities.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 6))
# Plot effective reproduction number R(t)
plt.plot(time, Rt, label="Effective Reproduction Number $R(t)$", color="darkorchid", linestyle="--", linewidth=2)
# Plot number of infected
plt.plot(time, state_counts_I / network_params["N"], label="Infected", color=state_colors["I"])

plt.xlabel("Time", size=14)
plt.ylabel("Value", size=14)
plt.title(f"{infection_model} model – $R(t)$ and Infected", size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f"Homogeneous_{infection_model}_Rt.png", dpi=300)
plt.close()


mu = model["params"].get("recovery_rate", None)
if mu is not None:
    threshold_theory = mu / k_average
    threshold_sim = beta
    print(f"Theoretical epidemic threshold (β_c = μ / <k>): {threshold_theory:.4f}")
    print(f"Simulation β: {threshold_sim:.4f}")
    if beta > threshold_theory:
        print("→ The infection spreads: β > β_c (above threshold)")
    else:
        print("→ The infection dies out: β < β_c (below threshold)")
else:
    print("Threshold not defined for models without recovery (e.g., SI).")


beta_range = np.linspace(0.001, 0.01, 25)
if infection_model != "SI": 
    results = scan_threshold(beta_range, G, infection_model, epidemic_params, network_params, t_steps)

# Plot
if infection_model != "SI":
    betas, infected_fractions = zip(*results)
    plt.plot(betas, infected_fractions, marker='o', color='darkorchid')
    if infection_model != "SI" and infection_model != "SEIR":
        beta_crit = epidemic_params['recovery_rate']/k_average
        print(beta_crit)
        plt.axvline(x= beta_crit, color='gray', linestyle='--', label='threshold')
    plt.xlabel("Infection rate β", size= 16)
    plt.ylabel("Final infected/recovered fraction", size= 16)
    plt.title(f"Epidemic Threshold Scan for {infection_model}", size= 20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Homogeneous_threshold_evaluation_{infection_model}.png", dpi=300)
    plt.close()

