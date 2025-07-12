import pandas as pd
import numpy as np 
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import haversine
from scipy.optimize import curve_fit
import seaborn as sns

# Import the county-county file, already transformed from .tsv file to .csv file
file = 'county_county.csv'

# Specify the columns that should be read as strings (fips codes contain leading zeros)
dtype = {
    'user_loc': str,
    'fr_loc': str
}

county_county = pd.read_csv(file, dtype=dtype)

# Save two new columns with the first 2 digits of FIPS code for the origin and destination countries
county_county['country_origin'] = county_county['user_loc'].str[:2]
county_county['country_dest'] = county_county['fr_loc'].str[:2]

# Save two new columns with the remaininder of the 5 digits FIPS code for the origin and destination counties
county_county['county_origin'] = county_county['user_loc'].apply(lambda x: str(x)[2:])
county_county['county_dest'] = county_county['fr_loc'].apply(lambda x: str(x)[2:])

# Only keep rows where the origin and destination countries are the same
county_county = county_county[county_county['country_origin'] == county_county['country_dest']]

# Filter the dataset to obtain only the rows where the origin and destination counties are different 
# we are not interested in the same county SCI flows
county_county = county_county[county_county['county_origin'] != county_county['county_dest']]

# Create one ebunch (An iterable container of edge tuples like a list, iterator, or file.) for each unique county 
# The ebunch should be in the form (county_origin, county_dest, scaled_sci)
ebunch = []

# Group the data by the origin country
grouped = county_county.groupby('country_origin')

# Iterate through groups
for country, group in grouped:
    ebunch_country = [(row['county_origin'], row['county_dest'], {'weight': row['scaled_sci']}) for index, row in group.iterrows()]
    ebunch.append(ebunch_country)

# Save the unique country codes to keep track of the countries in the network
country_code = np.unique(county_county['country_origin'])

# This is a dataset obtained by the manipulation and integration of two different datasets
# The first dataset contained informations about country and county names and their respective FIPS codes
# The second dataset contained informations about the latitude and longitude of each county
# Many points needed to be added manually since county subdivision changed since 2012 and the county-county dataset was from 2012

path = "county_latlng.csv"
Fips = pd.read_csv(path, dtype=str)

# Now we create one graph for each country
# Dictionaries were chosen to handle all the graphs and further analysis, since they make easier to access each country individually 
graphs_dict = {}

for i in range(len(country_code)):
    G=nx.Graph()
    # Edges are added from the ebunch already created and are weighted by the scaled_sci
    G.add_edges_from(ebunch[i])
    # country_code[i] is a string with the first two values of the fips code that identify the state e.g. '01' for Alabama
    state_names = country_code[i]
    G_name = state_names 
    graphs_dict[G_name] = G

import pandas as pd

# Create nodeID as a concatenation of state and county FIPS codes = full fips code 
# this is not needed for the nodes but for how edges are defined in the ebunch and consequently in the graphs
Fips["nodeID"] = Fips["StateFIPS"].astype(str) + Fips["CountyFIPS_3"].astype(str)

# Aggregate all nodes from the different graphs
all_nodes = {state + node for state, G in graphs_dict.items() for node in G.nodes()}

# Assign new numeric nodeID
Fips["new_nodeID"] = range(1, len(Fips) + 1)

# Construct and save the nodes DataFrame
# The structure is nodeID = n starting from 1, nodeLabel = CountyName, latitude and longitude
nodes = Fips[["new_nodeID", "CountyName", "lat", "lng"]]
nodes.columns = ["nodeID", "nodeLabel", "latitude", "longitude"]
nodes.to_csv("nodes.csv", index=False)

# Create mapping from old nodeID to new numeric nodeID
nodeID_map = dict(zip(Fips["nodeID"], Fips["new_nodeID"]))

# Aggregate edges and map to new nodeIDs
all_edges = [
    {"nodeID_from": nodeID_map[state + u], "nodeID_to": nodeID_map[state + v], "state" : state}
    for state, G in graphs_dict.items() for u, v in G.edges()
]

# Save edges DataFrame
edges = pd.DataFrame(all_edges)
edges.to_csv("edges.csv", index=False)

# Drop nodeID and new_nodeID columns
Fips.drop(columns=["nodeID", "new_nodeID"], inplace=True)

# Load the world counties shapefile
world = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip")

# Create `pos` dictionary with StateFIPS as key and CountyFIPS_3 as node names
pos = {}
for _, row in Fips.iterrows():
    state_fips = row['StateFIPS']
    county_fips = row['CountyFIPS_3']
    coordinates = (float(row['lng']), float(row['lat']))  # Convert to for plotting reasons 
    if state_fips not in pos:
        pos[state_fips] = {}
    pos[state_fips][county_fips] = coordinates

# Function to plot the network over the state map
def plot_state_network(state_fips):

    # Get state name
    state_name = Fips[Fips['StateFIPS'] == state_fips]['StateName'].iloc[0]
    
    # Filter the shapefile for the given state
    usa = world[world['STATEFP'] == state_fips]
    
    # Get the full graph for the state
    G_state = graphs_dict[state_fips].copy()
    
    # Get node positions and set up colors
    nodes = list(G_state.nodes())
    # This will be needed to assign colors to the nodes
    n_nodes = len(nodes)
    filtered_pos = pos[state_fips] # Filter the positions for the state
    
    # Chose 'gist_rainbow' colormap for visualization porpuses (other colormpas contained too light or too dark colors that were not visible) 
    cmap = plt.colormaps.get_cmap('gist_rainbow')
    # Normalize indices to range [0,1] for full color span
    node_colors = [cmap(i / (n_nodes - 1)) for i in range(n_nodes)]
    node_color_dict = dict(zip(nodes, node_colors))
    
    # Define edge properties: color based on origin and thickness based on SCI
    edge_colors = []
    edge_widths = []
    for u, _, data in G_state.edges(data=True):
        origin_node = data.get('gist_rainbow', u)  # Default to u if 'gist_rainbow' not present
        weight = data.get('weight', 1)
        edge_colors.append(node_color_dict.get(origin_node, "gray"))
        # Scale edge width based on max SCI value for that state
        max_sci = max(county_county[county_county['country_origin'] == state_fips]['scaled_sci'])
        edge_widths.append((weight / max_sci) * 50)
    
    # Plot the network over the state map
    fig, ax = plt.subplots(figsize=(10, 7))
    usa.plot(ax=ax, color='lightgray', edgecolor='black')
    
    nx.draw_networkx_nodes(G_state, pos=filtered_pos, ax=ax, node_size=100,
                           node_color=[node_color_dict[node] for node in G_state.nodes()])
    nx.draw_networkx_edges(G_state, pos=filtered_pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=1)
    nx.draw_networkx_labels(G_state, pos=filtered_pos, ax=ax)
    
    plt.title(f"County Friendship Network in {state_name}")
    
    # Save the plot
    output_folder = "Maps"
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"network_{state_name}.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Loop through all unique state FIPS codes and generate maps
for state_fips in Fips['StateFIPS'].unique():
    if state_fips != '11' and state_fips != '66' and state_fips != '69':  # Ensure the maps with only one or two nodes are excluded 
        plot_state_network(state_fips)

print(f"Saved networks map to: 'Maps' folder")

# Strength distribution of the networks
strength_by_state = {}
state_strength_summary = {}

# Create the "Strength" folder if it doesn't exist
output_folder = "Strength"
os.makedirs(output_folder, exist_ok=True)

for state, G in graphs_dict.items():

    # Get state name
    state_name = Fips[Fips['StateFIPS'] == state]['StateName'].iloc[0]

    # Get number of nodes 
    number_of_nodes = len(G.nodes())

    # Filter only states with more than 10 nodes
    if number_of_nodes > 10:
        
        strength = []
        
        for node in G.nodes():
            node_strength = 0
            for u, v, data in G.edges(data=True):
                if u == node:
                    node_strength += data['weight']
            strength.append(node_strength)
        
        # Store the strength values for the state, the mean and standard deviation
        strength_by_state[state] = strength
        mean_strength = np.mean(strength)
        std_strength = np.std(strength)

        # Store and print results
        state_strength_summary[state] = {
            'number of nodes': number_of_nodes,
            'mean_strength': mean_strength,
            'std_strength': std_strength
        }
        

# Save the strength distributions to a CSV file
Strength = pd.DataFrame(state_strength_summary).T
csv_path = os.path.join(output_folder, "strength_distributions.csv")
Strength.to_csv(csv_path)

# Save all the plots as JPG files
for state, strength in strength_by_state.items():

    # Get state name
    state_name = Fips[Fips['StateFIPS'] == state]['StateName'].iloc[0]
    
    plt.figure(figsize=(8, 5))
    plt.hist(strength, bins=30, color='darkorchid', edgecolor='indigo')
    plt.xlabel("Node Strength")
    plt.ylabel("Frequency")
    plt.title(f"Strength Distribution for {state_name}")
    
    plt.axvline(state_strength_summary[state]['mean_strength'], color='crimson', linestyle='dashed', linewidth=2,
                label=f"Mean: {state_strength_summary[state]['mean_strength']:.2e}")
    plt.legend()
    
    # Define the save path inside the Strength folder
    save_path = os.path.join(output_folder, f"strength_distribution_{state}.jpg")
    plt.savefig(save_path)
    plt.close()

print(f"All files saved in '{output_folder}' folder.")


# Dictionary to store centrality measures
centrality_by_state = {}

# Create a folder to store the plots
output_folder = "Strength_Centrality"
os.makedirs(output_folder, exist_ok=True)  # Ensure folder exists

for state, G in graphs_dict.items():

    # Get state name
    state_name = Fips[Fips['StateFIPS'] == state]['StateName'].iloc[0]

    # Get number of nodes
    number_of_nodes = len(G.nodes())

    # Filter only states with more than 10 nodes
    if number_of_nodes > 10:

        # Compute strength centrality given that I already have the strength of each node
        strength = strength_by_state[state]
        max_strength = max(strength) if max(strength) > 0 else 1  # Avoid division by zero
        
        # Compute normalized strength centrality
        strength_centrality = [s / max_strength for s in strength]

        # Store results
        centrality_by_state[state] = {'strength': strength_centrality}
 


# Convert dictionary into a DataFrame, handling list values correctly
Centrality = pd.DataFrame({state: pd.Series(data['strength']) for state, data in centrality_by_state.items()}).T
csv_path = os.path.join(output_folder, "strength_centrality.csv")
Centrality.to_csv(csv_path, index=True)

print(f"Saved all strength centrality distributions in '{output_folder}' folder.")

def plot_state_network_with_strength_centrality(state_fips):

    # Get state name
    state_name = Fips[Fips['StateFIPS'] == state_fips]['StateName'].iloc[0]
    
    # Filter the shapefile for the given state
    usa = world[world['STATEFP'] == state_fips]
    
    # Get the full graph for the state
    G_state = graphs_dict[state_fips].copy()
    
    # Get node positions and set up colors
    nodes = list(G_state.nodes())
    filtered_pos = pos[state_fips]  # Filter the positions for the state
    
    number_of_nodes = len(nodes)
    if number_of_nodes < 10:
        return
    
    # Compute strength centrality for the state
    strength_centrality = strength_by_state[state_fips]  # This should be precomputed for each state
    max_strength = max(strength_centrality) if max(strength_centrality) > 0 else 1  # Avoid division by zero
    normalized_strength = [s / max_strength for s in strength_centrality]
    
    # Scale node sizes based on normalized strength centrality (you can adjust the scaling factor)
    node_sizes = [1000 * s for s in normalized_strength]
    
    # Plot the network over the state map (no edges, just nodes)
    fig, ax = plt.subplots(figsize=(10, 7))
    usa.plot(ax=ax, color='lightgray', edgecolor='black')

    # Draw nodes with size proportional to strength centrality
    nx.draw_networkx_nodes(G_state, pos=filtered_pos, ax=ax, node_size=node_sizes, node_color='darkorchid')

    # Optionally, add labels if needed
    nx.draw_networkx_labels(G_state, pos=filtered_pos, ax=ax, font_size=8)
    
    plt.title(f"Strength Centrality Network for {state_name}")

    # Save the plot
    output_folder = "Strength_Centrality"
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"strength_centrality_{state_name}.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Loop through all unique state FIPS codes and generate maps
for state_fips in Fips['StateFIPS'].unique():
    if state_fips != '11' and state_fips != '66' and state_fips != '69' and state_fips != '33':  # Exclude maps with too few nodes
        plot_state_network_with_strength_centrality(state_fips)

print(f"Saved strength centrality maps to: 'Strength_Centrality' folder")


# Dictionary to store modularity measures
modularity_by_state = {}

# Create a folder to store the plots
output_folder = "Modularity"
os.makedirs(output_folder, exist_ok=True)  # Ensure folder exists

for state, G in graphs_dict.items():

    # Get state name
    state_name = Fips[Fips['StateFIPS'] == state]['StateName'].iloc[0]

    # Get number of nodes
    number_of_nodes = len(G.nodes())
    
    # Filter only states with more than 10 nodes
    if number_of_nodes > 10:
         
        for _, _, data in G.edges(data=True):
            weight = data.get('weight', 1)

        # Compute Louvain community detection
        louvain = nx.community.louvain_communities(G, weight='weight')

        # Compute asynchronous label propagation community detection
        asyn_label_prop = nx.community.asyn_lpa_communities(G, weight='weight')

        modularity_by_state[state] = {
            'louvain': louvain,
            'asyn_label_prop': list(asyn_label_prop)
        }

# Save the modularity results to CSV
modularity = pd.DataFrame(modularity_by_state)
csv_path = os.path.join(output_folder, "modularity_results.csv")
modularity.to_csv(csv_path, index=True)

print(f"Modularity results saved to: {csv_path}")

# Dictionary to store modularity measures
modularity_by_state = {}

for state, G in graphs_dict.items():

    # Get state name
    state_name = Fips[Fips['StateFIPS'] == state]['StateName'].iloc[0]

    # Get number of nodes
    number_of_nodes = len(G.nodes())
    
    # Filter only states with more than 10 nodes and just calculate it for Montana
    if number_of_nodes > 10 and state == '30':
         
        for _, _, data in G.edges(data=True):
            weight = data.get('weight', 1)

        # Compute Louvain community detection
        louvain = nx.community.louvain_communities(G, weight='weight')

        # Compute asynchronous label propagation community detection
        asyn_label_prop = nx.community.asyn_lpa_communities(G, weight='weight')

        modularity_by_state[state] = {
            'louvain': louvain,
            'asyn_label_prop': list(asyn_label_prop)
        }


def plot_communities(state_fips, communities, G_state, method_name):
    
    # Get state name
    state_name = Fips[Fips['StateFIPS'] == state_fips]['StateName'].iloc[0]

    # Filter the shapefile and pos dictionary
    usa = world[world['STATEFP'] == state_fips]  
    filtered_pos = pos[state_fips] 

    # Assign colors to communities
    cmap = plt.colormaps.get_cmap('gist_rainbow')
    n_communities = len(communities)

    if n_communities == 0:
        raise ValueError(f"No communities found for state {state_fips}")

    elif n_communities == 1:
        community_colors = {node: cmap(0) for node in communities[0]}  # Assign the same color

    else:
        community_colors = {
            node: cmap(i / (n_communities - 1))
            for i, community in enumerate(communities)
            for node in community
        }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    usa.plot(ax=ax, color='none', edgecolor='black')  # Transparent fill, black borders
    
    nx.draw_networkx_nodes(G_state, pos=filtered_pos, ax=ax, node_size=100,
                           node_color=[community_colors[node] for node in G_state.nodes()])
    
    nx.draw_networkx_labels(G_state, pos=filtered_pos, ax=ax)

    plt.title(f"{method_name} Community Detection in {state_name}")

    # Save the plot
    output_folder = "Community_Plots"
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"transparent_communities_{method_name}_{state_name}.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


for state_fips in Fips['StateFIPS'].unique():

    # Filter for the states where modularity was evaluated and the graph has more than 10 nodes
    if state_fips in modularity_by_state and len(graphs_dict[state_fips].nodes()) > 10 and state_fips == '30':

        # Get the state graph
        G_state = graphs_dict[state_fips]

        # Plot Louvain communities
        louvain_communities = modularity_by_state[state_fips]['louvain']
        plot_communities(state_fips, louvain_communities, G_state, "Louvain")

        # Plot Asynchronous Label Propagation communities
        asyn_label_prop = modularity_by_state[state_fips]['asyn_label_prop']
        plot_communities(state_fips, asyn_label_prop, G_state, "Asynchronous_LPA")

print(f"Saved community plots to 'Community_Plots' folder.")

import os
import networkx as nx
import matplotlib.pyplot as plt

def plot_communities(state_fips, communities, G_state, method_name):
    
    # Get state name
    state_name = Fips[Fips['StateFIPS'] == state_fips]['StateName'].iloc[0]

    # Filter the shapefile and positions dictionary
    usa = world[world['STATEFP'] == state_fips]  
    filtered_pos = pos[state_fips] 

    # Assign colors to communities
    cmap = plt.colormaps.get_cmap('gist_rainbow')
    n_communities = len(communities)

    if n_communities == 0:
        raise ValueError(f"No communities found for state {state_fips}")
    elif n_communities == 1:
        community_colors = {node: cmap(0) for node in communities[0]}
    else:
        community_colors = {
            node: cmap(i / (n_communities - 1))
            for i, community in enumerate(communities)
            for node in community
        }

    # Create plot with transparent background and only county borders
    fig, ax = plt.subplots(figsize=(10, 7))
    usa.plot(ax=ax, color='none', edgecolor='black')  # No fill, only borders

    nx.draw_networkx_nodes(G_state, pos=filtered_pos, ax=ax, node_size=100,
                           node_color=[community_colors[node] for node in G_state.nodes()])
    
    nx.draw_networkx_labels(G_state, pos=filtered_pos, ax=ax)

    plt.title(f"{method_name} Community Detection in {state_name}")

    # Save the plot as a PNG image with transparent background
    output_folder = "Community_Plots"
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"transparent_communities_{method_name}_{state_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

# Loop through all unique state FIPS codes and generate community plots
for state_fips in Fips['StateFIPS'].unique():
    # Filter for states where modularity was evaluated and the graph has more than 10 nodes
    if state_fips in modularity_by_state and len(graphs_dict[state_fips].nodes()) > 10 and state_fips == '30':
        G_state = graphs_dict[state_fips]

        # Plot Louvain communities
        louvain_communities = modularity_by_state[state_fips]['louvain']
        plot_communities(state_fips, louvain_communities, G_state, "Louvain")

        # Plot Asynchronous Label Propagation communities
        asyn_label_prop = modularity_by_state[state_fips]['asyn_label_prop']
        plot_communities(state_fips, asyn_label_prop, G_state, "Asynchronous_LPA")

print(f"Saved community plots to 'Community_Plots' folder.")


#  Upload the file Population 2017-2021.csv
# This dataset contains the population of each county in the US for the years 2017-2021
# The FIPS codes need to be read as string because of the leading zeros
dtype = {
    'STATEFP': str,
    'COUNTYFP': str,
}

population = pd.read_csv('Population 2017-2021.csv', dtype=dtype)

# Merge the Total Population column into Fips dataframe
Fips = Fips.merge(population[['STATEFP', 'COUNTYFP', 'Total Population']], 
                  left_on=['StateFIPS', 'CountyFIPS_3'], 
                  right_on=['STATEFP', 'COUNTYFP'], 
                  how='left')

# Identify missing nodes where population data is not available
missing_nodes = Fips[Fips['Total Population'].isna()][['StateFIPS', 'CountyFIPS_3']]

# Display missing nodes
if not missing_nodes.empty:
    print("Missing nodes in population data:")
    print(missing_nodes)
else:
    print("All nodes have population data.")

# Exclusion lists: skip these state FIPS codes and states with fewer than 10 nodes
exclude_states = ['60', '66', '69', '78']

# Remove the node with StateFIPS '02' and CountyFIPS_3 '261' from Fips (no population data)
Fips_filt = Fips[~((Fips['StateFIPS'] == '02') & (Fips['CountyFIPS_3'] == '261'))]

# List to store elasticity regression results
elasticity_results = []

# Create folder to save plots and results
output_folder = "Elasticity"
os.makedirs(output_folder, exist_ok=True)

# Define the linear model: log(SCI) = beta0 + beta1 * log(Distance)
def linear_model(x, beta0, beta1):
    return beta0 + beta1 * x


for state, G in graphs_dict.items():

    # Skip states in the exclusion list
    if state in exclude_states:
        continue
    
    # Get number of nodes
    number_of_nodes = len(G.nodes())

    # Skip states with fewer than 10 nodes
    if number_of_nodes < 10:
        continue

    # Get state name from Fips DataFrame
    state_name = Fips_filt[Fips_filt['StateFIPS'] == state]['StateName'].iloc[0]
    
    # Initialize lists for SCI and distance values (raw because before log transformation)
    raw_sci = []
    raw_distance = []
    
    # Get SCI data relevant to this state 
    state_sci_data = county_county[(county_county['country_origin'] == state) & 
                                   (county_county['country_dest'] == state)]
    
    # Iterate through county pairs in the SCI dataset
    for _, row in state_sci_data.iterrows():
        county_origin = row['county_origin']
        county_dest = row['county_dest']
        sci = row['scaled_sci']
        
        # Get latitude and longitude for each county from Fips
        loc1 = Fips_filt[(Fips_filt['StateFIPS'] == state) & (Fips_filt['CountyFIPS_3'] == county_origin)]
        loc2 = Fips_filt[(Fips_filt['StateFIPS'] == state) & (Fips_filt['CountyFIPS_3'] == county_dest)]
        
        if not loc1.empty and not loc2.empty:
            # Extract latitude and longitude as floats and create coordinate tuples
            lat1 = float(loc1.iloc[0]['lat'])
            lon1 = float(loc1.iloc[0]['lng'])
            lat2 = float(loc2.iloc[0]['lat'])
            lon2 = float(loc2.iloc[0]['lng'])
            coord1 = (lat1, lon1)
            coord2 = (lat2, lon2)
            
            # Compute Haversine distance (raw distance in miles)
            distance = haversine(coord1, coord2)
            
            # Store raw values (only if distance > 0 to avoid log(0))
            if distance > 0:
                raw_sci.append(sci)
                raw_distance.append(distance)
    

    # Separate the data into two subsets based on raw distance (threshold: 200 miles)
    log_sci_lt = []       # For distances < 200 miles
    log_distance_lt = []
    log_sci_ge = []       # For distances >= 200 miles
    log_distance_ge = []
    
    for sci, dist in zip(raw_sci, raw_distance):
        if dist < 200:
            log_sci_lt.append(np.log(sci))
            log_distance_lt.append(np.log(dist))
        else:
            log_sci_ge.append(np.log(sci))
            log_distance_ge.append(np.log(dist))
    
    # Fit regression on the full dataset
    all_log_distance = np.log(np.array(raw_distance))
    all_log_sci = np.log(np.array(raw_sci))
    try:
        beta_all, cov_all = curve_fit(linear_model, all_log_distance, all_log_sci)
        beta_all_err = np.sqrt(np.diag(cov_all))  
    except Exception as e:
        print(f"State {state_name}: Regression error on full dataset: {e}")
        continue

    # Fit regression for distances < 200 miles
    if len(log_distance_lt) > 1:
        beta_lt, cov_lt = curve_fit(linear_model, log_distance_lt, log_sci_lt)
        beta_lt_err = np.sqrt(np.diag(cov_lt))  
    else:
        beta_lt = [np.nan, np.nan]
        beta_lt_err = [np.nan, np.nan]

    # Fit regression for distances >= 200 miles
    if len(log_distance_ge) > 1:
        beta_ge, cov_ge = curve_fit(linear_model, log_distance_ge, log_sci_ge)
        beta_ge_err = np.sqrt(np.diag(cov_ge))  # Standard errors
    else:
        beta_ge = [np.nan, np.nan]
        beta_ge_err = [np.nan, np.nan]
    
    # Save regression results for this state
    elasticity_results.append({
        "state": state,
        "state_name": state_name,
        "number_of_nodes": number_of_nodes,
        "beta0_all": beta_all[0], "beta0_all_err": beta_all_err[0],
        "beta1_all": beta_all[1], "beta1_all_err": beta_all_err[1],
        "beta1_lt": beta_lt[1], "beta1_lt_err": beta_lt_err[1],
        "beta1_ge": beta_ge[1], "beta1_ge_err": beta_ge_err[1]
    })
    
    # Plot full dataset regression
    plt.figure(figsize=(8, 5))
    plt.scatter(all_log_distance, all_log_sci, color='darkorchid', marker = 'd', label='Data')
    x_vals = np.linspace(min(all_log_distance), max(all_log_distance), 100)
    plt.plot(x_vals, linear_model(x_vals, *beta_all), color='black',
         label=f'Fit (β₁ = {beta_all[1]:.2f} ± {beta_all_err[1]:.2f})')
    plt.xlabel("Log Distance")
    plt.ylabel("Log SCI")
    plt.title(f"Log SCI vs. Log Distance in {state_name} (Full Data)")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"{state_name}_full_regression.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot for distances < 200 miles
    if len(log_distance_lt) > 1:
        plt.figure(figsize=(8, 5))
        plt.scatter(log_distance_lt, log_sci_lt, color='darkorchid', marker = 'd', label='Data')
        x_vals = np.linspace(min(log_distance_lt), max(log_distance_lt), 100)
        plt.plot(x_vals, linear_model(x_vals, *beta_lt), color='black', 
                 label=f'Fit (β₁ = {beta_lt[1]:.2f}± {beta_lt_err[1]:.2f})')
        plt.xlabel("Log Distance")
        plt.ylabel("Log SCI")
        plt.title(f"Log SCI vs. Log Distance in {state_name} (Distance < 200 miles)")
        plt.legend()
        plt.savefig(os.path.join(output_folder, f"{state_name}_lt200_regression.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot for distances >= 200 miles
    if len(log_distance_ge) > 1:
        plt.figure(figsize=(8, 5))
        plt.scatter(log_distance_ge, log_sci_ge, color='darkorchid',  marker = 'd', label='Data')
        x_vals = np.linspace(min(log_distance_ge), max(log_distance_ge), 100)
        plt.plot(x_vals, linear_model(x_vals, *beta_ge), color='black',
                 label=f'Fit (β₁ = {beta_ge[1]:.2f}± {beta_ge_err[1]:.2f})')
        plt.xlabel("Log Distance")
        plt.ylabel("Log SCI")
        plt.title(f"Log SCI vs. Log Distance in {state_name} (Distance >= 200 miles)")
        plt.legend()
        plt.savefig(os.path.join(output_folder, f"{state_name}_ge200_regression.png"), dpi=300, bbox_inches='tight')
        plt.close()

# Save all elasticity results into a CSV file
elasticity = pd.DataFrame(elasticity_results)
elasticity.to_csv(os.path.join(output_folder, "elasticity_results.csv"), index=False)
print("Elasticity regression results saved.")


# Calculate average elasticity (β1) across states
mean_beta1_all = elasticity['beta1_all'].mean()
mean_beta1_lt = elasticity['beta1_lt'].mean()
mean_beta1_ge = elasticity['beta1_ge'].mean()

# Calculate error propagation for the average elasticity
mean_beta1_all_err = np.sqrt(np.mean(elasticity['beta1_all_err'])**2)
mean_beta1_lt_err = np.sqrt(np.mean(elasticity['beta1_lt_err'])**2)
mean_beta1_ge_err = np.sqrt(np.mean(elasticity['beta1_ge_err'])**2)


print("Average Elasticity across States:")
print(f"Full dataset elasticity (β1): {mean_beta1_all:.2f} ± {mean_beta1_all_err:.2f}")
print(f"Elasticity for distances < 200 miles (β1): {mean_beta1_lt:.2f} ± {mean_beta1_lt_err:.2f}")
print(f"Elasticity for distances >= 200 miles (β1): {mean_beta1_ge:.2f} ± {mean_beta1_ge_err:.2f}")

import pandas as pd

# Analyze correlation between county population and strength centrality for a given state.
def analyze_population_vs_strength(state_fips):
    
    # Get the population data for this state
    state_population_df = Fips[Fips['StateFIPS'] == state_fips]
    
    # Get the list of counties (nodes) for this state
    nodes = list(graphs_dict[state_fips].nodes())  

    # Create DataFrame mapping CountyFIPS_3 to strength centrality
    strength_df = pd.DataFrame({'CountyFIPS_3': nodes, 'Strength Centrality': strength_by_state[state_fips]})
  
    # Merge population data with strength centrality
    merged_df = pd.merge(state_population_df, strength_df, on='CountyFIPS_3')

    # Compute and print the correlation
    correlation = merged_df[['Total Population', 'Strength Centrality']].corr()
    print(f"Correlation for state {state_fips}:\n{correlation}\n")

    return correlation


# Loop through all states to analyze population vs. strength centrality
correlations = {}

for state_fips in Fips['StateFIPS'].unique():
    if state_fips not in {'09', '10', '11', '66', '69', '33'}:  # Exclude small states if needed
        num_nodes = len(graphs_dict[state_fips].nodes())  # Count nodes in the state's graph
        
        if num_nodes > 10:  # Only analyze if the state has more than 10 nodes
            correlations[state_fips] = analyze_population_vs_strength(state_fips)

# Save the correlation results to a CSV file in the folder 'Strength_Centrality'
correlation = pd.DataFrame(correlations)
correlation.to_csv("Strength_Centrality/correlation_results.csv")

def plot_population_vs_strength(state_fips):

    if state_fips not in correlations or correlations[state_fips] is None:
        print(f"Skipping state {state_fips} due to missing data.")
        return
    
    # Get the population and strength data for the state
    state_population = Fips[Fips['StateFIPS'] == state_fips]
    
    # Get list of nodes for this state
    nodes = list(graphs_dict[state_fips].nodes())
    
    # Ensure the number of nodes matches strength values
    if len(nodes) != len(strength_by_state[state_fips]):
        print(f"Skipping state {state_fips} due to mismatch in data size.")
        return
    
    # Create DataFrame mapping CountyFIPS_3 to strength centrality
    strength = pd.DataFrame({'CountyFIPS_3': nodes, 'Strength Centrality': strength_by_state[state_fips]})

    # Merge with population data
    merged_df = pd.merge(state_population, strength, on='CountyFIPS_3')

    # Scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=merged_df, x="Total Population", y="Strength Centrality", alpha=0.7)
    
    plt.xlabel("Total Population")
    plt.ylabel("Strength Centrality")
    plt.title(f"Population vs. Strength Centrality ({state_fips})")
    plt.xscale("log")  # Use log scale to better visualize variations
    plt.yscale("log")  # Use log scale for centrality
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Save the figure
    output_folder = "Correlation_Plots"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"correlation_{state_fips}.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

# Loop through all valid states and generate plots
for state_fips in correlations.keys():
    plot_population_vs_strength(state_fips)

print(f"Saved correlation plots in 'Correlation_Plots' folder.")




