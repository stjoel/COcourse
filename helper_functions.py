import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import geopandas as gpd
import fiona as fi

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from scipy.spatial.distance import pdist, squareform


def load_data(): 
    "data from https://kartkatalog.geonorge.no/metadata/administrative-enheter-municipalities/041f1e6e-bdbc-4091-b48f-8a5990f3cc5b"
    
    return gpd.read_file('/Users/joelfinnbjervig/Documents/CO course/code/Basisdata_0000_Norge_25833_Kommuner_GeoJSON.geojson', layer = 'administrative_enheter.kommune')


def get_positions(municipalities, municipalities_locations:list):
    muns  = pd.DataFrame([municipalities.navn[i].split('\"')[3] for i in range(len(municipalities))], columns = ['name'])

    muns_i = [muns[muns['name']==mun_name].index.values[0] for mun_name in municipalities_locations]
    
    x_positions = municipalities.geometry[muns_i].centroid.x.values
    y_positions = municipalities.geometry[muns_i].centroid.y.values
    
    positions = [(x, y) for x,y in zip(x_positions,y_positions) ]
    return positions


def view_locations(municipalities, locations: list, positions:list):
    ax = municipalities.plot(column = 'lokalid' , edgecolor = 'black', linewidth = 0.3)
    
    for dest, pos in zip(locations, positions):
        ax.annotate(text = dest, xy = pos, xytext = (pos[0]+1E4,pos[1]+1.5E4),
                    color = 'black', backgroundcolor = 'gray', size = 6,
                    #arrowprops =dict(arrowstyle='-', facecolor = 'black'),
                    bbox = dict(boxstyle = 'round4', alpha = 0.75, color = 'w'))
        
        ax.scatter(pos[0],pos[1], color = 'black', s = 10)
        
    ax.set_xlim([-0.15E6, .45E6])
    ax.set_ylim([6.4E6  , 7.2E6])
    ax.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    #plt.savefig('figures/locations2.svg', transparent=True)
    plt.show()


def view_all_possible_tours(municipalities, locations:list, positions):
    
    muns_n = len(positions)
    
    ax = municipalities.plot(column = 'lokalid', edgecolor = 'black', linewidth = 0.3)
    for i, (dest, pos) in enumerate(zip(locations, positions)):
        # plot locations
        ax.scatter(pos[0], pos[1], color = 'black', s = 30)
        
        # plot names of locations
        ax.annotate(text = dest, xy = pos, xytext = (pos[0]+1E4, pos[1]+1.5E4),
                    color = 'black', backgroundcolor = 'gray', size = 6,
                    #arrowprops =dict(arrowstyle='-', facecolor = 'black'),
                    bbox = dict(boxstyle = 'round4', alpha = 0.75, color = 'w'))
        # plot all possible tours between locations
        for j in range(i,muns_n):
            dx = pos[0] - positions[j][0]
            dy = pos[1] - positions[j][1]
            ax.plot([pos[0], pos[0]-dx],[pos[1], pos[1]-dy],
                    color = 'black', alpha = 0.5, linewidth = 1)
    
    # plot settings
    ax.set_xlim([-0.15E6, .45E6])
    ax.set_ylim([6.4E6, 7.2E6])
    ax.set_title(f"There are (n-1)!/2 = {int(np.math.factorial(muns_n-1)/2)} unique tours\nfor the n = {muns_n} locations")
    ax.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    plt.show()
    plt.savefig('figures/tour_all.svg', transparent=True)


def create_data_model(positions:list):
    """Stores the data for the problem."""
    data = {}
    
    data['locations']       = positions
    data['num_vehicles']    = 1 # one vehicle for TSP
    data['depot']           = 5 # Start from Oslo
    return data


def create_distance_callback(data, manager):
    """Creates callback to return distance between points."""
    distances_ = {}
    index_manager_ = manager
    # precompute distance between location to have distance callback in O(1)
    for from_counter, from_node in enumerate(data['locations']):
        distances_[from_counter] = {}
        for to_counter, to_node in enumerate(data['locations']):
            if from_counter == to_counter:
                distances_[from_counter][to_counter] = 0
            else:
                distances_[from_counter][to_counter] = (
                    abs(from_node[0] - to_node[0]) +
                    abs(from_node[1] - to_node[1]))
                
                """distances_[from_counter][to_counter] = np.sqrt(
                    (from_node[0] - to_node[0])**2 +
                    (from_node[1] - to_node[1])**2)"""
    def distance_callback(from_index, to_index):
        """Returns the manhattan or euclidean distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = index_manager_.IndexToNode(from_index)
        to_node = index_manager_.IndexToNode(to_index)
        return distances_[from_node][to_node]

    return distance_callback


class SolutionCallback(object):
        def __init__(self, model):
            self.model = model
            self.cost = []
            
        def __call__(self):
            self.cost.append(self.model.CostVar().Value())


def print_solution(manager, routing, assignment, cost, locations):
    """Prints assignment on console."""
    #print('Objective: {}'.format(assignment.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'tour for roadtrip:\n'
    plan = []
    tour_distance = 0
    while not routing.IsEnd(index):
        plan.append(manager.IndexToNode(index))
        previous_index = index
        index = assignment.Value(routing.NextVar(index))
        tour_distance+=routing.GetArcCostForVehicle(previous_index, index, 0)
    plan.append(plan[0])
    for p in plan[:-1]:
        plan_output += '{}'.format(locations[p]+' -> ')
    plan_output += '{}'.format(locations[plan[-1]]+'\n')
    
    plan_output += 'Distance of the tour: {} length units.\n'.format(tour_distance)
    print(plan_output)
    plt.plot(cost)
    plt.show()

    return plan


def ortools_solver(positions:list, locations:list)->list:
    
    data = create_data_model(positions)
    manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                           data['num_vehicles'], data['depot'])
    
    routing = pywrapcp.RoutingModel(manager)
    
    distance_callback = create_distance_callback(data, manager)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
            
    solution_callback = SolutionCallback(routing)
    routing.AddAtSolutionCallback(solution_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

     # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    assignment = routing.SolveWithParameters(search_parameters)
    
    # Print solution on console.
    if assignment:
        tour = print_solution(manager, routing, assignment, solution_callback.cost, locations)
    return tour


def view_ortools_sol(municipalities, positions, tour:list, locations:list):
    plt.figure(figsize = (20,20))
    ax = municipalities.plot(column = 'lokalid', edgecolor = 'black', linewidth = 0.3)
    ax.set_xlim([-0.15E6, .45E6])
    ax.set_ylim([6.4E6, 7.2E6])
    ax.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

    x_tour = [positions[i][0] for i in tour]
    y_tour = [positions[i][1] for i in tour]

    dx_arr  , dy_arr    = np.diff(np.array(x_tour)), np.diff(np.array(y_tour))

    for loc_i, i in enumerate(tour[:-1]):

        ax.annotate(text    = locations[loc_i],
                    xy      = (positions[loc_i][0], positions[loc_i][1]),
                    xytext  = (positions[loc_i][0]+1E4, positions[loc_i][1]+1.5E4),
                    color   = 'black',
                    backgroundcolor = 'gray',
                    size = 6,
                    bbox = dict(boxstyle = 'round4',
                                alpha = 0.75,
                                color = 'w'))
        
        ax.plot((x_tour[i], x_tour[i]+dx_arr[i]),
                (y_tour[i], y_tour[i]+dy_arr[i]),
                color = 'black',
                linewidth = 1)
    #plt.savefig('figures/tour_sol.svg', transparent=True)
    plt.show()


def nearest_neighbor(current_location, locations):
    "Find the location in locations that is nearest to location A."
    
    nn = min(locations, key=lambda c: distance(c, current_location))
    
    return nn


def distance(A, B):
    diff=0
    for a,b in zip(A,B): diff += np.abs(a-b)
    return diff
