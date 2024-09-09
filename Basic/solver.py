from qrisp import auto_uncompute, QuantumBool, QuantumDictionary, mcx, QuantumFloat, control
import numpy as np

import matplotlib.pyplot as plt


import networkx as nx

def input_obj_transformationa(input_obj):
    """
    Convert a graph coloring problem using networkx, s.t. it can be solved with backtracking.

    Parameters:
    - sudoku_board: 4x4 numpy array with numbers 0 to 3 for set fields and -1 for empty fields.

    Returns:
    - G: networkx graph representing the Sudoku problem.
    - empty_nodes: list of nodes corresponding to the empty fields.
    """
    if type(input_obj) == nx.graph:
        #TBD
        adj_matrix = nx.adjacency_matrix(input_obj)
        pass

    #Need to adjust to structure
    input_nodes = input_obj[0]
    input_edges = input_obj[1]
    edge_copy = input_edges.copy()
    adj_dict = {}
    for edge in edge_copy:
        for index in range(len(edge)):
            if edge[index] in list(adj_dict.keys()):
                adj_dict[edge[index]] +=[edge[1-index]]
            else:
                adj_dict.setdefault(edge[index], [edge[1-index]])
    #build a connection dict
    

    # Create an empty graph
    G = nx.Graph()
    empty_nodes = []
    # Add nodes and edges
    for i in range(len(input_nodes)):
        
        #test this for now, but might need check for double iteration over same edge
        if input_nodes[i] == -1:
            # Add node for each empty cell
            node = i
            adj_nodes = adj_dict[i]
            empty_nodes.append(node)
            G.add_node(node)
            for adj in adj_nodes:
                if input_nodes[adj] == -1: 
                    G.add_edge(node, adj, edge_type = "qq")
                else:
                    G.add_edge(node, adj, edge_type = "cq")


    return G, empty_nodes


def sudoku_to_graph(sudoku_board):
    """
    Convert a 4x4 Sudoku problem into a graph coloring problem using networkx.

    Parameters:
    - sudoku_board: 4x4 numpy array with numbers 0 to 3 for set fields and -1 for empty fields.

    Returns:
    - G: networkx graph representing the Sudoku problem.
    - empty_nodes: list of nodes corresponding to the empty fields.
    """

    # Create an empty graph
    G = nx.Graph()
    empty_nodes = []
    # Add nodes and edges
    for i in range(sudoku_board.shape[0]):
        for j in range(sudoku_board.shape[0]):
            if sudoku_board[i, j] == -1:

                # Add node for each empty cell
                node = (i, j)
                empty_nodes.append(node)
                G.add_node(node)

                # Connect to nodes in the same row
                for k in range(4):
                    if k != j:

                        # This distincts, wether it is a quantum-quantum or a
                        # classical quantum comparison.
                        # Multiple classical-quantum comparisons can be executed
                        # in a single QuantumDictionary call
                        if sudoku_board[i,k] == -1:
                            G.add_edge(node, (i, k), edge_type = "qq")
                        else:
                            G.add_edge(node, (i, k), edge_type = "cq")

                # Connect to nodes in the same column
                for k in range(4):
                    if k != i:
                        if sudoku_board[k,j] == -1:
                            G.add_edge(node, (k, j), edge_type = "qq")
                        else:
                            G.add_edge(node, (k, j), edge_type = "cq")

                # Connect to nodes in the same 2x2 subgrid
                subgrid_start_row = (i // 2) * 2
                subgrid_start_col = (j // 2) * 2
                for k in range(subgrid_start_row, subgrid_start_row + 2):
                    for l in range(subgrid_start_col, subgrid_start_col + 2):
                        if (k, l) != node:
                            if sudoku_board[k,l] == -1:
                                G.add_edge(node, (k, l), edge_type = "qq")
                            else:
                                G.add_edge(node, (k, l), edge_type = "cq")

    
    return G, empty_nodes

def extract_comparisons(problemType, problemToSolve):
    if problemType == "Graph":
        return extract_comparisons_coloring(problemToSolve)

    elif problemType == "Sudoku":
        return extract_comparisons_sudoku(problemToSolve)

def extract_comparisons_coloring(input_obj):
    """
    Takes a Sudoku board in the form of a numpy array
    where the empty fields are indicated by the value -1.

    Returns two lists:
    1. The quantum-quantum comparisons in the form of a list[(int, int)]
    2. The batched classical-quantum comparisons in the form dict({int : list[int]})
    """
    #Need to adjust to structure
    input_nodes = input_obj[0]
    input_edges = input_obj[1]

    num_empty_fields = sum([1 for index in range(len(input_nodes)) if input_nodes[index] == -1])
    #print(num_empty_fields)
    # Generate the comparison graph
    graph, empty_nodes = input_obj_transformationa( input_obj)

    # Generate the list of required comparisons

    # This dictionary contains the classical-quantum comparisons for each
    # quantum entry

    cq_checks = {q_assignment_index : [] for q_assignment_index in range(num_empty_fields)}
    #print(cq_checks)
    # This dictionary contains the quantum-quantum comparisons as tuples
    qq_checks = []

    # Each edge of the graph corresponds to a comparison.
    # We therefore iterate over the edges distinguish between the classical-quantum
    # and quantum-quantum comparisons

    for edge in graph.edges():
        edge_type = graph.get_edge_data(*edge)["edge_type"]

        # Append the quantum-quantum comparison to the corresponding list
        if edge_type == "qq":
            assigment_index_0 = empty_nodes.index(edge[0])
            assigment_index_1 = empty_nodes.index(edge[1])

            qq_checks.append((assigment_index_0, assigment_index_1))

        # Append the classical quantum comparison to the corresponding dictionary
        elif edge_type == "cq":

            if input_nodes[edge[1]] == -1:
                q_assignment_index = empty_nodes.index(edge[1])
                cq_checks[q_assignment_index].append(input_nodes[edge[0]])
            else:
                q_assignment_index = empty_nodes.index(edge[0])
                cq_checks[q_assignment_index].append(input_nodes[edge[1]])

    return qq_checks, cq_checks


def extract_comparisons_sudoku(sudoku_board):
    """
    Takes a Sudoku board in the form of a numpy array
    where the empty fields are indicated by the value -1.

    Returns two lists:
    1. The quantum-quantum comparisons in the form of a list[(int, int)]
    2. The batched classical-quantum comparisons in the form dict({int : list[int]})
    """

    num_empty_fields = np.count_nonzero(sudoku_board == -1)

    # Generate the comparison graph
    graph, empty_nodes = sudoku_to_graph(sudoku_board)

    # Generate the list of required comparisons

    # This dictionary contains the classical-quantum comparisons for each
    # quantum entry
    cq_checks = {q_assignment_index : [] for q_assignment_index in range(num_empty_fields)}

    # This dictionary contains the quantum-quantum comparisons as tuples
    qq_checks = []

    # Each edge of the graph corresponds to a comparison.
    # We therefore iterate over the edges distinguish between the classical-quantum
    # and quantum-quantum comparisons

    for edge in graph.edges():
        edge_type = graph.get_edge_data(*edge)["edge_type"]

        # Append the quantum-quantum comparison to the corresponding list
        if edge_type == "qq":
            assigment_index_0 = empty_nodes.index(edge[0])
            assigment_index_1 = empty_nodes.index(edge[1])

            qq_checks.append((assigment_index_0, assigment_index_1))

        # Append the classical quantum comparison to the corresponding dictionary
        elif edge_type == "cq":

            if sudoku_board[edge[1]] == -1:
                q_assignment_index = empty_nodes.index(edge[1])
                cq_checks[q_assignment_index].append(sudoku_board[edge[0]])
            else:
                q_assignment_index = empty_nodes.index(edge[0])
                cq_checks[q_assignment_index].append(sudoku_board[edge[1]])

    return qq_checks, cq_checks


def eval_qq_checks( qq_checks,
                q_assigments,
                h):
    """
    Batched cq_checks is a list of the form

    [(int, int)]

    Where each tuple entry corresponds the index
    of the quantum value that should be compared.
    branch_qa and height are the quantum values
    that specify the tree state.
    """
    # Create result list
    res_qbls = []

    # Iterate over all comparison tuples
    # to evaluate the comparisons.
    for ind_0, ind_1 in qq_checks:
        # Enter the control environment
        with control(h[min(ind_0, ind_1)]):
            # Evaluate the comparison
            eq_qbl = (q_assigments[ind_0] ==
                      q_assigments[ind_1])
        res_qbls.append(eq_qbl)

    # Return results
    return res_qbls

def cq_eq_check(q_value, cl_values):
    """
    Receives a QuantumVariable and a list of classical
    values and returns a QuantumBool, indicating whether
    the value of the QuantumVariable is contained in the
    list of classical values
    """

    if len(cl_values) == 0:
        # If there are no values to compare with, we
        # return False
        return QuantumBool()

    # Create dictionary
    qd = QuantumDictionary(return_type = QuantumBool())

    # Determine the values that q_value can assume
    value_range = [q_value.decoder(i) for i in range(2**q_value.size)]

    # Fill dictionary with entries
    for value in value_range:
        if value in cl_values:
            qd[value] = True
        else:
            qd[value] = False

    # Evaluate dictionary with quantum value
    return qd[q_value]


def eval_cq_checks( batched_cq_checks,
                    q_assigments,
                    h):
    """
    Batched cq_checks is a dictionary of the form

    {int : list[int]}

    Each key/value pair corresponds to
    one batched quantum-classical comparison.
    The keys represent the the quantum values
    as indices of q_assigments and the values
    are the list of classical values that
    the quantum value should be compared with.
    q_assigments and height are the quantum values
    that specify the state of the tree.
    """
    # Create result list
    res_qbls = []

    # Iterate over all key/value pairs to evaluate
    # the comparisons.
    for key, value in batched_cq_checks.items():
        # Enter the control environment
        with control(h[key]):
            # Evaluate the comparison
            eq_qbl = cq_eq_check(q_assigments[key],
                                 value)
        res_qbls.append(eq_qbl)

    # Return results
    return res_qbls



def check_singular_problem_assignment(problemType, problemToSolve, q_assigments, h):
    """
    Takes the following arguments:

    1. sudoku_board is Sudoku board in the form of a numpy array
    where the empty fields are indicated by the value -1.

    2. q_assigments is a QuantumArray of type
    type QuantumFloat, describing the assignments.

    3. h is a one-hot encoded QuantumVariable representing, which
    assignment should be checked for validity

    The function returns a QuantumBool, indicating whether
    the assigment indicated by h respects the rules of Sudoku.
    """  

    if problemType == "Graph":
        
        num_empty_fields = sum([1 for index in range(len(problemToSolve[0])) if problemToSolve[0][index] == -1])
        if num_empty_fields != len(q_assigments):
            raise Exception("Number of empty field and length of assigment array disagree.")

    elif problemType == "Sudoku":
        num_empty_fields = np.count_nonzero(problemToSolve == -1)
        if num_empty_fields != len(q_assigments):
            raise Exception("Number of empty field and length of assigment array disagree.")

    # Generate the comparisons
    qq_checks, cq_checks = extract_comparisons(problemType, problemToSolve)

    # Evaluate the comparisons
    comparison_qbls = []

    # quantum-quantum
    comparison_qbls += eval_qq_checks(qq_checks, q_assigments, h)

    # classical-quantum
    comparison_qbls += eval_cq_checks(cq_checks, q_assigments, h)

    # Allocate result
    sudoku_valid = QuantumBool()

    # Compute the result
    mcx(comparison_qbls, sudoku_valid, ctrl_state = 0, method = "balauca")

    return sudoku_valid




























#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging
from typing import Tuple
from typing import TypedDict

import os
import numpy as np

# from qiskit_ibm_runtime import QiskitRuntimeService


def run(problem_in, prec,shots, **kwargs: dict):
    """
    Run Qrisp QIRO on the local Qrisp simulator

    :param mapped_problem: dictionary with the keys 'graph' and 't'
    :type mapped_problem: any
    :param device_wrapper: instance of device
    :type device_wrapper: any
    :param config:
    :type config: Config
    :param kwargs: no additionally settings needed
    :type kwargs: any
    :return: Solution, the time it took to compute it and optional additional information
    :rtype: tuple(list, float, dict)
    """
    




    problemType =  "Graph"
    problemToSolve  = problem_in
    num_empty_fields = sum([1 for index in range(len(problem_in[0])) if problem_in[0][index] == -1])
    @auto_uncompute
    def accept(tree):
        return tree.h == 0

    @auto_uncompute
    def reject(tree):
        # Cut off the assignment with height 0
        # since it is not relevant for the sudoku
        # checker
        q_assigments = tree.branch_qa[1:]
        # Modify the height to reflect the cut off
        modified_height = tree.h[1:]
        assignment_valid = check_singular_problem_assignment(problemType,
                                                            problemToSolve,
                                                            q_assigments,
                                                            modified_height)
        return assignment_valid.flip()


    from qrisp.quantum_backtracking import QuantumBacktrackingTree as QBT

    tree = QBT(max_depth = num_empty_fields+1,
            branch_qv = QuantumFloat(2),
            accept = accept,
            reject = reject,
            subspace_optimization = True)

    # Initialize root
    tree.init_node([])

    #Perform QPE
    qpe_res = tree.estimate_phase(precision = prec)

    # Retrieve measurements
    mes_res = qpe_res.get_measurement(shots = shots)


    if mes_res[0]>0.375:
        print("Solution exists")
    elif mes_res[0]<0.25:
        print("No solution exists")
    else:
        print("Insufficent precision")



    tree = QBT(max_depth = num_empty_fields+1,
            branch_qv = QuantumFloat(2),
            accept = accept,
            reject = reject,
            subspace_optimization = True)

    sol = tree.find_solution(precision = prec)
    final_array = sol[::-1][1:]
    #print(sol[::-1][1:])
    print(final_array)
    return final_array


