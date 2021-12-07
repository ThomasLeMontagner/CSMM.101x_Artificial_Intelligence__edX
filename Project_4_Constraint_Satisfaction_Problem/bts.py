from copy import deepcopy


def backtracking_search(csp):
    return backtrack({}, csp)


def backtrack(assignment, csp):
    if is_complete(assignment, csp):
        return assignment

    var = select_unassigned_variables(assignment, csp)
    original_domain = deepcopy(csp.possible_values)
    for value in order_domain_values(var, csp):
        if is_value_consistent(var, value, assignment, csp):
            assign_value(var, value, assignment)
            fc = forward_checking(assignment, {}, csp, var, value)

            if fc:
                result = backtrack(assignment, csp)
                if not (result == False):
                    return result

            remove_value(var, assignment)
            csp.possible_values.update(original_domain)
    return False


def forward_checking(assignment, fc, csp, var, value):
    fc[var] = value

    for neighbor in csp.neighbors[var]:
        if neighbor not in assignment.keys() and value in csp.possible_values[neighbor]:
            if len(csp.possible_values[neighbor])==1:
                return False

            remaining_value = csp.possible_values[neighbor]
            remaining_value.remove(value)

            if len(remaining_value)==1:
                check_remaining_value(assignment, fc, csp, neighbor, remaining_value)
    return fc


# Raise False if the forward checking fails with the remaining value
def check_remaining_value(assignment, fc, csp, neighbor, remaining_value):
    check = forward_checking(assignment, fc, csp, neighbor, remaining_value)
    if not check:
        return False


# Return true if the sudoku is complete
def is_complete(assignment, csp):
    return set(csp.variables) == set(assignment.keys())


# Pick up variable with the less remaining value
def select_unassigned_variables(assignment, csp):
    uv = dict((var, len(csp.possible_values[var])) for var in get_remaining_variables(assignment, csp))
    return min(uv, key=uv.get)


# Choose the Least Constraining Value (lcv)
def order_domain_values(var, csp):
    return csp.possible_values[var]


# Return True if assigning value to var leaves place for other variables
def is_value_consistent(var, value, assignment, csp):
    for cv in csp.neighbors[var]:
        if cv in assignment.keys() and assignment[cv] == value:
            return False
    return True


# Assign value to var
def assign_value(var, value, assignment):
    assignment[var] = value


# Remove the value in the remaining values of var
def remove_value(var, assignment):
    del assignment[var]


# Return non assigned variables
def get_remaining_variables(assignment, csp):
    return set(csp.variables) - set(assignment.keys())


# Print the result
def get_result(assignment):
    result = ""
    for k in sorted(assignment.keys()):
        result += str(assignment[k])
    return result
