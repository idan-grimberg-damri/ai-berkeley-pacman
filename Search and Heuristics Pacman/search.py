# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """

    def getStartState(self):
        """
     Returns the start state for the search problem 
     """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


STATE = 0
ACTION = 1
PATH_COST = STEP_COST = 2
PARENT = 3


def graphSearch(problem, frontier):
    """
    general graph search
    param : problem - a description of the problem (given)
    param : frontier - the data structure we'll use for choosing the next node to expand
            for example: a LIFO-QUEUE
    returns: an ordered list that represents a solution
            for example : ['West', 'West', 'East', 'South']
    """
    from game import Directions

    # according to page 78 in the book
    # the components of a node (a bookkeeping data structure):
    state = problem.getStartState()
    action = Directions.STOP
    pathCost = 0
    parent = None  # we'll construct/restore the solution using the parent node

    # start with constructing the node that contains a start state
    # an example for what currNode can contain : ((2,4), 'East', 1, ((2,3), 'Stop', None))
    # in the example, we can reach currNode from position (2,3) after apply the action 'East'
    # currNode's grandparent is None, so in this example currNode's parent is a start state
    currNode = (state, action, pathCost, parent)

    # if the start state is a goal state
    if problem.isGoalState(state):
        return action

    # boolean variable for helping eliminate sparse expanded nodes if the frontier is a LIFO/FIFO Queue
    # if this value is False then we should check for a goal state after a node is selected for expansion
    shouldApplyGoalTestWhenGenerated = not isinstance(frontier, util.PriorityQueueWithFunction)
    # a list that contains nodes that we already popped from the frontier
    explored = []
    # a dictionary for a quick lookup for a duplicated node in the frontier
    # a key is a tuple that represents a state
    # value is a non-negative integer
    # that represents a path cost from a start state to a state represented by the key
    dictFrontier = {}

    frontier.push(currNode)
    # some stated are not a tuple but a key in the dictionary must be an immutable type
    dictFrontier[tuple(currNode[STATE])] = pathCost

    while not frontier.isEmpty():
        currNode = frontier.pop()
        # this is relevant only when the frontier is a PriorityQueueWithFunction
        if (not shouldApplyGoalTestWhenGenerated) and problem.isGoalState(currNode[STATE]):
            return solution(currNode)

        explored.append(currNode[STATE])
        # get all legal successors of curr_node
        successorsList = problem.getSuccessors(currNode[STATE])
        for successor in successorsList:
            # construct child node
            childNode = (successor[STATE], successor[ACTION], successor[STEP_COST] + currNode[PATH_COST], currNode)
            childNodeState = childNode[STATE]
            childNodePathCost = childNode[pathCost]
            # if we need to check for a goal state right after generated childNode
            # and childNode is a goal state then return a solution
            if shouldApplyGoalTestWhenGenerated and problem.isGoalState(childNodeState):
                return solution(childNode)

            isChildNodeAlreadyInFrontier = tuple(childNodeState) in dictFrontier
            # if childNode is not in the frontier and not in the explored set
            # then add him to the frontier and add a related entry in the dictionary
            if not (isChildNodeAlreadyInFrontier or childNodeState in explored):
                frontier.push(childNode)
                dictFrontier[tuple(childNodeState)] = childNodePathCost
            # else if the frontier is a PriorityQueueWithFunction and childNode is already in the frontier
            # then check if the new path from a start state to childNode is cheaper
            # than the one currently in the frontier
            elif (not shouldApplyGoalTestWhenGenerated) and isChildNodeAlreadyInFrontier:
                childNodePriorityInFrontier = dictFrontier.get(tuple(childNodeState))
                if childNodePathCost < childNodePriorityInFrontier:
                    # update the dictionary to the new and cheaper path cost from
                    # a start state to childNode
                    dictFrontier[tuple(childNodeState)] = childNodePathCost
                    # I didn't do an update to the frontier, since in the documentation for PriorityQueueWithFunction
                    # they wrote that we can just push to the frontier a duplicate with a better priority
                    frontier.push(childNode)

    # no solution
    return []


def solution(goalState):
    """
    construct/restore a solution (a list of ordered actions from left to right,
        that if you apply them on the problem then
        you'll be able to get from a start state to a goal state)
    param : node - a goal state
    returns : a list that represents a solution
            for example : ['West', 'West', 'East', 'South']
    """
    listOfActions = []
    currNode = goalState
    # construct the solution:
    # while there're nodes available on the path from start to goal
    while currNode:
        # append the action you'll need to apply for getting to currNode
        # from currNode's parent
        listOfActions.append(currNode[ACTION])
        # restore the path
        currNode = currNode[PARENT]
    # we're given a path : goalState---->startState
    # but we want a path : startState---->goalState
    listOfActions.reverse()
    # without the first action ('Stop')
    lst = listOfActions[1:]
    return lst


def depthFirstSearch(problem):
    """
  Search the deepest nodes in the search tree first [p 74].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.18].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
    from util import Stack

    lifoQueue = Stack()

    return graphSearch(problem, lifoQueue)


def breadthFirstSearch(problem):
    "Search the shallowest nodes in the search tree first. [p 74]"
    from util import Queue

    fifoQueue = Queue()

    return graphSearch(problem, fifoQueue)


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.

     """
    from util import PriorityQueueWithFunction

    # construct a function g:
    # param : node - a node that contains a state in the state space of the problem
    # returns : the cost of the path from a start state to node (in searchAgents.py)
    g = lambda node: node[PATH_COST]

    priorityQueue = PriorityQueueWithFunction(g)

    return graphSearch(problem, priorityQueue)


def nullHeuristic(state, problem=None):
    """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    from searchAgents import manhattanHeuristic
    from util import PriorityQueueWithFunction

    # in g,h,f:
    # param : node - a node that contains a state in the state space of the problem

    # construct function g
    # returns : the cost of the path from a start state to node (in searchAgents.py)
    g = lambda node: node[PATH_COST]

    # construct function h
    # returns : the estimated cost of the path from node to a goal state
    #           as implemented in manhattanHeuristic
    h = lambda node: heuristic(node[STATE], problem)

    # construct function f
    # returns : the estimated cost of the cheapest path through node
    f = lambda node: g(node) + h(node)

    priorityQueue = PriorityQueueWithFunction(f)

    return graphSearch(problem, priorityQueue)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
