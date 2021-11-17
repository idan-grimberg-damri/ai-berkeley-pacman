# searchAgents.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
This file contains all of the agents that can be selected to 
control Pacman.  To select an agent, use the '-p' option
when running pacman.py.  Arguments can be passed to your agent
using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a searchFunction=depthFirstSearch

Commands to invoke other search strategies can be found in the 
project description.

Please only change the parts of the file you are asked to.
Look for the lines that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the
project description for details.

Good luck and happy searching!
"""
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import searchAgents


class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP


#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
  This very general search agent finds a path using a supplied search algorithm for a
  supplied search problem, then returns actions to follow that path.
  
  As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)
  
  Options for fn include:
    depthFirstSearch or dfs
    breadthFirstSearch or bfs
    
  
  Note: You should NOT change any code in SearchAgent
  """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in dir(searchAgents):
                heur = getattr(searchAgents, heuristic)
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in dir(searchAgents) or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = getattr(searchAgents, prob)
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
    This is the first time that the agent sees the layout of the game board. Here, we
    choose a path to the goal.  In this phase, the agent should compute the path to the
    goal and store it in a local variable.  All of the work is done in this method!
    
    state: a GameState object (pacman.py)
    """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state)  # Makes a new search problem
        self.actions = self.searchFunction(problem)  # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
    Returns the next action in the path chosen earlier (in registerInitialState).  Return
    Directions.STOP if there is no further action to take.
    
    state: a GameState object (pacman.py)
    """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
  A search problem defines the state space, start state, goal test,
  successor function and cost function.  This search problem can be 
  used to find paths to a particular point on the pacman board.
  
  The state space consists of (x,y) positions in a pacman game.
  
  Note: this search problem is fully specified; you should NOT change it.
  """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True):
        """
    Stores the start and goal.  
    
    gameState: A GameState object (pacman.py)
    costFn: A function from a search state (tuple) to a non-negative number
    goal: A position in the gameState
    """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
    Returns successor states, the actions they require, and a cost of 1.
    
     As noted in search.py:
         For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
    """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
    Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999
    """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


class StayEastSearchAgent(SearchAgent):
    """
  An agent for position search with a cost function that penalizes being in
  positions on the West side of the board.  
  
  The cost function for stepping into a position (x,y) is 1/2^x.
  """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)


class StayWestSearchAgent(SearchAgent):
    """
  An agent for position search with a cost function that penalizes being in
  positions on the East side of the board.  
  
  The cost function for stepping into a position (x,y) is 2^x.
  """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

"""
 self.corners contains exactly 4 items 
 An image of the corners we visited can be represented as an integer 0 <= img <= 2^4 - 1 
 in binary : 0000 <= img <= 1111, so I will use bitwise operators to manipulate
 the image of the corners.
 An example in binary, assume we order the images of the corners in an increasing order:
     first image of self.corners is 0000 which says that we didn't visit any corner yet
     last image of self.corners can be  1111  which says that we visited all the corners
"""
# maximum most significant digit position for an image value in binary
MAX_BINARY_MSD_POSITION = 3
# 1111 in binary, means we visited all the corners
GOAL_VALUE = 15


class CornersProblem(search.SearchProblem):
    """
  This search problem finds paths through all four corners of a layout.

  A state in the state space for this problem is a list of the form [pacman's position, corners image]
  s.t:
  pacman position is an ordered pair of positive integers
  corners image is an integer between 0 to 15, as explained above
    for example : [(6,6), 15] is a goal state
  """

    def __init__(self, startingGameState):
        """
    Stores the walls, pacman's starting position and corners.
    """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))

        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0  # Number of search nodes expanded
        "*** YOUR CODE HERE ***"
        # the second element means that we didn't visit any cornet yet
        self.startState = [self.startingPosition, 0]
        # if we start from a corner then update the corners image
        if self.startingPosition in self.corners:
            self.updateCornersImage(self.startState)

    def getStartState(self):
        "Returns the start state (in your state space, not the full Pacman state space)"
        return self.startState

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem

        returns : true if we visited all the corners , false otherwise
        """
        # should be equal to 15 (1111 in binary)
        # this means that we visited all the corners
        return state[1] == GOAL_VALUE

    def getSuccessors(self, state):
        """
    Returns successor states, the actions they require, and a cost of 1.
    
     As noted in search.py:
         For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
    """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            hitsWall = self.walls[nextx][nexty]
            "*** YOUR CODE HERE ***"
            if not hitsWall:
                nextPosition = (nextx, nexty)
                nextCornersImage = state[1]
                nextState = [nextPosition, nextCornersImage]
                # if the next position is a corner's position then update nextState
                # to contain appropriate cornersImage value
                if nextPosition in self.corners:
                    self.updateCornersImage(nextState)

                stepCost = 1
                successors.append((nextState, action, stepCost))

        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """
    Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999.  This is implemented for you.
    """
        if actions == None: return 999999
        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)

    def updateCornersImage(self, state):
        # get the index of the corner in self.corners that we just visited or about to visit
        i = self.corners.index(state[0])
        # change the corners image
        # for example , if state = [(1, 1), 0] then i = 0 and
        # we get : state[1] = 0 logical_or ( 2^(3-0)) -> 0 logical_or 8 = 8 which is 1000 in binary
        state[1] = state[1] | (1 << (MAX_BINARY_MSD_POSITION - i))


NOT_IN_DICT = -1


def cornersHeuristic(state, problem):
    """
  A heuristic for the CornersProblem that you defined.

    state:   The current search state 
             (a data structure you chose in your search problem)
    
    problem: The CornersProblem instance for this layout.

  Short explanation:
  ------------------
  Any path, and also the optimal path from a start state to a goal state
  must be at least the minimum sum of manhattan distances
  for all possible options we can take at a given moment, where options
  are the corners we didn't visited yet.
  For 0 <= i <= 4 options,  there're i! paths like the one I described above.
  So we can compute all possibilities, (4! + 3! + 2! + 1!+ 0! *total* possibilities)
  for any state, we can then take the minimum value we got from the computation.
  For each permutation, we calculate the related manhattan distances and take the minimum value.
  This value is the output of this heuristic function.

  """
    from itertools import permutations
    # if state[1] == 15 (1111 in binary) then we reached all the corners
    if state[1] == GOAL_VALUE:
        return 0

    corners = problem.corners  # These are the corner coordinates
    cornersImage = state[1]
    IndexesOfCornersToExplore = []  # a non-negative number i belong to this set if we didn't yet explore corners[i]

    for i in range(4):
        if not cornersImage & (1 << (4 - i - 1)):  # if we didn't yet visit corners[i]
            IndexesOfCornersToExplore.append(i)

    # get an iterator for a list of tuples such that each tuple is a permutation
    # of indexes we didn't yet explore
    # for example , if we didn't visit corners[0], and corners [3]
    # then permutation will be equal to [(0, 3), (3, 0)]
    permutations = permutations(IndexesOfCornersToExplore)

    return minDistancePermutations(state, permutations, corners)


def manhattanDistance(currPosition, target):
    return abs(currPosition[0] - target[0]) + abs(currPosition[1] - target[1])


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem


class FoodSearchProblem:
    """
  A search problem associated with finding the a path that collects all of the 
  food (dots) in a Pacman game.
  
  A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
    pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
    foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food 
  """

    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {}  # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0  # returns True if ate all dots, False otherwise

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()  # copy the current grid of food for this state
                nextFood[nextx][nexty] = False  # set the grid to False at entry of the next position
                successors.append((((nextx, nexty), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def foodHeuristic(state, problem):
    """
  Your heuristic for the FoodSearchProblem goes here.
  
  This heuristic must be consistent to ensure correctness.  First, try to come up
  with an admissible heuristic; almost all admissible heuristics will be consistent
  as well.
  
  If using A* ever finds a solution that is worse uniform cost search finds,
  your heuristic is *not* consistent, and probably not admissible!  On the other hand,
  inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
  
  The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a 
  Grid (see game.py) of either True or False. You can call foodGrid.asList()
  to get a list of food coordinates instead.
  
  If you want access to info like walls, capsules, etc., you can query the problem.
  For example, problem.walls gives you a Grid of where the walls are.
  
  If you want to *store* information to be reused in other calls to the heuristic,
  there is a dictionary called problem.heuristicInfo that you can use. For example,
  if you only want to count the walls once and store that value, try:
    problem.heuristicInfo['wallCount'] = problem.walls.count()
  Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']

  Short Explanation:
  -----------
  Similar idea to one I used for cornersHeuristic - use permutations.
  This time, Build a set of the farthest food positions from the current postion of pacman
  and compute all the possible options to take a path to each of food's positions, in incremental manner.
  If pacman position is (3, 1) and the set of food positions is {(1,1), (5, 1)}
  then compute manhattan distance from (3,1) to (1,1) and then sum the result to the
  manhattan distance of (1,1) to (5,1)
  no take a new sum :
  compute manhattan distance from (3,1) to (5,1) and then sum the result to the
  manhattan distance of (1,1) to (1,1)  then take the minimum value between all the sums you computed
  and return this value.

  The size of the set in the wrost case:
  There're N dots, so we have N! permutations to calculate
  from my implementation, in the worst case N = 2 * min(height of maze, width of maze)
  """
    position, foodGrid = state  # decouples the tuple (state) to position and foodGrid
    "*** YOUR CODE HERE ***"

    from itertools import permutations

    top, right = foodGrid.height, foodGrid.width
    # will contain positions of the farthest food dots from the current position of pacman
    simpleFood = set()
    # False if we should scan farthest dots according to columns. Initial value is True
    scanDotsAccordingToRows = True
    # determine whether to scan outer dots according to rows or according to columns
    # the indication for what to choose is the minimum value between the height and width of the maze
    if top < right:
        rangeLimit = top  # loop from 0 to rangeLimit - 1 in the for loop below
        end = right - 1  # end is the number of columns - 1
        # a is the identity element with respect to the multiplication operator
        # b will zero an expression that represents an index to index foodGrid
        # the purpose of a and b is do decide how to index foodGrip
        # if a = 1 and b = 0 then we will index foodGrid such that the rows are fixed with
        # respect to each iteration of the for loop
        a, b = 1, 0
    else:
        rangeLimit = right
        end = top - 1  # end is t he number of rows - 1
        a, b = 0, 1  # symmetric to what I wrote above
        scanDotsAccordingToRows = False

    for i in range(rangeLimit):
        currIndex, currBound = 0, end
        if scanDotsAccordingToRows:
            # go right while we can go right in the current row and
            # the current position does not contains food
            while currIndex <= currBound and not foodGrid[currIndex * a + i * b][i * a + currIndex * b]:
                currIndex += 1
            # symmetric in action to the previous while loop
            while currIndex <= currBound and not foodGrid[currBound * a + i * b][i * a + currBound * b]:
                currBound -= 1
            # if found food, add the farthest food dots (if they're the same, it's ok
            # since simpleFood is a set)
            if currIndex <= currBound:
                simpleFood.add((currIndex * a + i * b, i * a + currIndex * b))
                simpleFood.add((currBound * a + i * b, i * a + currBound * b))

    simpleFoodLength = len(simpleFood)
    # if pacman ate all the food
    if simpleFoodLength == 0:
        return 0
    # keep a list of indexes in the range of simpleFood's indexes
    indexes = []
    for i in range(simpleFoodLength):
        indexes.append(i)
    # get an iterator of tuples s.t each tuple is a permutation of simpleFood's indexes
    indexesPermutations = permutations(indexes)
    # transform the set to a list to be able to index the list
    simpleFood = list(simpleFood)
    # compute the sum of manhattan distances form current pacman position to all
    # the food positions according to the indexes permutations
    # to this for each permutation and finally return the minimum sum that
    # computed for each permutation
    return minDistancePermutations(state, indexesPermutations, simpleFood)


class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"

    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        "Returns a path (a list of actions) to the closest dot, starting from gameState"
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        # using BFS to find to nearest food position
        from util import Queue
        from search import graphSearch
        fifoQueue = Queue()
        return graphSearch(problem, fifoQueue)


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.
    
    This search problem is just like the PositionSearchProblem, but
    has a different goal test, which you need to fill in below.  The
    state space and successor function do not need to be changed.
    
    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.
    
    You can use this search problem to help you fill in 
    the findPathToClosestDot method.
  """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        """
    The state is Pacman's position. Fill this in with a goal test
    that will complete the problem definition.
    """
        x, y = state

        "*** YOUR CODE HERE ***"
        # returns true if the current position of pacman is also a food position
        # returns false otherwise
        return self.food[x][y] is True


##################
# Mini-contest 1 #
##################

class ApproximateSearchAgent(Agent):
    "Implement your contest entry here.  Change anything but the class name."

    def registerInitialState(self, state):
        "This method is called before any moves are made."
        "*** YOUR CODE HERE ***"

    def getAction(self, state):
        """
    From game.py: 
    The Agent will receive a GameState and must return an action from 
    Directions.{North, South, East, West, Stop}
    """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def mazeDistance(point1, point2, gameState):
    """
  Returns the maze distance between any two points, using the search functions
  you have already built.  The gameState can be any game state -- Pacman's position
  in that state is ignored.
  
  Example usage: mazeDistance( (2,4), (5,6), gameState)
  
  This might be a useful helper function for your ApproximateSearchAgent.
  """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
    return len(search.bfs(prob))


def minDistancePermutations(state, permutations, positionsContainer):
    """
    param : state - the current state from the state space of the problem
    param : permutations - an iterator for a list of tuples. Each tuple contains a variable number
      of elements, each element is an index that will be used to index positionsContainer
        for example : an iterator for this list - [(0, 1)(1, 0)]
    param : positionsContainer - a data structure that contains coordinates
        for example : [(2, 4), (6, 1)]
    """
    minSumOfManhattanDistances = 999999  # maximum value possible for any instances of problems in this project
    for perm in permutations:
        sum = 0  # calculate a new sum for each permutation
        source = state[0]  # the initial position is always the current position of pacman
        # for each element (an index) in the permutation's tuple
        for i in perm:
            #  calculate the manhattan distance from source to target (positionsContainer[i])
            md = manhattanDistance(source, positionsContainer[i])
            #  add to the current sum the previous manhattan distance
            sum = sum + md
            #  the new source will be the current target
            source = positionsContainer[i]
        # if the current sum of manhattan distances for the current permutation
        # is smaller than the minimum sum that computed so far
        # then update the minimum sum the to the current sum
        if sum < minSumOfManhattanDistances:
            minSumOfManhattanDistances = sum

    return minSumOfManhattanDistances
