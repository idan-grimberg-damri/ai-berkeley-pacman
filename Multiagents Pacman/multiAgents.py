# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""

maman 13
introduction to AI
"""
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


# returns a list of available food positions
def getDots(oldFood):
    dots = []
    for i in range(oldFood.width):
        for j in range(oldFood.height):
            if oldFood[i][j]:
                dots.append((i, j))
    return dots


# returns the minimum manhattan distance between newPos to all the positions in positions
def getMinManhattanDistance(newPos, positions):
    min = 999999
    md = 0
    for pos in positions:
        md = manhattanDistance(newPos, pos)
        if md < min:
            min = md
    return md


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

    def getAction(self, gameState):
        """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
        # Collect legal moves and successor states
        # this value can be for example : ['Stop', 'East', 'North']
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are BETTER.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        oldCapsules = currentGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        # don't want a tackle between ghosts and pacman
        for i in range(len(newGhostStates)):
            nextGhostPos = successorGameState.getGhostPosition(i + 1)
            if newScaredTimes[i] == 0 and manhattanDistance(newPos, nextGhostPos) < 2:
                return -999999
        # I reused this value since it's good for not getting pacman get stuck in a position
        nextGameScore = successorGameState.getScore()
        if oldFood:
            # I choose the md (manhattan distance) to the closest dot
            # to maintain useful progress as much as possible (or give a local goal to pacman)
            # if the md is low then this is good for pacman
            # but the this is a hill climbing with global maximum algorithm so
            # I multiplied the md by -1
            nextGameScore = nextGameScore - getMinManhattanDistance(newPos, getDots(oldFood))
        if oldCapsules:
            # I got low scores if I didn't multiply buy a low integer greater then 1
            # and less then or equal to the number of capsules
            # so I gave the capsules less significance then the closest dot and
            # I choose the value 1.5 since it gave good results
            nextGameScore = nextGameScore - 1.5 * getMinManhattanDistance(newPos, oldCapsules)

        return nextGameScore


def manhattanDistance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

"""
param: obj -  instance of MultiAgentSearchAgent
param: gameState - the current game state
returns: utility evaluation of the current gameState

Explanation:
If I used only the default evaluation function then sometimes pacman got stuck between two actions,
therefore I used a random value from {0, 1} to add to the default evaluation, in this case, if pacman
get stuck, he will get out of this situation very quickly (in most cases)
I also used a motviation value for pacman which is the minimum manhattan distance between pacman's current
position to the closest food. Small value is better therefore I did a subtraction operation 
"""
def myEvaluation(obj, gameState):
    pacPos = gameState.getPacmanPosition()
    return obj.evaluationFunction(gameState) + random.choice([0, 1]) - getMinManhattanDistance(pacPos, getDots(gameState.getFood()))



class MinimaxAgent(MultiAgentSearchAgent):
    """
         Your minimax agent (question 2)

         who's turn is it?
             ( depth ) modulo ( number of agents ) == 0 -> Max , else, Min


        the depth is changed only if all the entities finished their action
        and this is because all the entities do actions simultaneously
        so if we finished some sequence of actions:
          action_0(pacman)->action_1(ghost_1)->...->action_n(gohst_n)
        then next we will be at this transition : action(ghost_n) -> firstAction(pacman)
        the level in the minimax tree will increase in this case since it's represents
        a transition to a new state such that in new state all the entities are positioned
        according to their actions in the sequence above, that means, we're transitioning to a new state
        in the state space of the current game

        this is in contrast for example to TIC Tac Toe where there're explicit turns to each entity


        the depth is changed in a non-increasing manner starting from the value "depth" of self.
        we stop the progress of game states when the depth reaches to zero.


  """

    def getAction(self, gameState):

        """
              Returns the minimax action from the current gameState using self.depth
              and self.evaluationFunction.

            """
        "*** YOUR CODE HERE ***"

        currAgentIndex = 0  # pacman
        nextAgentIndex = 1
        chosenUtility = -999999  # global minimum

        legalActions = set(gameState.getLegalActions(currAgentIndex)) - {'Stop'}
        #  iterate over all possible actions and get the maximum utility
        for action in legalActions:
            successor = gameState.generateSuccessor(currAgentIndex, action)
            utility = self.minValue(successor, self.depth, nextAgentIndex)
            if chosenUtility < utility:
                chosenUtility = utility
                chosenAction = action

        return chosenAction

    #  inspired from the book's related procedure
    def minValue(self, gameState, depth, currAgentIndex):

        # if we reached the depth limit or
        # gameState is a terminal state then return the evaluation function for this state
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return myEvaluation(self, gameState)


        numOfAgents = gameState.getNumAgents()
        nextAgentIndex = (1 + currAgentIndex) % numOfAgents
        chosenUtility = 999999  # global maximum
        legalActions = gameState.getLegalActions(currAgentIndex)
        # get the minimum utility (smaller values are best for ghosts)
        for action in legalActions:
            successor = gameState.generateSuccessor(currAgentIndex, action)
            if nextAgentIndex == 0:  # pacman's turn is next
                utility = self.maxValue(successor, depth - 1)
            else:  # another ghost's turns is next
                utility = self.minValue(successor, depth, nextAgentIndex)

            if utility < chosenUtility:
                chosenUtility = utility

        return chosenUtility

    #  inspired from the book's related procedure
    def maxValue(self, gameState, depth):

        # if we reached the depth limit or
        # gameState is a terminal state then return the evaluation function for this state
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return myEvaluation(self, gameState)

        currAgentIndex = 0
        nextAgentIndex = 1
        chosenUtility = -999999  # global minimum

        legalActions = set(gameState.getLegalActions(currAgentIndex)) - {'Stop'}

        for action in legalActions:
            successor = gameState.generateSuccessor(currAgentIndex, action)
            utility = self.minValue(successor, depth, nextAgentIndex)
            if chosenUtility < utility:
                chosenUtility = utility

        return chosenUtility


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)

    extending the procedures minValue and maxValue above

    if ghost1 calls minValueAlphaBeta such that ghosts2 is next then we just take the minimum
    value between ghosts1 beta value do the value returned from ghosts2 turn

    the rest is relaying on the book (page 170) and the extension of minValue and maxValue above
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
        "*** YOUR CODE HERE ***"
        alpha = -999999
        beta = -alpha
        currAgentIndex = 0
        nextAgentIndex = 1

        legalActions = set(gameState.getLegalActions(currAgentIndex)) - {'Stop'}
        for action in legalActions:
            successor = gameState.generateSuccessor(currAgentIndex, action)
            utility = self.minValueAlphaBeta(successor, alpha, beta, self.depth, nextAgentIndex)
            if alpha < utility:
                alpha = utility
                chosenAction = action

        return chosenAction

    def maxValueAlphaBeta(self, gameState, alpha, beta, depth):

        if depth == 0 or gameState.isWin() or gameState.isLose():
            return myEvaluation(self, gameState)

        currAgentIndex = 0
        nextAgentIndex = 1

        legalActions = set(gameState.getLegalActions(currAgentIndex)) - {'Stop'}

        for action in legalActions:
            successor = gameState.generateSuccessor(currAgentIndex, action)
            utility = self.minValueAlphaBeta(successor, alpha, beta, depth, nextAgentIndex)
            if beta <= utility:
                return utility
            if alpha < utility:
                alpha = utility

        return alpha

    def minValueAlphaBeta(self, gameState, alpha, beta, depth, currAgentIndex):

        if depth == 0 or gameState.isWin() or gameState.isLose():
            return myEvaluation(self, gameState)

        nextAgentIndex = (1 + currAgentIndex) % gameState.getNumAgents()

        legalActions = gameState.getLegalActions(currAgentIndex)

        for action in legalActions:
            successor = gameState.generateSuccessor(currAgentIndex, action)
            if nextAgentIndex == 0:
                utility = self.maxValueAlphaBeta(successor, alpha, beta, depth - 1)
            else:
                utility = self.minValueAlphaBeta(successor, alpha, beta, depth, nextAgentIndex)

            if utility <= alpha:
                return utility
            if utility < beta:
                beta = utility

        return beta


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
        "*** YOUR CODE HERE ***"
        currAgentIndex = 0  # pacman
        nextAgentIndex = 1
        chosenUtility = -999999  # global minimum

        legalActions = set(gameState.getLegalActions(currAgentIndex)) - {'Stop'}
        #  iterate over all possible actions and get the maximum utility
        for action in legalActions:
            successor = gameState.generateSuccessor(currAgentIndex, action)
            utility = self.expectiMax(successor, self.depth, nextAgentIndex)
            if chosenUtility < utility:
                chosenUtility = utility
                chosenAction = action

        return chosenAction

    # used simple average calculation
    def expectiMax(self, gameState, depth, currAgentIndex):
        # if we reached the depth limit or
        # gameState is a terminal state then return the evaluation function for this state
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return myEvaluation(self, gameState)

        numOfAgents = gameState.getNumAgents()
        nextAgentIndex = (1 + currAgentIndex) % numOfAgents
        legalActions = gameState.getLegalActions(currAgentIndex)
        currSum = 0
        numOfEvaluations = 0

        for action in legalActions:
            successor = gameState.generateSuccessor(currAgentIndex, action)
            if nextAgentIndex == 0:  # pacman's turn is next
                currSum = currSum + self.expectiMax(successor, depth - 1, nextAgentIndex)
            else:  # another ghost's turns is next
                currSum = currSum + self.expectiMax(successor, depth, nextAgentIndex)

            numOfEvaluations = numOfEvaluations + 1

        return float(float(currSum) / numOfEvaluations)







def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
  """

    def getAction(self, gameState):
        """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
