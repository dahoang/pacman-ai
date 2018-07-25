# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # closest ghost distance
        ghostDistances = []
        distanceClosestGhost = -1000000
        for ghost in newGhostStates:
            ghostDistances.append( (util.manhattanDistance(newPos, ghost.getPosition())) )
        if min ( ghostDistances ) == 0:
            return distanceClosestGhost
        else:
            distanceClosestGhost = min ( ghostDistances )

        # closest food distance
        foodList = newFood.asList()
        foodDistances = []
        distanceClosestFood = 0
        if len(foodList) > 0:
            for food in foodList:
                foodDistances.append( (util.manhattanDistance(newPos, food)) )
            distanceClosestFood = min ( foodDistances )

        # Negative because inverse relationship of distance is wanted for food, but
        # it is the opposite for ghost ( -1/ghostDistance format is needed ).
        # It is also necessary for the weight of the foodlist to be large; if it is too
        # small, Pacman will be too scared of the ghosts.
        score = -distanceClosestFood - 5/distanceClosestGhost - 30*len(foodList)

        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        valueActionList = [(float("-inf"), None)] # Tuple of (value, action)
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.ghostMin(successor, 1, self.depth)
            valueActionList.append( (value, action) )

        # This is the top of the tree
        pacmanAction = max(valueActionList)[1]
        return pacmanAction

    def ghostMin(self, gameState, ghostIndex, depth):
        ghostActions = gameState.getLegalActions(ghostIndex)
        lastGhost = gameState.getNumAgents() - 1
        valueList = [float("inf")] # valueList stores scores returned from ghosts to minimize

        if depth == 0 or len(ghostActions) == 0:
            return self.evaluationFunction(gameState)

        for action in ghostActions:
            successor = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == lastGhost:
                # Go down one depth and restart the process
                value = self.pacmanMax(successor, depth - 1)
            else:
                # Move on to the next ghost
                value = self.ghostMin(successor, ghostIndex + 1, depth)
            valueList.append(value)

        minValue = min(valueList)
        return minValue

    def pacmanMax(self, gameState, depth):
        pacmanActions = gameState.getLegalActions(0)
        valueList = [float("-inf")]

        if depth == 0 or len(pacmanActions) == 0:
            return self.evaluationFunction(gameState)

        for action in pacmanActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.ghostMin(successor, 1, depth)
            valueList.append(value)

        # Get the max out of the ghostMin values below it.
        maxValue = max(valueList)
        return maxValue

class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        pacmanAction = (float("-inf"), None)
        valueActionList = [pacmanAction]  # Tuple of (value, action)
        alpha = float("-inf")
        beta = float("inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.ghostMin(successor, 1, self.depth, alpha, beta)
            valueActionList.append((value, action))
            pacmanAction = max(valueActionList)

            if pacmanAction[0] > beta: # if max value > beta
                return pacmanAction[1] # return best action
            alpha = max(alpha, pacmanAction[0])

        # This is the top of the tree
        return pacmanAction[1]

    def ghostMin(self, gameState, ghostIndex, depth, alpha, beta):
        ghostActions = gameState.getLegalActions(ghostIndex)
        lastGhost = gameState.getNumAgents() - 1
        valueList = [float("inf")]  # valueList stores scores returned from ghosts to minimize

        if depth == 0 or len(ghostActions) == 0:
            return self.evaluationFunction(gameState)

        for action in ghostActions:
            successor = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == lastGhost:
                # Go down one depth and restart the process
                value = self.pacmanMax(successor, depth - 1, alpha, beta)
                valueList.append(value)
                if min(valueList) < alpha:
                    return min(valueList)
                beta = min(beta, min(valueList))
            else:
                # Move on to the next ghost
                value = self.ghostMin(successor, ghostIndex + 1, depth, alpha, beta)
                valueList.append(value)
                if min(valueList) < alpha:
                    return min(valueList)
                beta = min(beta, min(valueList))

        return min(valueList)

    def pacmanMax(self, gameState, depth, alpha, beta):
        pacmanActions = gameState.getLegalActions(0)
        valueList = [float("-inf")]
        # value = float("-inf")

        if depth == 0 or len(pacmanActions) == 0:
            return self.evaluationFunction(gameState)

        for action in pacmanActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.ghostMin(successor, 1, depth, alpha, beta)
            valueList.append(value)
            if max(valueList) > beta:
                return max(valueList)
            alpha = max(alpha, max(valueList))

        # Get the max out of the ghostMin values below it.
        # maxValue = max(valueList)
        return max(valueList)

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
        # Same as MinimaxAgent but uses avg value when calculating ghost values

        legalActions = gameState.getLegalActions(0)
        valueActionList = [(float("-inf"), None)]  # Tuple of (value, action)
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.ghostAvg(successor, 1, self.depth)
            valueActionList.append((value, action))

        # This is the top of the tree
        pacmanAction = max(valueActionList)[1]
        return pacmanAction

    def ghostAvg(self, gameState, ghostIndex, depth):
        ghostActions = gameState.getLegalActions(ghostIndex)
        lastGhost = gameState.getNumAgents() - 1
        avgValue = 0
        actionCount = 0

        if depth == 0 or len(ghostActions) == 0:
            return self.evaluationFunction(gameState)

        for action in ghostActions:
            successor = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == lastGhost:
                # Go down one depth and restart the process
                value = self.pacmanMax(successor, depth - 1)
            else:
                # Move on to the next ghost
                value = self.ghostAvg(successor, ghostIndex + 1, depth)
            avgValue += value
            actionCount += 1

        avgValue /= actionCount
        return avgValue

    def pacmanMax(self, gameState, depth):
        pacmanActions = gameState.getLegalActions(0)
        valueList = [float("-inf")]

        if depth == 0 or len(pacmanActions) == 0:
            return self.evaluationFunction(gameState)

        for action in pacmanActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.ghostAvg(successor, 1, depth)
            valueList.append(value)

        # Get the max out of the ghostMin values below it.
        maxValue = max(valueList)
        return maxValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Evaluates the current game state, distance to the closest ghost,
                   distance to the closest food, and number of food pellets
                   and applies an inverse relationship in the score
    """
    # Useful information you can extract from a GameState (pacman.py)
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    "*** YOUR CODE HERE ***"
    # closest ghost distance
    ghostDistances = []
    distanceClosestGhost = float("-inf")
    for ghost in ghostStates:
        ghostDistances.append((util.manhattanDistance(position, ghost.getPosition())))
    if min(ghostDistances) == 0:
        return distanceClosestGhost
    else:
        distanceClosestGhost = min(ghostDistances)

    # closest food distance
    foodList = food.asList()
    foodDistances = []
    distanceClosestFood = 0
    if len(foodList) > 0:
        for food in foodList:
            foodDistances.append((util.manhattanDistance(position, food)))
        distanceClosestFood = min(foodDistances)

    # Negative because inverse relationship of distance is wanted for food, but
    # it is the opposite for ghost ( -1/ghostDistance format is needed ).
    # It is also necessary for the weight of the foodlist to be large; if it is too
    # small, Pacman will be too scared of the ghosts.
    score = scoreEvaluationFunction(currentGameState)
    score = score - distanceClosestFood - 4 / distanceClosestGhost - 30 * len(foodList)

    return score

# Abbreviation
better = betterEvaluationFunction

