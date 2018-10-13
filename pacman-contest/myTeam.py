# myTeam.py
# ---------
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

from captureAgents import CaptureAgent
from baselineTeam import ReflexCaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
import copy
from util import nearestPoint
from game import Actions

'''
Michael Rinaldi, Phaelyn Kotuby, Calvin Pomerantz

Leeroy Agents
Time's up, let's do this

Agents that were originally meant to take the most efficient route possible
One agent that values pellets with a higher y coordinate more
One agent that values pellets with a lower y coordinate more
Eventually they will converge in the middle

We ended up using many tactics that involve calculating enemy ghost positions,
for both offense and defense
'''

DEBUG = True
DEFENSE_TIMER_MAX = 100.0
USE_BELIEF_DISTANCE = True
arguments = {}

MINIMUM_PROBABILITY = .0001
PANIC_TIME = 200
CHEAT = False
beliefs = []
beliefsInitialized = []
FORWARD_LOOKING_LOOPS = 1


def createTeam(firstIndex, secondIndex, isRed,
               first='LeeroyTopAgent', second='LeeroyBottomAgent', **args):
    if 'numTraining' in args:
        arguments['numTraining'] = args['numTraining']
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


# LeeroyAgent inherits from ApproximateQAgent
# Our agent does worse with training, because we did not focus on training
class ApproximateQAgent(CaptureAgent):

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.numTraining = 0
        # if 'numTraining' in arguments:
        # self.numTraining = arguments['numTraining']
        self.episodesSoFar = 0
        self.epsilon = 0.05
        self.discount = 0.8
        self.alpha = 0.2

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.lastAction = None
        CaptureAgent.registerInitialState(self, gameState)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # def chooseAction(self, state):
    #     # Append game state to observation history...
    #     self.observationHistory.append(state)
    #     # Pick Action
    #     legalActions = state.getLegalActions(self.index)
    #     action = None
    #     if (DEBUG):
    #         print self.newline()
    #         print "AGENT " + str(self.index) + " choosing action!"
    #     if len(legalActions):
    #         if util.flipCoin(self.epsilon) and self.isTraining():
    #             action = random.choice(legalActions)
    #             if (DEBUG):
    #                 print "ACTION CHOSE FROM RANDOM: " + action
    #         else:
    #             action = self.computeActionFromQValues(state)
    #             if (DEBUG):
    #                 print "ACTION CHOSE FROM Q VALUES: " + action
    #
    #     self.lastAction = action
    #
    #     foodLeft = len(self.getFood(state).asList())
    #     # Prioritize going back to start if we have <= 2 pellets left
    #     if foodLeft <= 2:
    #         bestDist = 9999
    #         for a in legalActions:
    #             successor = self.getSuccessor(state, a)
    #             pos2 = successor.getAgentPosition(self.index)
    #             dist = self.getMazeDistance(self.start, pos2)
    #             if dist < bestDist:
    #                 action = a
    #                 bestDist = dist
    #
    #     if (DEBUG):
    #         print "AGENT " + str(self.index) + " chose action " + action + "!"
    #     return action
    def isHomeSide(self, node, gameState):
        width, height = gameState.data.layout.width, gameState.data.layout.height
        if self.index in [0, 2]:
            if node[0] < width / 2:
                return True
            else:
                return False
        else:
            if node[0] > width / 2 - 1:
                return True
            else:
                return False

    def initialLegalAction(self, gameState):
        legalAction = {}
        corners = []
        walls = copy.deepcopy(gameState.getWalls())
        for x in range(0, gameState.data.layout.width):
            for y in range(0, gameState.data.layout.height):
                if not gameState.hasWall(x, y):
                    possible = []
                    adjacentToConers = 0
                    for action, position in Actions._directionsAsList:
                        px, py = position
                        ny = py + y
                        nx = px + x
                        if not walls[nx][ny]: possible.append(((nx, ny), action, 1))
                    legalAction[(x, y)] = possible
                    if len(possible) <= 2:
                        corners.append((x, y))
        return legalAction, corners

    def aStarSearch(self, gameState, myPos, minDist):
        """Search the node that has the lowest combined cost and heuristic first."""
        explored, tempCost, Q = [], [], util.PriorityQueue()
        explored.append(myPos)
        tempCost.append(0)
        Q.push([myPos, []], minDist)
        while not Q.isEmpty():
            node, path = Q.pop()
            pathCost = tempCost[explored.index(node)]
            if self.isHomeSide(node, gameState):
                return path
            for successor in self.legalAction[node]:
                state, action, cost = successor
                if state not in explored:
                    childDist = float('inf')
                    explored.append(state)
                    tempCost.append(childDist)
                else:
                    childDist = tempCost[explored.index(state)]
                if pathCost + cost < childDist:
                    childDist = pathCost + cost
                    Q.update([state, path + [action]], childDist + minDist)
                    tempCost[explored.index(state)] = childDist
        return []

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        successor = self.getSuccessor(gameState, action)
        features = util.Counter()
        features['score'] = self.getScore(successor)
        if not self.red:
            features['score'] *= -1
        features['choices'] = len(successor.getLegalActions(self.index))
        return features

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        bestValue = -999999
        bestActions = None
        for action in state.getLegalActions(self.index):
            # For each action, if that action is the best then
            # update bestValue and update bestActions to be
            # a list containing only that action.
            # If the action is tied for best, then add it to
            # the list of actions with the best value.
            value = self.getQValue(state, action)
            if (DEBUG):
                print "ACTION: " + action + "           QVALUE: " + str(value)
            if value > bestValue:
                bestActions = [action]
                bestValue = value
            elif value == bestValue:
                bestActions.append(action)
        if bestActions == None:
            return Directions.STOP  # If no legal actions return None
        return random.choice(bestActions)  # Else choose one of the best actions randomly

    def getWeights(self):
        return self.weights

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        bestValue = -999999
        noLegalActions = True
        for action in state.getLegalActions(self.index):
            # For each action, if that action is the best then
            # update bestValue
            noLegalActions = False
            value = self.getQValue(state, action)
            if value > bestValue:
                bestValue = value
        if noLegalActions:
            return 0  # If there is no legal action return 0
        # Otherwise return the best value found
        return bestValue

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        total = 0
        weights = self.getWeights()
        features = self.getFeatures(state, action)
        for feature in features:
            # Implements the Q calculation
            total += features[feature] * weights[feature]
        return total

    def getReward(self, gameState):
        foodList = self.getFood(gameState).asList()
        return -len(foodList)

    def observationFunction(self, gameState):
        if len(self.observationHistory) > 0 and self.isTraining():
            self.update(self.getCurrentObservation(), self.lastAction, gameState, self.getReward(gameState))
        return gameState.makeObservation(self.index)

    def isTraining(self):
        return self.episodesSoFar < self.numTraining

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        if (DEBUG):
            print self.newline()
            print "AGENT " + str(self.index) + " updating weights!"
            print "Q VALUE FOR NEXT STATE: " + str(self.computeValueFromQValues(nextState))
            print "Q VALUE FOR CURRENT STATE: " + str(self.getQValue(state, action))
        difference = (reward + self.discount * self.computeValueFromQValues(nextState))
        difference -= self.getQValue(state, action)
        # Only calculate the difference once, not in the loop.
        newWeights = self.weights.copy()
        # Same with weights and features.
        features = self.getFeatures(state, action)
        for feature in features:
            # Implements the weight updating calculations
            newWeight = newWeights[feature] + self.alpha * difference * features[feature]
            if (DEBUG):
                print "AGENT " + str(self.index) + " weights for " + feature + ": " + str(
                    newWeights[feature]) + " ---> " + str(newWeight)
            newWeights[feature] = newWeight
        self.weights = newWeights.copy()
        # print "WEIGHTS AFTER UPDATE"
        # print self.weights

    def newline(self):
        return "-------------------------------------------------------------------------"

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.final(self, state)
        if self.isTraining() and DEBUG:
            print "END WEIGHTS"
            print self.weights
        self.episodesSoFar += 1
        if self.episodesSoFar == self.numTraining:
            print "FINISHED TRAINING"


'''
The REAL LeeroyCaptureAgent
Has many weights
'successorScore' - we highly value eating our pellets
'leeroyDistanceToFood' - we slightly value moving closer to a certain pellet
    Remember, we use y-based calculations for leeroyDistance.
'ghostDistance' - We value moving away from enemy non-scared ghosts
'stop' - we highly penalize stopping
'legalActions' - we highly value moving to spaces that have more possible actions
'powerPelletValue' - we highly value eating a power pellet
'backToSafeZone' - sometimes, we highly value moving back to your side of the field
    For instance, when cashing in pellets, or when being chased by a ghost
'chaseEnemyValue' - if we are close enough to an enemy Pacman, we highly value chasing it
'''


class LeeroyCaptureAgent(ApproximateQAgent):

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)
        self.favoredY = 0.0
        self.defenseTimer = 0.0
        self.lastNumReturnedPellets = 0.0
        self.getLegalPositions(gameState)
        self.originalFood = len(self.getFood(gameState).asList())

    def __init__(self, index):
        self.actionHistory=[]
        self.count=0
        ApproximateQAgent.__init__(self, index)
        self.weights = util.Counter()
        self.weights['successorScore'] = 100
        self.weights['leeroyDistanceToFood'] = -1
        self.weights['ghostDistance'] = 5
        self.weights['stop'] = -1000
        self.weights['legalActions'] = 100
        self.weights['powerPelletValue'] = 100
        self.distanceToTrackPowerPelletValue = 3
        self.weights['backToSafeZone'] = -1
        self.minPelletsToCashIn = 8
        self.weights['chaseEnemyValue'] = -100
        self.chaseEnemyDistance = 5
        self.threatenedDistance = 5
        # dictionary of (position) -> [action, ...]
        # populated as we go along; to use this, call self.getLegalActions(gameState)
        self.legalActionMap = {}
        self.legalPositionsInitialized = False
        if DEBUG:
            print "INITIAL WEIGHTS"
            print self.weights

    def getWinningBy(self, gameState):
        if self.red:
            return gameState.getScore()
        else:
            return -1 * gameState.getScore()

    def getLegalPositions(self, gameState):
        if not self.legalPositionsInitialized:
            self.legalPositions = []
            walls = gameState.getWalls()
            for x in range(walls.width):
                for y in range(walls.height):
                    if not walls[x][y]:
                        self.legalPositions.append((x, y))
            self.legalPositionsInitialized = True
        return self.legalPositions

    def getLegalActions(self, gameState):
        """
        legal action getter that favors
        returns list of legal actions for Pacman in the given state
        """
        currentPos = gameState.getAgentState(self.index).getPosition()
        if currentPos not in self.legalActionMap:
            self.legalActionMap[currentPos] = gameState.getLegalActions(self.index)
        return self.legalActionMap[currentPos]

    # If we are near the end of the game, we determine if we should just cash in pellets
    def shouldRunHome(self, gameState):
        winningBy = self.getWinningBy(gameState)
        numCarrying = gameState.getAgentState(self.index).numCarrying
        return (gameState.data.timeleft < PANIC_TIME
                and winningBy <= 0
                and numCarrying > 0
                and numCarrying >= abs(winningBy))

    def getFeatures(self, gameState, action):
        self.observeAllOpponents(gameState)
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)

        # Compute distance to the nearest food
        # uses leeroy distance so its prioritizes either top or bottom food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            leeroyDistance = min([self.getLeeroyDistance(myPos, food) for food in foodList])
            features['leeroyDistanceToFood'] = leeroyDistance

        # Grab all enemies
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemyPacmen = [a for a in enemies if a.isPacman and a.getPosition() != None]
        nonScaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
        scaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]

        # Computes distance to enemy non scared ghosts we can see
        dists = []
        for index in self.getOpponents(successor):
            enemy = successor.getAgentState(index)
            if enemy in nonScaredGhosts:
                if USE_BELIEF_DISTANCE:
                    dists.append(self.getMazeDistance(myPos, self.getMostLikelyGhostPosition(index)))
                else:
                    dists.append(self.getMazeDistance(myPos, enemy.getPosition()))
        # Use the smallest distance
        if len(dists) > 0:
            smallestDist = min(dists)
            features['ghostDistance'] = smallestDist

        features['powerPelletValue'] = self.getPowerPelletValue(myPos, successor, scaredGhosts)
        features['chaseEnemyValue'] = self.getChaseEnemyWeight(myPos, enemyPacmen)

        # If we cashed in any pellets, we shift over to defense mode for a time
        if myState.numReturned != self.lastNumReturnedPellets:
            self.defenseTimer = DEFENSE_TIMER_MAX
            self.lastNumReturnedPellets = myState.numReturned
        # If on defense, heavily value chasing after enemies
        if self.defenseTimer > 0:
            self.defenseTimer -= 1
            features['chaseEnemyValue'] *= 100

        # If our opponents ate all our food (except for 2), we rush them
        if len(self.getFoodYouAreDefending(successor).asList()) <= 2:
            features['chaseEnemyValue'] *= 100

        # Heavily prioritize not stopping
        if action == Directions.STOP:
            features['stop'] = 1

        # The total of the legalActions you can take from where you are AND
        # The legalActions you can take in all future states
        # It depends on how many loops we do
        features['legalActions'] = self.getLegalActionModifier(gameState, FORWARD_LOOKING_LOOPS)

        # Adding value for cashing in pellets
        features['backToSafeZone'] = self.getCashInValue(myPos, gameState, myState)

        # Adding value for going back home
        features['backToSafeZone'] += self.getBackToStartDistance(myPos, features['ghostDistance'])

        if self.shouldRunHome(gameState):
            features['backToSafeZone'] = self.getMazeDistance(self.start, myPos) * 10000

        return features

    def getWeights(self):
        return self.weights

    # Adds (maze distance) to (the difference in y between the food and our favored y)
    def getLeeroyDistance(self, myPos, food):
        return self.getMazeDistance(myPos, food) + abs(self.favoredY - food[1])

    # If there are not any scared ghosts, then we value eating pellets
    def getPowerPelletValue(self, myPos, successor, scaredGhosts):
        powerPellets = self.getCapsules(successor)
        minDistance = 0
        if len(powerPellets) > 0 and len(scaredGhosts) == 0:
            distances = [self.getMazeDistance(myPos, pellet) for pellet in powerPellets]
            minDistance = min(distances)
        return max(self.distanceToTrackPowerPelletValue - minDistance, 0)

    def getCashInValue(self, myPos, gameState, myState):
        # if we have enough pellets, attempt to cash in
        if myState.numCarrying >= self.minPelletsToCashIn:
            return self.getMazeDistance(self.start, myPos)
        else:
            return 0

    # If threatened, go back to start
    def getBackToStartDistance(self, myPos, smallestGhostPosition):
        if smallestGhostPosition > self.threatenedDistance or smallestGhostPosition == 0:
            return 0
        else:
            return self.getMazeDistance(self.start, myPos) * 1000

    def getChaseEnemyWeight(self, myPos, enemyPacmen):
        if len(enemyPacmen) > 0:
            # Computes distance to enemy pacmen we can see
            dists = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemyPacmen]
            # Use the smallest distance
            if len(dists) > 0:
                smallestDist = min(dists)
                return smallestDist
        return 0

    # Uses our beliefs based on the noisyDistance, and we just use the highest belief
    def getMostLikelyGhostPosition(self, ghostAgentIndex):
        return max(beliefs[ghostAgentIndex])

    # We loop over each possible legal action and tally up the possible actions from there
    def getLegalActionModifier(self, gameState, numLoops):
        legalActions = self.getLegalActions(gameState)
        numActions = len(legalActions)
        for legalAction in legalActions:
            if numLoops > 0:
                newState = self.getSuccessor(gameState, legalAction)
                numActions += self.getLegalActionModifier(newState, numLoops - 1)
        return numActions

    '''
    Beliefs section-----------------------
    '''

    def initializeBeliefs(self, gameState):
        beliefs.extend([None for x in range(len(self.getOpponents(gameState)) + len(self.getTeam(gameState)))])
        for opponent in self.getOpponents(gameState):
            self.initializeBelief(opponent, gameState)
        beliefsInitialized.append('done')

    def initializeBelief(self, opponentIndex, gameState):
        belief = util.Counter()
        for p in self.getLegalPositions(gameState):
            belief[p] = 1.0
        belief.normalize()
        beliefs[opponentIndex] = belief

    def observeAllOpponents(self, gameState):
        if len(beliefsInitialized):
            for opponent in self.getOpponents(gameState):
                self.observeOneOpponent(gameState, opponent)
        else:  # Opponent indices are different in initialize() than anywhere else for some reason
            self.initializeBeliefs(gameState)

    def observeOneOpponent(self, gameState, opponentIndex):
        pacmanPosition = gameState.getAgentPosition(self.index)
        allPossible = util.Counter()
        # We might have a definite position for the agent - if so, no need to do calcs
        maybeDefinitePosition = gameState.getAgentPosition(opponentIndex)
        if maybeDefinitePosition != None:
            allPossible[maybeDefinitePosition] = 1
            beliefs[opponentIndex] = allPossible
            return
        noisyDistance = gameState.getAgentDistances()[opponentIndex]
        for p in self.getLegalPositions(gameState):
            # For each legal ghost position, calculate distance to that ghost
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            modelProb = gameState.getDistanceProb(trueDistance,
                                                  noisyDistance)  # Find the probability of getting this noisyDistance if the ghost is at this position
            if modelProb > 0:
                # We'd like to find the probability of the ghost being at this distance
                # given that we got this noisy distance
                # p(noisy | true) = p(true | noisy) * p(true) / p(noisy)
                # p(noisy) is 1 - we know that for certain.
                # So return p(true | noisy) * p(true)
                oldProb = beliefs[opponentIndex][p]
                # Add a small constant to oldProb because a ghost may travel more than
                # 13 spaces - if that happens then we don't want to think it's prob is 0
                allPossible[p] = (oldProb + MINIMUM_PROBABILITY) * modelProb
            else:
                allPossible[p] = 0
        allPossible.normalize()
        beliefs[opponentIndex] = allPossible

    def observationFunction(self, gameState):
        # Cheats. Sneakily hidden at the bottom of the agent definition ;)
        if CHEAT:
            return gameState
        else:
            return ApproximateQAgent.observationFunction(self, gameState)
    def chooseAction(self, gameState):
        # Append game state to observation history...
        # self.observationFunction(gameState)
        self.observationHistory.append(gameState)
        self.observeAllOpponents(gameState)
        # Pick Action
        legalActions = gameState.getLegalActions(self.index)
        action = None
        enemyAgents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [enemyAgent.getPosition() for enemyAgent in enemyAgents if not enemyAgent.isPacman]
        agentPos = gameState.getAgentPosition(self.index)
        enemyIndexes = []
        for ene in self.getOpponents(gameState):
            if not gameState.getAgentState(ene).isPacman:
                enemyIndexes.append(ene)
        enemyDis = 999999
        enemyPos = None
        enemyInx=-1
        for ghost in enemyIndexes:
            ghostPos = gameState.getAgentPosition(ghost)
            if ghostPos != None:
                dis = self.getMazeDistance(agentPos, ghostPos)
                if dis < enemyDis:
                    enemyDis = dis
                    enemyPos = ghostPos
                    enemyInx = ghost

        # if (SHOW):
        #     print "AGENT " + str(self.index) + " choosing action!"
        if len(legalActions):
            if util.flipCoin(self.epsilon) and self.episodesSoFar < self.numTraining:
                action = random.choice(legalActions)
                # if (SHOW):
                #     print "ACTION CHOSE FROM RANDOM: " + action
            else:
                action = self.computeActionFromQValues(gameState)
                # if (SHOW):
                #     print "ACTION CHOSE FROM Q VALUES: " + action

        foodlist = self.getFood(gameState).asList()
        carryFood = self.originalFood - len(foodlist)
        if self.isHomeSide(gameState.getAgentPosition(self.index), gameState):
            self.originalFood = self.originalFood - carryFood
            carryFood = 0
        self.lastAction = action
        if (len(ghosts) > 0 and not self.isHomeSide(agentPos, gameState)):
            if enemyDis <= 3 and gameState.getAgentState(enemyInx).scaredTimer <= 4:
                self.refreshLegalActionAndCorners(gameState, enemyPos, 1)
                heruisticDist = self.breadthFirstSearch(agentPos, gameState)
                if heruisticDist > 0:
                    actionlist = self.aStarSearch(gameState, agentPos, heruisticDist)
                    action = actionlist[0]
                    return action
        if carryFood > 6 and enemyInx != -1 and gameState.getAgentState(enemyInx).scaredTimer <= 4:
            if len(enemyPos) > 0:
                self.refreshLegalActionAndCorners(gameState, enemyPos, 1)
            heruisticDist = self.breadthFirstSearch(agentPos, gameState)
            if heruisticDist > 0:
                actionlist = self.aStarSearch(gameState, agentPos, heruisticDist)
                action = actionlist[0]
                return action
        if len(foodlist) <= 2:
            if enemyPos!=None:
                self.refreshLegalActionAndCorners(gameState, enemyPos, 1)
            heruisticDist = self.breadthFirstSearch(agentPos, gameState)
            if heruisticDist > 0:
                actionlist = self.aStarSearch(gameState, agentPos, heruisticDist)
                action = actionlist[0]
                return action

        self.actionHistory.append(action)
        if len(self.actionHistory) > 10:
            if self.actionHistory[len(self.actionHistory) - 1] == self.actionHistory[len(self.actionHistory) - 3]:
                if self.actionHistory[len(self.actionHistory) - 1] != self.actionHistory[len(self.actionHistory) - 2]:
                    self.count += 1
                    if self.count > 3:
                        action = random.choice(legalActions)
        return action

    def breadthFirstSearch(self, myPos, gameState):
        """Search the shallowest nodes in the search tree first."""
        explored, frontier = [], util.Queue()
        explored.append(myPos)
        frontier.push([myPos, []])
        while not frontier.isEmpty():
            node, path = frontier.pop()
            if self.isHomeSide(node, gameState):
                return len(path)
            for successor in self.legalAction[node]:
                nextState, action, cost = successor
                if nextState not in explored:
                    explored.append(nextState)
                    frontier.push([nextState, path + [action]])
        return 0

    def refreshLegalActionAndCorners(self, gameState, enemyPos, n):
        legalAction = {}
        corners = []
        walls = self.enemyDomain(gameState, enemyPos, n)
        for x in range(0, gameState.data.layout.width):
            for y in range(0, gameState.data.layout.height):
                if not gameState.hasWall(x, y):
                    possible = []
                    adjacentToConers = 0
                    for action, position in Actions._directionsAsList:
                        px, py = position
                        ny = py + y
                        nx = px + x
                        if not walls[nx][ny]: possible.append(((nx, ny), action, 1))
                    legalAction[(x, y)] = possible
                    if len(possible) <= 2:
                        corners.append((x, y))
        self.legalAction = legalAction
        self.corners = corners

    def enemyDomain(self, gameState, enemyPos, n):
        x, y = enemyPos
        x = int(x)
        y = int(y)
        walls = copy.deepcopy(gameState.getWalls())
        for i in range(0, n):
            walls[x + i][y] = True
            walls[x][y + i] = True
            walls[x + i][y + i] = True
            walls[x + i][y - i] = True
            walls[x - i][y] = True
            walls[x][y - i] = True
            walls[x - i][y - i] = True
            walls[x - i][y + i] = True
        return walls

# Leeroy Top Agent - favors pellets with a higher y
class LeeroyTopAgent(LeeroyCaptureAgent):

    def registerInitialState(self, gameState):
        LeeroyCaptureAgent.registerInitialState(self, gameState)
        self.favoredY = gameState.data.layout.height


# Leeroy Bottom Agent - favors pellets with a lower y
class LeeroyBottomAgent(LeeroyCaptureAgent):

    def registerInitialState(self, gameState):
        LeeroyCaptureAgent.registerInitialState(self, gameState)
        self.favoredY = 0.0

