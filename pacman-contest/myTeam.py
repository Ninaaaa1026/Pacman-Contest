from captureAgents import CaptureAgent
import random, time, util, sys
from game import Directions
from util import nearestPoint
import copy
from game import Actions

SHOW = True
DEFENSE_TIMER_MAX = 100.0
USE_BELIEF_DISTANCE = True
arguments = {}

MINIMUM_PROBABILITY = .0001
beliefs = []
beliefsInitialized = []
FORWARD_LOOKING_LOOPS = 1


def createTeam(firstIndex, secondIndex, isRed,
               first='TopAgent', second='BottomAgent', **args):
    if 'numTraining' in args:
        arguments['numTraining'] = args['numTraining']
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


# Our agent does worse with training, because we did not focus on training
class ApproximateQAgent(CaptureAgent):

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.numTraining = 0
        self.episodesSoFar = 0
        self.epsilon = 0.05
        self.discount = 0.8
        self.alpha = 0.2
        self.weights = util.Counter()
        self.weights['successorScore'] = 100
        self.weights['distanceToFood'] = -1
        # self.weights['ghostDistance'] = -10
        self.weights['stop'] = -1000
        self.weights['legalActions'] = 100
        self.weights['capsuleValue'] = 100
        # self.weights['capsuleDist'] = -3
        self.weights['eatCapsule'] = 100
        self.weights['isCorner'] = -10
        self.distanceToTrackCapsuleValue = 3
        self.minBeansToCashIn = 8
        self.weights['chaseEnemyValue'] = -200
        self.chaseEnemyDistance = 5
        self.threatenedDistance = 2
        self.legalActionMap = {}
        self.initialLegalPos = False
        if SHOW:
            print "INITIAL WEIGHTS"
            print self.weights

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.lastAction = None
        self.legalAction, self.corners = self.initialLegalAction(gameState)
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

    """""""""
    def getFeatures(self, gameState, action):

        Returns a counter of features for the state

        successor = self.getSuccessor(gameState, action)
        features = util.Counter()
        features['score'] = self.getScore(successor)
        if not self.red:
            features['score'] *= -1
        features['choices'] = self.getLegalActionsMethod(gameState, action)
        return features
    """""""""

    def getLegalActionsMethod(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        return len(successor.getLegalActions(self.index))

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        bestValue = -999999
        bestActions = None
        for action in state.getLegalActions(self.index):

            value = self.getQValue(state, action)
            if value > bestValue:
                bestActions = [action]
                bestValue = value
            elif value == bestValue:
                bestActions.append(action)
        if bestActions == None:
            return Directions.STOP
        # If no legal actions return None
        return bestActions[0]  # Else choose one of the best actions randomly

    def computeValueFromQValues(self, state):

        illegaAction = True
        valueList = []
        for action in state.getLegalActions(self.index):
            # For each action, if that action is the best then
            # update bestValue
            illegaAction = False
            value = self.getQValue(state, action)
            valueList.append(value)
        bestValue = max(valueList)
        if illegaAction:
            return 0  # If there is no legal action return 0
        # Otherwise return the best value found
        return bestValue

    def getQValue(self, state, action):
        """""""""
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """""""""
        sumList = []
        total = 0
        weights = self.weights
        features = self.getFeatures(state, action)
        for feature in features:
            # Implements the Q calculation
            sumList.append(features[feature] * weights[feature])
        return sum(sumList)

    def observationFunction(self, gameState):
        if len(self.observationHistory) > 0 and self.episodesSoFar < self.numTraining:
            self.update(self.getCurrentObservation(), self.lastAction, gameState, self.getReward(gameState))
        return gameState.makeObservation(self.index)

    def update(self, state, action, nextState, reward):

        # Should update your weights based on transition

        difference = (reward + self.discount * self.computeValueFromQValues(nextState))
        difference -= self.getQValue(state, action)
        # Only calculate the difference once, not in the loop.
        newWeights = self.weights.copy()
        # Same with weights and features.
        features = self.getFeatures(state, action)
        for feature in features:
            # Implements the weight updating calculations
            newWeight = newWeights[feature] + self.alpha * difference * features[feature]
            newWeights[feature] = newWeight
        self.weights = newWeights.copy()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.final(self, state)
        if self.episodesSoFar < self.numTraining and SHOW:
            print "END WEIGHTS"
            print self.weights
        self.episodesSoFar += 1
        if self.episodesSoFar == self.numTraining:
            print "FINISHED TRAINING"

    def getLegalActions(self, gameState):
        currentPos = gameState.getAgentState(self.index).getPosition()
        if currentPos not in self.legalActionMap:
            self.legalActionMap[currentPos] = gameState.getLegalActions(self.index)
        return self.legalActionMap[currentPos]

    def getFeatures(self, gameState, action):
        self.observeAllOpponents(gameState)
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)

        # Compute distance to the nearest food
        currentFoodList = self.getFood(gameState).asList()
        if len(currentFoodList) > 0:  # This should always be True,  but better safe than sorry
            minDist = min(
                [self.getMazeDistance(myPos, food) + abs(self.favoredY - food[1]) for food in currentFoodList])
            features['distanceToFood'] = minDist

        # Grab all enemies
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

        enemyPacmen = [a for a in enemies if a.isPacman and a.getPosition() != None]
        nonScaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
        scaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
        capsulesList = self.getCapsules(successor)

        enemyDists, enemyPos, enemyDis = self.getMiniDist(successor, myPos, nonScaredGhosts)
        enemyPacmanDists, enemyPacmanPos, enemyPacmanDis = self.getMiniDist(successor, myPos, enemyPacmen)

        if myPos in self.corners:
            features['isCorner'] = 8

        # Use the smallest distance
        if len(enemyDists) > 0:
            if not self.isHomeSide(gameState.getAgentPosition(self.index), gameState):
                if enemyDis == self.getMazeDistance(gameState.getAgentPosition(self.index), enemyPos):
                    features['ghostDistance'] = enemyDis

        features['capsuleValue'] = self.getCapsuleValue(myPos, gameState, scaredGhosts)
        # capsuleDis=999999
        # capsulesList = self.getCapsules(successor)
        # capsulesMinDistance = 0
        # if len(capsulesList) > 0 and len(scaredGhosts) == 0:
        #     dis=[]
        #     for pellet in capsulesList:
        #         dis.append(self.getMazeDistance(myPos, pellet))
        #     capsuleDis=min(dis)
        #     features['capsuleDist'] = capsuleDis
        features['eatCapsule'] = -len(capsulesList)

        if len(scaredGhosts) > 0:
            features['successorScore'] = -len(foodList) * 10
            features['distanceToFood'] = minDist * 0.1

        # Use the smallest distance
        if len(enemyPacmanDists) > 0:
            # if self.isHomeSide(gameState.getAgentPosition(self.index), gameState):
            if self.defendList[self.index] and not self.isHomeSide(
                    successor.getAgentPosition(self.index), successor):
                for i in self.getTeam(gameState):
                    if i != self.index:
                        if self.isHomeSide(gameState.getAgentPosition(i), gameState):
                            self.defendList[self.index] = False
                            self.defendList[i] = True
                            print self.defendList
                        else:
                            self.defendList[self.index] = True
                            self.defendList[i] = False
                            print self.defendList
            elif self.defendList[self.index] and self.isHomeSide(
                    successor.getAgentPosition(self.index), successor):
                self.defendList[self.index] = True
                self.defendList[i] = False
            elif not self.defendList[self.index] and self.isHomeSide(
                    successor.getAgentPosition(self.index), successor):
                for i in self.getTeam(gameState):
                    if i != self.index:
                        if not self.isHomeSide(gameState.getAgentPosition(i), gameState):
                            self.defendList[self.index] = True
                            self.defendList[i] = False

            if self.defendList[self.index]:
                features['chaseEnemyValue'] = enemyPacmanDis
                features['distanceToFood'] = 0

                # elif not self.isHomeSide(gameState.getAgentPosition(i), gameState) and self.isHomeSide(
                #         gameState.getAgentPosition(self.index), gameState):
                #     self.defendList[self.index] = True
                #     self.defendList[i] = False
                #     features['chaseEnemyValue'] = enemyPacmanDis
                # elif not self.isHomeSide(gameState.getAgentPosition(i), gameState) and not self.isHomeSide(
                #         gameState.getAgentPosition(self.index), gameState):
                #     self.defendList[self.index] = True
                #     self.defendList[i] = False
                #     features['chaseEnemyValue'] = enemyPacmanDis
        else:
            features['distanceToFood'] = minDist * 0.2

            # if enemyPacmanDis == self.getMazeDistance(gameState.getAgentPosition(self.index), enemyPacmanPos):

            # If on defense, heavily value chasing after enemies
        if self.defenseTimer > 0:
            self.defenseTimer -= 1
            features['chaseEnemyValue'] *= 100
            # If our opponents ate all our food (except for 2), we rush them
        if len(self.getFoodYouAreDefending(successor).asList()) <= 10:
            features['chaseEnemyValue'] *= 100

        # If we cashed in any pellets, we shift over to defense mode for a time
        if myState.numReturned != self.lastNumReturnedBeans:
            self.defenseTimer = DEFENSE_TIMER_MAX
            self.lastNumReturnedBeans = myState.numReturned

        # Heavily prioritize not stopping
        if action == Directions.STOP:
            features['stop'] = 1

        # The total of the legalActions you can take from where you are AND
        # The legalActions you can take in all future states
        # It depends on how many loops we do
        features['legalActions'] = self.getLegalActionModifier(gameState, FORWARD_LOOKING_LOOPS)
        return features

    def getMiniDist(self, successor, myPos, opponentsList):
        # Computes distance to enemy non scared ghosts we can see
        dists = []
        enemyPos = None
        enemyDis = 999999
        for index in self.getOpponents(successor):
            enemy = successor.getAgentState(index)
            if enemy in opponentsList:
                if USE_BELIEF_DISTANCE:
                    dis = self.getMazeDistance(myPos, self.getMostLikelyGhostPosition(index))
                else:
                    dis = self.getMazeDistance(myPos, enemy.getPosition())
                if dis < enemyDis:
                    enemyDis = dis
                    enemyPos = enemy.getPosition()
                dists.append(dis)
        return dists, enemyPos, enemyDis

    # If there are not any scared ghosts, then we value eating pellets
    def getCapsuleValue(self, myPos, successor, scaredGhosts):
        powerPellets = self.getCapsules(successor)
        minDistance = 0
        if len(powerPellets) > 0 and len(scaredGhosts) == 0:
            distances = [self.getMazeDistance(myPos, pellet) for pellet in powerPellets]
            minDistance = min(distances)
        return max(self.distanceToTrackCapsuleValue - minDistance, 0)

    # Uses our beliefs based on the noisyDistance, and we just use the highest belief
    def getMostLikelyGhostPosition(self, ghostAgentIndex):
        return max(beliefs[ghostAgentIndex])

    # # We loop over each possible legal action and tally up the possible actions from there
    def getLegalActionModifier(self, gameState, numLoops):
        legalActions = self.getLegalActions(gameState)
        numActions = len(legalActions)
        for legalAction in legalActions:
            if numLoops > 0:
                newState = self.getSuccessor(gameState, legalAction)
                numActions += self.getLegalActionModifier(newState, numLoops - 1)
        return numActions

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
            trueDistance = self.getMazeDistance(p, pacmanPosition)
            modelProb = gameState.getDistanceProb(trueDistance,
                                                  noisyDistance)  # Find the probability of getting this noisyDistance if the ghost is at this position
            if modelProb > 0:
                oldProb = beliefs[opponentIndex][p]
                allPossible[p] = (oldProb + MINIMUM_PROBABILITY) * modelProb
            else:
                allPossible[p] = 0
        allPossible.normalize()
        beliefs[opponentIndex] = allPossible


class ReflexAgent(ApproximateQAgent):

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)
        self.favoredY = 0.0
        self.defenseTimer = 0.0
        self.lastNumReturnedBeans = 0.0
        self.legalAction, self.corners = self.initialLegalAction(gameState)
        self.getLegalPositions(gameState)
        self.originalFood = len(self.getFood(gameState).asList())
        self.actionHistory = []
        self.count = 0
        self.defendList = {}
        for i in self.getTeam(gameState):
            if i <= 1:
                self.defendList[i] = True
            else:
                self.defendList[i] = False
        print self.defendList

    def __init__(self, index):
        ApproximateQAgent.__init__(self, index)

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

    def getLegalPositions(self, gameState):
        if not self.initialLegalPos:
            self.legalPositions = []
            walls = gameState.getWalls()
            for x in range(walls.width):
                for y in range(walls.height):
                    if not walls[x][y]:
                        self.legalPositions.append((x, y))
            self.initialLegalPos = True
        return self.legalPositions

    def chooseAction(self, gameState):
        # Append game state to observation history...
        # self.observationFunction(gameState)
        self.observationHistory.append(gameState)
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
        for ghost in enemyIndexes:
            ghostPos = gameState.getAgentPosition(ghost)
            if ghostPos != None:
                dis = self.getMazeDistance(agentPos, ghostPos)
                if dis < enemyDis:
                    enemyDis = dis
                    enemyPos = ghostPos
                    enemyInx = ghost

        if (SHOW):
            print "AGENT " + str(self.index) + " choosing action!"
        if len(legalActions):
            if util.flipCoin(self.epsilon) and self.episodesSoFar < self.numTraining:
                action = random.choice(legalActions)
                if (SHOW):
                    print "ACTION CHOSE FROM RANDOM: " + action
            else:
                action = self.computeActionFromQValues(gameState)
                if (SHOW):
                    print "ACTION CHOSE FROM Q VALUES: " + action

        foodlist = self.getFood(gameState).asList()
        carryFood = self.originalFood - len(foodlist)
        self.lastAction = action
        if (len(ghosts) > 0 and not self.isHomeSide(agentPos, gameState)):
            if enemyDis <= 5 and gameState.getAgentState(enemyInx).scaredTimer <= 4:
                self.refreshLegalActionAndCorners(gameState, enemyPos, 1)
                heruisticDist = self.breadthFirstSearch(agentPos, gameState)
                if heruisticDist > 0:
                    actionlist = self.aStarSearch(gameState, agentPos, heruisticDist)
                    # print actionlist
                    action = actionlist[0]
                    return action
            if len(foodlist) <= 2:
                heruisticDist = self.breadthFirstSearch(agentPos, gameState)
                if heruisticDist > 0:
                    actionlist = self.aStarSearch(gameState, agentPos, heruisticDist)
                    # print actionlist
                    action = actionlist[0]
                    return action

        self.actionHistory.append(action)
        if len(self.actionHistory) > 10:
            if self.actionHistory[len(self.actionHistory) - 1] == self.actionHistory[len(self.actionHistory) - 3]:
                if self.actionHistory[len(self.actionHistory) - 1] != self.actionHistory[len(self.actionHistory) - 2]:
                    self.count += 1
                    if self.count > 2:
                        action = random.choice(legalActions)

        # foodLeft = len(foodlist)
        # # Prioritize going back to start if we have <= 2 pellets left
        # if foodLeft <= 2:
        #     bestDist = 9999
        #     for a in legalActions:
        #         successor = self.getSuccessor(gameState, a)
        #         pos2 = successor.getAgentPosition(self.index)
        #         dist = self.getMazeDistance(self.start, pos2)
        #         if dist < bestDist:
        #             action = a
        #             bestDist = dist
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


class TopAgent(ReflexAgent):

    def registerInitialState(self, gameState):
        ReflexAgent.registerInitialState(self, gameState)
        self.favoredY = gameState.data.layout.height


class BottomAgent(ReflexAgent):

    def registerInitialState(self, gameState):
        ReflexAgent.registerInitialState(self, gameState)
        self.favoredY = 0.0
