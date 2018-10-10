# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
from game import Actions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState,a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

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

    def evaluate(self,gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self,gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self,gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def findBorders(self, gameState, isRed):
        borders = []
          # Put the border points
        domain=[]
        if isRed:
            j = gameState.data.layout.width / 2-1
            for i in range(gameState.data.layout.height):
                for m in range(j-7,j):
                    if gameState.hasWall(m, i) == False:
                        borders.append((m, i))
        else:
            j = gameState.data.layout.width / 2
            for i in range(gameState.data.layout.height):
                for m in range(j, j+7):
                    if gameState.hasWall(m, i) == False:
                        borders.append((m, i))
        return borders

    def initializeCornersAndStates(self, gameState):
        corners=[]
        states=[]
        legalAction={}
        walls=gameState.getWalls()
        for x in range(0, gameState.data.layout.width):
            for y in range(0, gameState.data.layout.height):
                if not gameState.hasWall(x, y):
                    states.append((x, y))
                    possible = []
                    adjacentToConers = 0
                    for action, position in Actions._directionsAsList:
                        px, py = position
                        ny = py + y
                        nx = px + x
                        if not walls[nx][ny]: possible.append(action)
                        if (nx, ny) in corners: adjacentToConers = adjacentToConers + 1
                    legalAction[(x, y)] = possible
                    if len(possible) <=2:
                        corners.append((x, y))
        return corners,states,legalAction


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.enemyIndex= self.getOpponents(gameState)
        self.borders=self.findBorders(gameState,self.red)
        self.corners,self.states,self.legalAction=self.initializeCornersAndStates(gameState)
        self.values = util.Counter()
        self.discount = 0.9
        self.iterations = 30
        self.rewards={}
        self.food = len(self.getFood(gameState).asList())
        #self.offend = True

    def chooseAction(self, gameState):
        start = time.time()
        foodlist = self.getFood(gameState).asList()
        carryFood = self.food - len(foodlist)
        if gameState.getAgentPosition(self.index) in self.borders:
            self.food = self.food - carryFood
            carryFood = 0
        self.refreshReward(gameState,carryFood)
        self.valueIteration(gameState, carryFood)
        myPos = gameState.getAgentPosition(self.index)
        action = None
        maxValue = -9999999
        for a in self.legalAction[myPos]:
            if a!=Directions.STOP:
                qValue = self.calculaateQValue(myPos, a, gameState, carryFood)
                if qValue >= maxValue:
                    maxValue = qValue
                    action = a
        print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        if len(foodlist) <= 2:
            bestDist = 9999
            for a in gameState.getLegalActions(self.index):
                successor = self.getSuccessor(gameState, a)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = a
                    bestDist = dist
            return bestAction
        return action

    def side(self, state,gameState):
        width, height = gameState.data.layout.width, gameState.data.layout.height
        if self.index in [0,2]:
            # red
            if state[0] < width / 2:
                return 1.0
            else:
                return 0.0
        else:
            # blue
            if state[0] > width / 2-1:
                return 1.0
            else:
                return 0.0

    def calculaateQValue(self, state, action, gameState, carryFood):
        sumofSuccessors = 0
        for nextState, prob in self.getProbability(state, action):
            sumofSuccessors = sumofSuccessors + prob*(self.rewards[nextState]+ self.discount * self.values[nextState])
        return sumofSuccessors

    def refreshReward(self,gameState,carryFood):
        for state in self.states:
            features = self.getOffendFeatures(state, gameState, carryFood)
            weights = self.getOffendWeights(state, gameState, carryFood)
            self.rewards[state] = features * weights

    def getProbability(self, state, action):
        probOfActions = util.Counter()
        actions = self.legalAction[state]
        for a in actions:
            if a == action:
                nextSate = Actions.getSuccessor(state, action)
                probOfActions[nextSate] = 0.8
            else:
                nextSate = Actions.getSuccessor(state, a)
                probOfActions[nextSate] = 0.2
        probOfActions.normalize()
        return probOfActions.items()

    def valueIteration(self,gameState,carryFood):
        for i in range(0, self.iterations):
            oldValues = util.Counter()
            for state in self.states:
                maxvalue = float(-9999999)
                for action in self.legalAction[state]:
                    sumofSuccessors = 0
                    for nextState, prob in self.getProbability(state, action):
                        sumofSuccessors = sumofSuccessors + prob * (
                                    self.rewards[nextState] + self.discount * self.values[nextState])
                    maxvalue = max(sumofSuccessors, maxvalue)
                    oldValues[state] = maxvalue
            self.values = oldValues

    def getEnemyPos(self, gameState):
        enemyPos = []
        for enemy in self.enemyIndex:
            pos = gameState.getAgentPosition(enemy)
            if pos != None:
                print(pos)
                if not gameState.getAgentState(enemy).isPacman:
                    enemyPos.append(pos)
        return enemyPos

    def getEnemyDomain(self,gameState,pos,n,m):
        enemyDomain=[]
        if pos != None:
            x,y =pos
            #print(pos)
            for i in range(n, m):
                enemyDomain.append((x + i, y))
                enemyDomain.append((x - i, y))
                enemyDomain.append((x ,y + i))
                enemyDomain.append((x ,y - i))
                enemyDomain.append((x + i, y+ i))
                enemyDomain.append((x - i, y+ i))
                enemyDomain.append((x + i, y- i))
                enemyDomain.append((x - i, y- i))
        return enemyDomain

    def getOffendFeatures(self, state, gameState,carryFood):
        features = util.Counter()
        foodList = self.getFood(gameState).asList()
        if state in foodList:
            features['eatFood'] = 1.0-len(foodList)
        else:
            features['eatFood'] = -len(foodList)

        capsules = self.getCapsules(gameState)
        if state in capsules:
            features['eatCapsule'] = 1.0 -len(capsules)
        else:
            features['eatCapsule'] = -len(capsules)

        if state in self.borders:
            features['carryFood'] = 0
            features['dropFood'] = carryFood
        else:
            features['carryFood'] = carryFood
            features['dropFood'] = 0

        #if self.side(gameState.getAgentPosition(self.index),gameState)==0:
        #print("side : "+str(self.side(state,gameState)))
        enemyDis=9999999
        enemyPos=None
        for ghost in self.enemyIndex:
            if not gameState.getAgentState(ghost).isPacman:
                pos = gameState.getAgentPosition(ghost)
                if pos !=None:
                    dis=self.getMazeDistance(state, pos)
                    if dis<enemyDis:
                        enemyDis=dis
                        enemyPos=pos
        if (enemyDis <= 6):
            if enemyPos!=None:
                if state in self.getEnemyDomain(gameState, pos, 0,3):
                    if state in self.borders:
                        features['border'] = 50000
                    if state in self.corners:
                        features['corner'] = 100000.0
                    features['ghostPos'] = 50.0
                elif state in self.getEnemyDomain(gameState,pos,4,6):
                    if state in self.borders:
                        features['border'] = 10000
                    if state in self.corners:
                        features['corner'] = 50000.0
                    features['ghostPos'] = 5.0
                                # elif state in self.getEnemyDomain(gameState,pos,5,6):
                                #     if state in self.corners:
                                #         features['corner'] = 500.0
                                #     features['ghostPos'] = 1.0
        else:
            features['ghostPos'] = 0.1
        return features

    def getOffendWeights(self, state,gameState,carryFood):
        weights = util.Counter()
        weights['eatFood'] = 2000
        weights['carryFood']=-20
        weights['dropFood']=10
        weights['border'] = 1000
        weights['eatCapsule'] = 1000
        weights['ghostPos'] = -4000
        weights['corner'] = -50
        return weights


class DefensiveReflexAgent(ReflexCaptureAgent):
    lastSuccess = 0
    flag = 1
    flag2 = 0
    currentFoods = []
    s = []
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def getMostDenseArea(self, gameState):
        ourFood = self.getFoodYouAreDefending(gameState).asList()
        distance = [self.getMazeDistance(gameState.getAgentPosition(self.index), a) for a in ourFood]
        nearestFood = ourFood[0]
        nearestDstance = distance[0]

        for i in range(len(distance)):
            if distance[i] < nearestDstance:
                nearestFood = ourFood[i]
                nearestDstance = distance[i]
        return nearestFood

    def getFeatures(self, gameState, action):
        self.start = self.getMostDenseArea(gameState)

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        self.s = (18, 7)
        # print self.s
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        features['Boundries'] = self.getMazeDistance(myPos, self.s)

        if (self.flag2 == 0):
            self.flag2 = 1
            self.currentFoods = self.getFoodYouAreDefending(gameState).asList()
        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            pos = [a.getPosition() for a in invaders]
            nearestPos = pos[0]
            nearestDst = dists[0]

            for i in range(len(dists)):
                if dists[i] < nearestDst:
                    nearestPos = pos[i]
                    nearestDst = dists[i]

            features['invaderPDistance'] = nearestDst
            # print len(self.currentFoods), len(self.getFoodYouAreDefending(gameState).asList())
            if (features['invaderDistance'] == 1 or features['invaderPDistance'] == 1 or features[
                'invaderLDistance'] == 1):
                # print "here1"
                self.flag = 0
                self.lastSuccess = nearestPos
                features['invaderLDistance'] = self.getMazeDistance(myPos, self.lastSuccess)
                self.currentFoods = self.getFoodYouAreDefending(gameState).asList()
                # print "Got Him", self.lastSuccess , self.flag

            if (len(self.currentFoods) > len(self.getFoodYouAreDefending(gameState).asList())):
                # print "here2"
                nextFoods = self.getFoodYouAreDefending(gameState).asList()
                # print "Found Him"
                for i in range(len(self.currentFoods)):
                    # print self.currentFoods[i][0], self.currentFoods[i][1], nextFoods[i][0], nextFoods[i][1]
                    if (len(self.currentFoods) > 0 and len(nextFoods) > i):
                        # print "i: ",i,len(nextFoods),self.currentFoods[i][0],nextFoods[i][0],self.currentFoods[i][1],nextFoods[i][1]
                        if (self.currentFoods[i][0] != nextFoods[i][0] or self.currentFoods[i][1] != nextFoods[i][1]):
                            features['invaderPDistance'] = self.getMazeDistance(myPos, self.currentFoods[i])
                            # print "MYYYY", self.currentFoods[i]
                            self.lastSuccess = self.currentFoods[i]
                            self.currentFoods = nextFoods
                            break

                            # elif(self.flag==0):
                            # print "here4"
                            # features['invaderDistance']=self.getMazeDistance(myPos, self.lastSuccess)

                            # elif(self.flag==1):
                            # print "here3"
                            # if(self.lastSuccess==0):
                            # print "hii"
                            # self.lastSuccess=self.getMostDenseArea(gameState)
                            # features['invaderDistance']=self.getMazeDistance(myPos, self.lastSuccess)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'invaderPDistance': -20,
                'invaderLDistance': -5, 'Boundries': -10, 'stop': -100, 'reverse': -2}
