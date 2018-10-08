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
import random, time, util
from game import Directions
import game
from util import nearestPoint
import copy
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##################
# DefensiveAgent #
##################

class DefensiveAgent(CaptureAgent):

    def registerInitialState(self, gameState):

        self.allDir = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.directions = {Directions.NORTH: (0, 1),
                           Directions.SOUTH: (0, -1),
                           Directions.EAST: (1, 0),
                           Directions.WEST: (-1, 0)}
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
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

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)

      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()

      # Computes whether we're on defense (1) or offense (0)
      features['onDefense'] = 1
      if myState.isPacman: features['onDefense'] = 0

      # Computes distance to invaders we can see
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      features['numInvaders'] = len(invaders)
      if len(invaders) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        features['invaderDistance'] = min(dists)

      if action == Directions.STOP: features['stop'] = 1
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1

      return features

    def getWeights(self, gameState, action):
        self.alpha=0.5
        self.gamma=0.9

        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

    def getEnemyPos(self, gameState):
        enemyPos = []
        for enemy in self.getOpponents(gameState):
            pos = gameState.getAgentPosition(enemy)
            # Will need inference if None
            if pos != None:
                if gameState.getAgentState(enemy).isPacman:
                    enemyPos.append((enemy, pos))
        return enemyPos


##################
# OffensiveAgent #
##################

class OffensiveAgent(CaptureAgent):
  def registerInitialState(self, gameState):
      self.moves = 0
      self.allDir = [(-1, 0), (1, 0), (0, 1), (0, -1)]
      self.directions = {Directions.NORTH: (0, 1),
                         Directions.SOUTH: (0, -1),
                         Directions.EAST: (1, 0),
                         Directions.WEST: (-1, 0)}
      self.start = gameState.getAgentPosition(self.index)
      CaptureAgent.registerInitialState(self, gameState)
      self.closestDistance = util.Counter()

  def chooseAction(self, gameState):
      """
  Picks among actions randomly.
  """
      actions = gameState.getLegalActions(self.index)

      # You can profile your evaluation time by uncommenting these lines
      # start = time.time()
      values = [self.evaluate(gameState, a) for a in actions]
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


      myPos = gameState.getAgentState(self.index).getPosition()
      distEnemy, posEnemy = self.enemyDist(gameState)
      foodList = self.getFoodYouAreDefending(gameState).asList()
      for ghost in posEnemy:
           if ghost in self.findBoardersDomain(gameState, not self.red) and myPos in self.findBoardersDomain(gameState,
                                                                                                             self.red) :
              drift= myPos[1] - ghost[1]
              # if myPos[0]==gameState.data.layout.width/2:
              #     goalX=int(myPos[0]+5)
              # elif myPos[0]==gameState.data.layout.width/2-1:
              #     goalX=int(myPos[0]-5)
              if drift in range(-3, 3) :
                  minShiftDist=999
                  shift=None
                  for food in foodList:
                      shiftDist=self.getMazeDistance(food, myPos)
                      if shiftDist<minShiftDist:
                          minShiftDist=shiftDist
                          shift=food
                  # self.aviodGhostDistance(myPos, ghost, gameState, 4)
                  # height = int(gameState.data.layout.height / 3)
                  # goalY = 1
                  # if myPos[1] < 2 * height:
                  #     goalY = int(myPos[1] + height)
                  #     while gameState.hasWall(goalX, goalY) and goalY in range(1, int(gameState.data.layout.height)):
                  #         goalY += 1
                  # else:
                  #     goalY = int(myPos[1] - height)
                  #     while gameState.hasWall(goalX, goalY) and goalY in range(1, int(gameState.data.layout.height)):
                  #         goalY -= 1
                  bestDist = 9999
                  for action in actions:
                     successor = self.getSuccessor(gameState, action)
                     pos2 = successor.getAgentPosition(self.index)
                     dist = self.getMazeDistance(food, pos2)
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

  def evaluate(self, gameState, action):
      """
      Computes a linear combination of features and feature weights
      """
      features = self.getFeatures(gameState, action)
      weights = self.getWeights(gameState, action)
      return features * weights

  def getFeatures(self, gameState, action):
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)
      foodList = self.getFood(successor).asList()

      features['eatFood'] = -len(foodList)  # self.getScore(successor)
      myPos = successor.getAgentState(self.index).getPosition()
      # Compute distance to the nearest food

      if len(foodList) > 0:  # This should always be True,  but better safe than sorry
          features['distanceToFood'] = min([self.getMazeDistance(myPos, food) for food in foodList])

      distanceToClosestExits = min(
          [self.getMazeDistance(myPos, i) for i in self.findBoarders(successor,self.red)])

      # if self.moves >0 :
      #     self.moves -= 1
      #
      # if myPos in self.getCapsules(successor):
      #     self.moves =120

    # Compute distance to enemy
      features['escapeGhost'] = .1
      features['aviodGhost'] = .1
      distEnemy,posEnemy = self.enemyDist(successor)
      enemyDix=[]
      for ene in  self.getOpponents(successor):
        if not successor.getAgentState(ene).isPacman:
            enemyDix.append(ene)
      if distEnemy != None:
        if distEnemy <= 2:
            for ghost in enemyDix:
                if (successor.getAgentState(ghost).scaredTimer <=8 ):
                    print(successor.getAgentState(ghost).scaredTimer)
                    features['ghostClosing'] = 4*(1-self.side(successor))
                    features['escapeGhost'] = self.escapeGhostDistance(myPos, posEnemy[0], successor,1)*(1-self.side(successor))
                    if len(posEnemy) > 1:
                        features['escapeGhost'] = self.escapeGhostDistance(myPos, posEnemy[1], successor,1)*(1-self.side(successor))
        elif distEnemy <= 4:
            for ghost in enemyDix:
                if (successor.getAgentState(ghost).scaredTimer <=8 ):
                    print(successor.getAgentState(ghost).scaredTimer)
                    features['ghostClosing'] = 1*(1-self.side(successor))
                    features['aviodGhost'] = self.aviodGhostDistance(myPos, posEnemy[0], successor,1)*(1-self.side(successor))
                    if len(posEnemy) > 1:
                        features['aviodGhost'] = self.aviodGhostDistance(myPos, posEnemy[1],successor,1)*(1-self.side(successor))
        else:
            features['ghostClosing'] = 0


      features['distanceToBorders'] = distanceToClosestExits

      if myPos in foodList:
          self.foodNum += 1.0
      if self.side(successor) == 0.0:
          self.foodNum = 0.0
      features['carryFood'] = self.foodNum * distanceToClosestExits * (1-self.side(successor))

      # Dropping off food heuristic
      features['dropFood'] = self.foodNum * (self.side(successor))


    # Compute distance to capsule
      capsules = self.getCapsules(successor)
      if (len(capsules) > 0):
          minCapsuleDist = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
          features['eatCapsule'] = -len(capsules)
      else:
          minCapsuleDist = .1
      features['capsuleDist'] = minCapsuleDist

      if (action == Directions.STOP):
          features['stop'] = 1.0
      else:
          features['stop'] = 0.0


      return features

  def getWeights(self, gameState, action):

    weights = util.Counter()
    successor = self.getSuccessor(gameState, action)
    weights['stop'] = -1000

    weights['dropFood'] = 20
    weights['carryFood'] = -10
    weights['distanceToFood'] = -200
    weights['eatFood'] = 5000

    weights['eatCapsule'] = 2000
    weights['capsuleDist'] = -100

    weights['distanceToBorders'] = -5
    weights['ghostClosing'] = -300
    weights['escapeGhost']=-100
    weights['aviodGhost']=-100


    distEnemy, posEnemy = self.enemyDist(successor)
    enemyDix = []
    for ene in self.getOpponents(successor):
        if not successor.getAgentState(ene).isPacman:
            enemyDix.append(ene)
    if distEnemy != None:
        for ghost in enemyDix:
            if (successor.getAgentState(ghost).scaredTimer > 8):
                print(successor.getAgentState(ghost).scaredTimer)
                weights['distanceToBorders'] = -5
                weights['escapeGhost'] = 0
                weights['aviodGhost'] = 0
                weights['ghostClosing'] = 0

                weights['eatCapsule'] = 5
                weights['capsuleDist'] = -5

                weights['carryFood'] = -5
                weights['eatFood'] = 4000
                weights['distanceToFood'] = -1000
            else:
                if distEnemy<=4:
                    print("close")
                    weights['carryFood'] = -100
                    weights['eatFood'] = 100
                    weights['dropFood'] = 10000
                    weights['distanceToFood'] = -50

                    weights['distanceToBorders'] = -500
                    weights['ghostClosing'] = -2000

                    weights['eatCapsule'] = 20000
                    weights['capsuleDist'] = -300
                    weights['goBack'] = -50
                if distEnemy<=2:
                    weights['escapeGhost'] = -3000
                    weights['aviodGhost'] = -30
                elif distEnemy<=4:
                    # myPos = gameState.getAgentState(self.index).getPosition()
                    # distEnemy, posEnemy = self.enemyDist(gameState)
                    # for ghost in posEnemy:
                    #     if ghost not in self.findBoardersDomain(gameState, not self.red) and myPos not in self.findBoardersDomain(
                    #         gameState, self.red):
                    weights['escapeGhost'] = -30
                    weights['aviodGhost'] = -3000
    return weights

  def getEnemyPos(self, gameState):
      enemyPos = []
      for enemy in self.getOpponents(gameState):
          pos = gameState.getAgentPosition(enemy)
          # Will need inference if None
          if pos != None:
              if not gameState.getAgentState(enemy).isPacman:
                  enemyPos.append((enemy, pos))
      return enemyPos

  def enemyDist(self, gameState):
      pos = self.getEnemyPos(gameState)
      minDist = None
      minPosition = []
      if len(pos) > 0:
          minDist = float('inf')
          myPos = gameState.getAgentPosition(self.index)
          for i, p in pos:
              dist = self.getMazeDistance(myPos, p)
              if dist < minDist:
                  minDist = dist
                  minPosition = []
                  minPosition.append(p)
              elif dist == minDist:
                  minPosition.append(p)
      return minDist, minPosition

  def side(self, gameState):
      width, height = gameState.data.layout.width, gameState.data.layout.height
      position = gameState.getAgentPosition(self.index)
      if self.index % 2 == 1:
          # red
          if position[0] < width / (2)+1:
              return 1.0
          else:
              return 0.0
      else:
          # blue
          if position[0] > width / 2:
              return 1.0
          else:
              return 0.0

  def findBoarders(self, gameState, isRed):
      boarders = []
      # Put the border points
      if isRed:
          j = gameState.data.layout.width / 2-1
      else:
          j = gameState.data.layout.width / 2
      for i in range(gameState.data.layout.height):
          if gameState.hasWall(j, i) == False:
              boarders.append((j, i))
      return boarders

  def findBoardersDomain(self, gameState, isRed):
      boarders = []
      # Put the border points
      if isRed:
          j = gameState.data.layout.width / 2 - 1
      else:
          j = gameState.data.layout.width / 2
      for i in range(gameState.data.layout.height):
          if isRed:
              for pos in range(j - 3, j):
                  if gameState.hasWall(j, i) == False:
                      boarders.append((j, i))
          else:
              for pos in range(j, j + 3):
                  if gameState.hasWall(j, i) == False:
                      boarders.append((j, i))
      return boarders

  def enemyDomain(self, myPos, enemyPos, gameState, n):
      x1, y1 = myPos
      x2, y2 = enemyPos
      walls = copy.deepcopy(gameState.getWalls())
      walls[x2][y2] = True
      for i in range(1, n):
          walls[x2 + 1 * i][y2] = True
          walls[x2][y2 + 1 * i] = True
          walls[x2 + 1 * i][y2 + 1 * i] = True
          walls[x2 + 1 * i][y2 - 1 * i] = True
          walls[x2 - 1 * i][y2] = True
          walls[x2][y2 - 1 * i] = True
          walls[x2 - 1 * i][y2 - 1 * i] = True
          walls[x2 - 1 * i][y2 + 1 * i] = True
      return walls

  def escapeGhostDistance(self, myPos, enemyPos, gameState, n):
      return len(self.breadthFirstSearch(myPos, gameState, self.enemyDomain(myPos, enemyPos, gameState, n), True))

  def aviodGhostDistance(self, myPos, enemyPos, gameState, n):
      return len(self.breadthFirstSearch(myPos, gameState, self.enemyDomain(myPos, enemyPos, gameState, n), False))

  def breadthFirstSearch(self, myPos, gameState, walls, isEscape):
      """Search the shallowest nodes in the search tree first."""
      explored, frontier = [], util.Queue()
      explored.append(myPos)
      frontier.push([myPos, []])

      while not frontier.isEmpty():
          node, path = frontier.pop()
          if not isEscape:
              if node in self.getFood(gameState):
                  return path
          else:
              if node in self.findBoarders(gameState, self.red):
                  return path
          actionList = []
          for direction in self.allDir:
              x, y = direction
              nx, ny = node
              if not walls[(int)(x + nx)][(int)(y + ny)]:
                  for i in self.directions.keys():
                      if self.directions.get(i) == direction:
                          actionList.append([(x + nx, y + ny), i])
          for successor in actionList:
              nextState, action = successor
              if nextState not in explored:
                  explored.append(nextState)
                  frontier.push([nextState, path + [action]])
      return []




