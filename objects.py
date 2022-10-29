from __future__ import annotations
import pygame
from typing import Any, List
import random
from NeuralNetwork import NeuralNetwork, load as loadNeuralNetwork
import numpy as np
from constants import *

# Player is a pygame.sprite.Sprite
# - surface
# - rectangle
# - score
# - dead
# - jumping
# - ducking
# - dead
# - id
# - fitness
# - size (basically y-position)
class Player(pygame.sprite.Sprite):
    POS_X = 100
    JUMP_ACCELERATION = 5
    def __init__(self, brain: NeuralNetwork = None, load: bool = False) -> None:
        super(Player, self).__init__()
        self.surf = pygame.Surface((50,50))
        self.surf.fill((255,255,255))
        self.rect = self.surf.get_rect(
            bottomleft=(Player.POS_X, GROUND_Y)
        )

        self.acceleration = 0
        self.speed = JUMP_HEIGHT
        self.jumping = False
        self.ducking = False
        self.size = 40
        self.score = 0
        self.fitness = 0
        self.dead = False
        self.id = "".join([str(random.randint(0,9)) for _ in range(9)])

        # the players neural network
        if load:
            self.brain = loadNeuralNetwork()
        else:
            if brain is not None:
                self.brain = brain
            else:
                self.brain = NeuralNetwork(3, 14, 2, 0.1)

    # overload to string function to print player
    def __str__(self) -> str:
        return "Best Player: " + str(self.score)

    # update player position
    # Input: Sprite group of obstacles
    #        List of Obstacles
    def update(self, obstacles: pygame.sprite.Group, obstaclesList: List[Obstacle]) -> None:
        if pygame.sprite.spritecollideany(self, obstacles):
            self.dead = True
            self.kill()
        if len(obstaclesList) >= 2:
            output = self.think(obstaclesList)
            if max(output) == output[0]:
                self.jump()
            else:
                pass
        # only be able to jump if not jumping
        if self.jumping:
            self.rect.move_ip(0,-self.speed)
            self.speed -= GRAVITY
            if self.speed < -JUMP_HEIGHT:
                self.jumping = False
                self.speed = JUMP_HEIGHT

        if self.ducking:
            self.size -= GRAVITY
            if self.size < 0:
                self.ducking = False
                self.size = 40
                self.rect.move_ip(0,-30)

    # Makes player jump (for now) just sets jumping to True
    def jump(self) -> None:
        if not self.jumping and not self.ducking:
            self.jumping = True
            self.score -= 1

    # makes player size reduce to 20
    def duck(self) -> None:
        if not self.jumping and not self.ducking:
            self.ducking = True
            self.rect.move_ip(0,+30)

    # get input np array for neural network, (for now) consisting of player speed, x pos of next 2 obstacles
    # Input: List of Obstacles
    def inputForNeuralNetwork(self, obstaclesList: List[Obstacle]) -> np.ndarray:
        return np.asarray([self.speed / JUMP_HEIGHT, obstaclesList[0].rect.x / (SCREEN_WIDTH + 300),
                         obstaclesList[1].rect.x / (SCREEN_WIDTH + 300)])

    # query neural network for output
    # Input: List of Obstacles
    # Output: output of neural network, float
    def think(self, obstaclesList: List[Obstacle]) -> float:
        return self.brain.query(self.inputForNeuralNetwork(obstaclesList))



# Obstacle is a pygame.sprite.Sprite
# - surface
# - rectangle
# - const speed
class Obstacle(pygame.sprite.Sprite):
    SPEED = 5

    def __init__(self) -> None:
        super(Obstacle, self).__init__()
        self.high = False
        self.surf = pygame.Surface((25,200 if self.high else 70)) 
        self.surf.fill((255,255,255))
        self.rect = self.surf.get_rect(
            bottomleft=(
                random.randint(SCREEN_WIDTH + 100, SCREEN_WIDTH + 300),
                #SCREEN_WIDTH + 200,
                (GROUND_Y - 25 if self.high else GROUND_Y)
                #GROUND_Y
            )
        )
    
    # update position of obstacle
    # Input: List of all players
    #        List of all Obstacles
    def update(self, players: List[Player], obstaclesList: List[Obstacle]) -> None:
        global score

        self.rect.move_ip(-Obstacle.SPEED, 0)
        if self.rect.right < 0:
            for player in players:
                if not player.dead:
                    player.score += 2
            #print(obstaclesList)
            if self in obstaclesList:
                obstaclesList.remove(self)
            self.kill()

    # decide if obstacle should be spawned
    # Input: Latest obstacle
    # Output: Should obstacle spawn, as bool
    @staticmethod
    def shouldObstacleSpawn(obstacle: Obstacle) -> bool:
        if obstacle.rect.x < SCREEN_WIDTH - 140:
            return True
        else:
            return False

            