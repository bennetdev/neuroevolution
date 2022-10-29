import pygame
from NeuralNetwork import NeuralNetwork
from constants import *
from geneticAlgorithm import eliteSelection
from objects import *


# Game
pygame.init()

# Create the screen object
# The size is determined by the constant SCREEN_WIDTH and SCREEN_HEIGHT
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

players = []
obstacle = Obstacle()

# Sprite groups
obstacles = pygame.sprite.Group()
obstaclesList = []
allSprites = pygame.sprite.Group()
allSprites.add(obstacle)
obstacles.add(obstacle)

# init players
for i in range(NUM_PLAYERS):
    p = Player()
    players.append(p)
    allSprites.add(p)

# Setup the clock for constant framerate
clock = pygame.time.Clock()


generation = 1
bestPlayer = players[0]
bestPlayersScores = [(bestPlayer.score, bestPlayer.id)]

# Game Loop
framerate = 120
running = True
while running:
    
    sortedPlayers = sorted(players, key=lambda p : p.score, reverse=True)
    

    # Event handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                sortedPlayers[0].brain.save()
            elif event.key == pygame.K_b:
                print(bestPlayer)
            elif event.key == pygame.K_f:
                framerate = (framerate) + 120 if framerate < 300 else 120

    # spawn new obstacle
    if Obstacle.shouldObstacleSpawn(obstacle):
        # Create the new enemy and add it to sprite groups
        obstacle = Obstacle()
        obstacles.add(obstacle)
        obstaclesList.append(obstacle)
        allSprites.add(obstacle)

    # update sprites
    for index,p in enumerate(players):
        if not p.dead:
            p.update(obstacles, obstaclesList)
    obstacles.update(players, obstaclesList)

    screen.fill((0, 0, 0))

    # draw all sprites
    for entity in allSprites:
        screen.blit(entity.surf, entity.rect)


    # Score display
    font = pygame.font.Font('freesansbold.ttf', 32)

    text = font.render(str(sortedPlayers[0].score) + ", Alive: " + str(len([p for p in players if not p.dead]))+ " Generation: " + str(generation), True, (255,255,255), (0,0,0))

    screen.blit(text, text.get_rect())

    # Check if any enemies have collided with the player
    if all([p.dead for p in players]):
        # If so, then remove the player and stop the loop
        for player in players:
            player.kill()

        # restart game
        obstacles.empty()
        allSprites.empty()
        obstaclesList = []

        # update all time best player
        if sortedPlayers[0].score > bestPlayer.score:
            bestPlayer = sortedPlayers[0]

        # append to all best players
        bestPlayersScores.append((bestPlayer.score, bestPlayer.id))
            
        # create new generation
        newPlayers = eliteSelection(players)
        players = []
        for p in newPlayers:
            players.append(p)
            allSprites.add(p)

        obstacle = Obstacle()
        obstacles.add(obstacle)
        allSprites.add(obstacle)
        obstaclesList.append(obstacle)
        generation += 1

    # Flip the display
    pygame.display.flip()

    # Ensure program maintains a rate of framerate frames per second
    clock.tick(framerate)