from typing import Any, List
from objects import Player
import random
from constants import *

# calculate fitness out of the score of each player
# Input: List of all players
def calculateFitness(players: List[Player]) -> None:
    summe = sum([p.score for p in players])
    for player in players:
        player.fitness = player.score / summe if summe != 0 else 1
    

# create next generation of players, by Eliteselection
# Input: Current players, List of Players
# Output: List of new Players
def eliteSelection(currentPlayers: List[Player]) -> List[Player]:
    sortedPlayers = sorted(currentPlayers, key=lambda p : p.score, reverse=True)
    lastSurvivorIndex = int(BEST_X_PERCENT * NUM_PLAYERS)

    newPlayers = []
    for i in range(0, lastSurvivorIndex):
        for _ in range(0,4):
            #parent2 = sortedPlayers[random.randint(0, lastSurvivorIndex - 1)]
            #newPlayer = Player(sortedPlayers[i].brain.crossover(parent2.brain))
            newPlayer = Player(sortedPlayers[i].brain.copy())
            newPlayer.brain.mutate(MUTATION_RATE)
            newPlayers.append(newPlayer)
    return newPlayers

# create next generation of players, by Roulette selection
# Input: Current players, List of Players
# Output: List of new Players
def rouletteSelection(currentPlayers: List[Player]) -> List[Player]:
    survivors = sorted(currentPlayers, key=lambda p : p.score, reverse=True)[:BEST_X_PERCENT * NUM_PLAYERS]
    calculateFitness(survivors)

    sortedPlayers = sorted(survivors, key=lambda p : p.fitness, reverse=True)

    parents = random.choices(survivors, weights=(p.fitness for p in sortedPlayers), k=NUM_PLAYERS * 4)

    newPlayers = []
    for i in range(0, NUM_PLAYERS):
        #parent2 = parents[i+1]

        newPlayer = Player(parents[i].brain.copy())
        #newPlayer = Player(parents[i].brain.crossover(parent2.brain))
        newPlayer.brain.mutate(MUTATION_RATE)
        newPlayers.append(newPlayer) 

    return newPlayers
