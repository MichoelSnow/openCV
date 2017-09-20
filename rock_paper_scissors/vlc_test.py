


import pygame
pygame.mixer.init()
pygame.mixer.music.load('The_Computer_Wins.mp3')
pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue
