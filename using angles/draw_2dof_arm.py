import pygame
import math
import numpy as np
import random


class ARM2D(object):
	def __init__ (self):
		# Initialize pygame
		pygame.init()
		self.SCREEN_WIDTH = 500
		self.SCREEN_HEIGHT = 250
		
		self.FPS = 15
		self.l1 = 100
		self.l2 = 100
				
		self.arm_thickness = 5
		self.origin = [self.SCREEN_WIDTH//2 , self.SCREEN_HEIGHT-self.arm_thickness]#-self.arm_thickness
		
		self.color_black = [0,0,0]
		self.color_white = [255,255,255]
		
	def reset(self):
		# Angle in radian
		self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
		self.clock = pygame.time.Clock()

	def step(self, angle1, angle2):
		pygame.event.pump()
		self.screen.fill(self.color_white)
		# Set arm angle
		self.angle1 = angle1 * math.pi/180.0
		self.angle2 = angle2 * math.pi/180.0

		self.arm_1_point_list = [self.origin, [self.origin[0] + self.l1*math.cos(self.angle1), self.origin[1] - self.l1*math.sin(self.angle1)]]
		self.arm_2_point_list = [[self.origin[0] + self.l1*math.cos(self.angle1), self.origin[1] - self.l1*math.sin(self.angle1)],[self.arm_1_point_list[1][0]+self.l2*math.cos(self.angle1+self.angle2),self.arm_1_point_list[1][1]-self.l2*math.sin(self.angle1+self.angle2)]]
		pygame.draw.lines(self.screen, self.color_black, False, self.arm_1_point_list, self.arm_thickness)
		pygame.draw.lines(self.screen, self.color_black, False, self.arm_2_point_list, self.arm_thickness)

		# Save screenshot
		pygame.image.save(self.screen, "./inv_images/" + str([angle1,angle2]) + ".jpg")
		# Print angle
		# print ("Arm Angle : ", angle1)

		print("end_effector_point: ",self.arm_2_point_list[1])

		self.clock.tick(self.FPS)
		pygame.display.flip()



if __name__ == "__main__":   
	game = ARM2D()
	game.reset()

	for i in range(90):
		for j in range(90):
			game.step(i,j)

	