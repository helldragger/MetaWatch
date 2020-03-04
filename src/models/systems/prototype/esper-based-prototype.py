# Objective, implement a basic hero simulation workspace, including world space and basic ai. (choice of action)

import esper 
# guides here -> https://github.com/benmoran56/esper
# TODO implement hero spawning.
import time

dt = 0.200 # time step between frames. keep this in seconds?

##################################
#  Define some Components:
##################################
class Velocity:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class Position:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self):
        return f"(x:{self.x}, y:{self.y}, z:{self.z})"

class Cooldown:
    def __init__(self, duration=0):
        self.duration = duration

class Hero:
    def __init__(self, name="hero"): 
        self.name = name

class Ability:
    def __init__(self, name="ability"): # TODO integrate more gameplay systems data
        self.name = name

class Team:
    def __init__(self, team=1):
        self.team = team

class Health:
    def __init__(self, health=200):
        self.health = health

class Wall:
    def __init__(self, x=0, y=0, z=0, width=0, height=0, degree=0):
        self.x = x # X bottom of the wall
        self.y = y # Y bottom of the wall
        self.z = z # Z bottom of the wall
        self.width = width # size on the X axe of the wall
        self.height = height # size on the Y axe of the wall
        self.degree = degree # orientation par rapport à 0 (axe X), 90 (axe Y), 180 (axe X).
################################
#  Define some Processors:
################################
class MovementProcessor(esper.Processor): # allows us to have a grid like system.
    def __init__(self):
        super().__init__()

    def process(self): # ici l'entité est un héros ou un projectile.
        for _, (vel, pos) in self.world.get_components(Velocity, Position):
            pos.x += vel.x
            pos.y += vel.y
            pos.z += vel.z
            print("Current Position: {}".format((int(pos.x), int(pos.y), int(pos.z))))

class CooldownProcessor(esper.Processor): # Allows to process abilities cooldowns
    def __init__(self):
        super().__init__()

    def process(self): # ici l'entité est une compétence.
        for ent, (cooldown, ability) in self.world.get_components(Cooldown, Ability):
            cooldown.duration -= dt
            if cooldown.duration <= 0:
                self.world.remove_component(ent, Cooldown)
                print(f"{ability.name} can now be used.")

class ViewProcessor(esper.Processor): # prints the world view
    def __init__(self):
        super().__init__()

    def process(self): # ici les entitées sont toutes entitees placées sur le monde donc murs et joueurs compris.
        for ent, (position, hero) in self.world.get_components(Position, Hero):
            print(f"Hero {hero.name} @ {position}")
        
# TODO view.
##########################################################
# Instantiate everything, and create your main logic loop:
##########################################################

def main():
    # Create a World instance to hold everything:
    world = esper.World()

    # Instantiate a Processor (or more), and add them to the world:
    movement_processor = MovementProcessor()
    world.add_processor(movement_processor)

    cooldown_processor = CooldownProcessor()
    world.add_processor(cooldown_processor)
    
    view_processor = ViewProcessor()
    world.add_processor(view_processor)
    # step 1 : pick situations compositions. (positionning, amount of heroes)
    # step 2 : pick heroes.
    # step 3 : pick an action for each hero at every step. (movement, ability)
    # step 4a : calculate the stats for each outcome, and continue as step 3 until end of simulation (max amount of steps? full wipe out?)
    # step 4b : log every step actions as a sequence during the simulation.
    # step 4c : calculate performance metrics considering each sides stats and performed actions after each step.
    # step 5: write up all possibilities and keep the results. 
    # Create entities, and assign Component instances to them:
    hero1Team1 = world.create_entity(Hero(name="1"),Team(team=1), Position(x=0, y=5))
    hero2Team2 = world.create_entity(Hero(name="2"),Team(team=2), Position(x=0, y=-5))
    
    #player = world.create_entity()
    #world.add_component(player, Velocity(x=0.9, y=1.2))
    #world.add_component(player, Position(x=5, y=5))

    # A dummy main loop:
    try:
        while True:
            # Call world.process() to run all Processors.
            # TODO add random moves to check if the data flow is working. 
            world.process()
            time.sleep(1) # seconds.

    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    print("\nHeadless Example. Press Ctrl+C to quit!\n")
    main()

