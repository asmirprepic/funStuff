import numpy as np

class Blob:
  def __init__(self,SIZE):
    
    self.SIZE = SIZE
    self.x = np.random.randint(0,SIZE)
    self.y = np.random.randint(0,SIZE)
    
    

  def __str__(self):
    return f"{self.x,self.y}"

  def __sub__(self,other):
    return (self.x-other.x,self.y-other.y)

  def action(self,choice):
    movement_options = {
            0: (0, -1),   # Up
            1: (1, 0),    # Right
            2: (0, 1),    # Down
            3: (-1, 0),   # Left
            4: (1, -1),   # Up-Right
            5: (1, 1),    # Down-Right
            6: (-1, -1),  # Up-Left
            7: (-1, 1),   # Down-Left
            8: (0, 0),    # No movement
        }
    self.move(*movement_options.get(choice, (0, 0)))
    return movement_options.get(choice, (0, 0))

  def move(self,x = None,y=None):
    if x is None:
        self.x += np.random.randint(-1, 2)
    else:
        self.x += x

    if  y is None:
        self.y += np.random.randint(-1, 2)
    else:
        self.y += y

    # Boundary checks
    self.x = max(0, min(self.x, self.SIZE - 1))
    self.y = max(0, min(self.y, self.SIZE - 1))
    

  def reset_position(self):
    self.x = np.random.randint(0,self.SIZE)
    self.y = np.random.randint(0,self.SIZE)