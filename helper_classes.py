class Rectangle():
    """
    This class gives the coordinates for the corners of a rectangle, allowing all rectangles to conform to the same super class.
    """
    def __init__(self, start_x: int, start_y: int, end_x: int, end_y: int):
            self.start_x = start_x
            self.start_y = start_y
            self.end_x = end_x
            self.end_y = end_y


    def top_left_corner(self):
        return (self.start_x, self.start_y)
    
    def bottom_right_corner(self):
        return (self.end_x, self.end_y)