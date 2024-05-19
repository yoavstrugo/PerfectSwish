import tkinter as tk
from tkinter import Canvas
import pickle

# Set initial positions for the points
initial_positions = [(100, 100), (1000, 100), (1000, 600), (100, 600)]


class MovablePoints:
    def __init__(self, root):
        self.root = root
        # full screen canvas
        self.canvas = Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
        self.canvas.pack()

        self.points = []
        self.create_points()

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.select_point)
        self.canvas.bind("<B1-Motion>", self.move_point)

        # Bind the close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.selected = None

    def create_points(self):
        """Create initial points on the canvas."""

        for pos in initial_positions:
            x, y = pos
            r = 5  # Radius of the circle
            point = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='red', outline='red', tags='point')
            # add number near the point that will move with the point



            self.points.append((point, (x, y)))

    def select_point(self, event):
        """Select a point to move."""
        # Detect if a point was clicked
        self.selected = None
        for point, (x, y) in self.points:
            if abs(event.x - x) < 10 and abs(event.y - y) < 10:
                self.selected = point
                return

    def move_point(self, event):
        """Move the selected point with mouse drag."""
        if self.selected:
            r = 5
            self.canvas.coords(self.selected, event.x - r, event.y - r, event.x + r, event.y + r)
            # Update the stored position
            for i, (point, pos) in enumerate(self.points):
                if point == self.selected:
                    self.points[i] = (point, (event.x, event.y))
                    break

    def on_close(self):
        """Handle the window close event."""
        # Save positions to a file
        positions = [pos for _, pos in self.points]
        with open('point_positions.pkl', 'wb') as f:
            pickle.dump(positions, f)
        self.root.destroy()


# Create the main window and pass it to the MovablePoints class
root = tk.Tk()
app = MovablePoints(root)
root.mainloop()
# print the positions
with open('point_positions.pkl', 'rb') as f:
    positions = pickle.load(f)
    print(positions)