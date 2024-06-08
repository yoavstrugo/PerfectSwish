import tkinter as tk

from screeninfo import get_monitors


def get_root_screen(root: tk.Tk):
    """
    Get the screen on which the root window is located.
    :return:
    """
    for monitor in get_monitors():
        if monitor.x <= root.winfo_x() <= monitor.x + monitor.width and \
                monitor.y <= root.winfo_y() <= monitor.y + monitor.height:
            return monitor
    return None

def get_other_screen(control_screen):
    """
    Get the screen other than the given one, under the assumption that there are only two screens.
    :param control_screen:
    :return:
    """
    for monitor in get_monitors():
        if monitor != control_screen:
            return monitor

    raise ValueError("There are not exactly two screens.")