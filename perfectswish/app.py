import tkinter as tk

from perfectswish.api import data_io, webcam
from perfectswish.api.monitor_utils import get_other_screen, get_root_screen
from perfectswish.phases.crop_phase import CropPhase
from perfectswish.phases.game_phase import GamePhase
from perfectswish.phases import ProjectAlignPhase

DATA_FILE = 'app_data.dat'
CAMERA = 0


class PerfectSwishApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perfect Swish")
        self.geometry("800x600")

        self.__frames: list[tuple[tk.Frame, tk.Toplevel | None]] = []
        self.__cached_data = None
        self.load_data()
        self._webcam_capture = webcam.WebcamCapture(CAMERA, 1920, 1080, 24)
        self.resizable(False, False)

        self._phase = None
        self.__root_screen = get_root_screen(self)
        self.__other_screen = get_other_screen(self.__root_screen)

        self.protocol('WM_DELETE_WINDOW', self.quit)
        self.crop_phase()

    @staticmethod
    def phase_switch(func):
        def wrapper(self, *args, **kwargs):
            if self._phase:
                name = self._phase.name
                data = self._phase.get_data()
                self.save_data(name, data)
                self._phase.destroy()
            func(self, *args, **kwargs)

        return wrapper

    def quit(self):
        if self._phase:
            name = self._phase.name
            data = self._phase.get_data()
            self.save_data(name, data)
            self._phase.destroy()
        self.destroy()
        self._webcam_capture.release()

    def load_data(self):
        try:
            self.__cached_data = data_io.load_data(DATA_FILE)
            if self.__cached_data is None:
                self.__cached_data = dict()
        except FileNotFoundError:
            self.__cached_data = dict()

    def save_data(self, key, data):
        self.__cached_data[key] = data
        data_io.save_data(DATA_FILE, self.__cached_data)

    def set_crop_rect(self, pts):
        self.crop_rect = pts

    @phase_switch
    def crop_phase(self):
        data = self.__cached_data.get('crop')
        self._phase = CropPhase(self, data, self.project_align_phase, None, self._webcam_capture)

    @phase_switch
    def project_align_phase(self):
        data = self.__cached_data.get('project_align')
        crop_data = self.__cached_data.get('crop')
        self._phase = ProjectAlignPhase(self, self.__other_screen, data, crop_data, self._webcam_capture,
                                        self.game_phase, self.crop_phase)

    @phase_switch
    def game_phase(self):
        crop_data = self.__cached_data.get('crop')
        project_data = self.__cached_data.get('project_align')
        self._phase = GamePhase(self, crop_data, project_data, self._webcam_capture, self.__other_screen,
                                None, None)


if __name__ == '__main__':
    app = PerfectSwishApp()
    app.mainloop()