import tkinter as tk

from perfectswish.phases.crop_phase import CropPhase
from perfectswish.phases.game_phase import GamePhase
from perfectswish.phases.logo_phase import LogoPhase
from perfectswish.phases.projection_align_phase import ProjectAlignPhase
from perfectswish.settings import DEFAULT_APP_GEOMETRY
from perfectswish.utils import data_io, webcam
from perfectswish.utils.monitor_utils import get_other_screen, get_root_screen

DATA_FILE = 'app_data.dat'
CAMERA = 1


class PerfectSwishApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perfect Swish")
        self.geometry(DEFAULT_APP_GEOMETRY)

        self.__frames: list[tuple[tk.Frame, tk.Toplevel | None]] = []
        self.__cached_data = None
        self.load_data()
        # self._webcam_capture = webcam.WebcamCapture(
        #     r'C:\Users\TLP-266\PyCharmProject\PerfectSwish\videos\full_balls_no_green.mp4', 1920, 1080, 15)
        self._webcam_capture = webcam.WebcamCapture(CAMERA, 1920, 1080, 15)
        self.resizable(False, False)

        self._phase = None
        self.__root_screen = get_root_screen(self)
        self.__other_screen = get_other_screen(self.__root_screen)

        self.protocol('WM_DELETE_WINDOW', self.quit)
        # if self.__cached_data.get('project_align') is not None:
        #     self.logo_phase()
        # else:
        self.crop_phase()

    @staticmethod
    def phase_switch(gemoetry=DEFAULT_APP_GEOMETRY):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if self._phase:
                    name = self._phase.name
                    data = self._phase.get_data()
                    self.save_data(name, data)
                    self._phase.destroy()
                self.geometry(gemoetry)
                func(self, *args, **kwargs)

            return wrapper

        if callable(gemoetry):
            func = gemoetry
            gemoetry = DEFAULT_APP_GEOMETRY
            return decorator(func)
        else:
            return decorator

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
    def logo_phase(self):
        self._phase = LogoPhase(self, self.__cached_data.get('project_align'), self.__other_screen,
                                self.project_align_phase)

    @phase_switch
    def crop_phase(self):
        data = self.__cached_data.get('crop')
        self._phase = CropPhase(self, data, self.project_align_phase, None,
                                self._webcam_capture)

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
        self._phase = GamePhase(self, crop_data, project_data, self._webcam_capture.create_multiprocess(),
                                self.__other_screen, fps=24, balls_update_rate=24)


if __name__ == '__main__':
    app = PerfectSwishApp()
    app.mainloop()
