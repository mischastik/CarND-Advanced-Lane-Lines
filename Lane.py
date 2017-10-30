import numpy as np

class Line():
    def __init__(self, iir_filter_weight = 0.25):
        self.history_length = 5
        # was the line detected in the last iteration?
        self.detected = False
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #polynomial coefficients for the most recent fit (metric)
        self.current_fit_cr = [np.array([False])]
        #polynomial coefficients of recent fits
        self.recent_fits = []
        #polynomial coefficients of recent fits (metric)
        self.recent_fits_cr = []
        self.iir_average_fit = None
        self.iir_filter_weight = iir_filter_weight



        # x values of the last fit of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def add_valid_fit(self, fit, fit_cr):
        self.detected = True
        self.current_fit = fit
        self.current_fit_cr = fit_cr
        self.recent_fits.append(fit)
        self.recent_fits_cr.append(fit_cr)
        if len(self.recent_fits) > self.history_length:
            self.recent_fits.pop(0)
            self.recent_fits_cr.pop(0)
        if self.iir_average_fit is None:
            self.iir_average_fit = fit
        else:
            self.iir_average_fit = self.iir_average_fit * (1.0 - self.iir_filter_weight) + fit * self.iir_filter_weight

    def get_fir_average_fit(self):
        if len(self.recent_fits) == 0:
            return None
        avg_fit = np.zeros_like(self.recent_fits[0])
        for item in self.recent_fits:
            avg_fit = avg_fit + item
        avg_fit /= len(self.recent_fits)

        return avg_fit

    def get_fir_average_fit_cr(self):
        if len(self.recent_fits_cr) == 0:
            return None
        avg_fit = np.zeros_like(self.recent_fits_cr[0])
        for item in self.recent_fits_cr:
            avg_fit = avg_fit + item
        avg_fit /= len(self.recent_fits_cr)

        return avg_fit

