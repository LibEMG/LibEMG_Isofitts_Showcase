from pathlib import Path

from tkinter import *
from libemg.streamers import myo_streamer
from libemg.gui import GUI
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler, RegexFilter, FilePackager
from libemg.utils import make_regex
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import OnlineEMGClassifier, EMGClassifier, EMGRegressor, OnlineEMGRegressor
from libemg.environments.isofitts import IsoFitts
from libemg.environments.controllers import ClassifierController, RegressorController

class Menu:
    def __init__(self):
        streamer, sm = myo_streamer()

        # Create online data handler to listen for the data
        self.odh = OnlineDataHandler(sm)

        self.model = None
        self.model_str = None

        self.window = None
        self.initialize_ui()
        self.window.mainloop()

    def initialize_ui(self):
        # Create the simple menu UI:
        self.window = Tk()
        if not self.model_str:
            self.model_str = StringVar(value='LDA')
        else:
            self.model_str = StringVar(value=self.model_str.get())
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.title("Game Menu")
        self.window.geometry("500x250")

        # Label 
        Label(self.window, font=("Arial bold", 20), text = 'LibEMG - Isofitts Demo').pack(pady=(10,20))
        # Train Model Button
        Button(self.window, font=("Arial", 18), text = 'Get Training Data', command=self.launch_training).pack(pady=(0,20))
        # Start Isofitts
        Button(self.window, font=("Arial", 18), text = 'Start Isofitts', command=self.start_test).pack()
        
        # Model Input
        self.model_type = IntVar()
        r1 = Radiobutton(self.window, text='Classification', variable=self.model_type, value=1)
        r1.pack()
        r1.select() # default to classification
        r2 = Radiobutton(self.window, text='Regression', variable=self.model_type, value=2)
        r2.pack()

        frame = Frame(self.window)
        Label(self.window, text="Model:", font=("Arial bold", 18)).pack(in_=frame, side=LEFT, padx=(0,10))
        Entry(self.window, font=("Arial", 18), textvariable=self.model_str).pack(in_=frame, side=LEFT)
        frame.pack(pady=(20,10))

    def start_test(self):
        self.window.destroy()
        self.set_up_model()
        if self.regression_selected():
            controller = RegressorController()
        else:
            controller = ClassifierController(output_format=self.model.output_format, num_classes=5)
        IsoFitts(controller, num_trials=8, num_circles=8, save_file=self.model_str.get() + ".pkl").run()
        # Its important to stop the classifier after the game has ended
        # Otherwise it will continuously run in a seperate process
        self.model.stop_running()
        self.initialize_ui()

    def regression_selected(self):
        return self.model_type.get() == 2
    
    def launch_training(self):
        self.window.destroy()
        if self.regression_selected():
            args = {'media_folder': 'animation/', 'data_folder': Path('data', 'regression').absolute().as_posix(), 'rep_time': 50}
        else:
            args = {'media_folder': 'images/', 'data_folder': Path('data', 'classification').absolute().as_posix()}
        training_ui = GUI(self.odh, args=args, width=700, height=700, gesture_height=300, gesture_width=300)
        training_ui.download_gestures([1,2,3,4,5], "images/")
        training_ui.start_gui()
        self.initialize_ui()

    def set_up_model(self):
        WINDOW_SIZE = 40
        WINDOW_INCREMENT = 5

        # Step 1: Parse offline training data
        if self.regression_selected():
            regex_filters = [
                RegexFilter(left_bound='regression/C_0_R_', right_bound='_emg.csv', values=['0', '1', '2'], description='reps')
            ]
            metadata_fetchers = [
                FilePackager(RegexFilter(left_bound='animation/', right_bound='.txt', values=['collection'], description='labels'), package_function=lambda x, y: True)
            ]
            labels_key = 'labels'
            metadata_operations = {'labels': 'last_sample'}
        else:
            regex_filters = [
                RegexFilter(left_bound = "classification/C_", right_bound="_R", values = ["0","1","2","3","4"], description='classes'),
                RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = ["0", "1", "2"], description='reps'),
            ]
            metadata_fetchers = None
            labels_key = 'classes'
            metadata_operations = None

        odh = OfflineDataHandler()
        odh.get_data('./', regex_filters, metadata_fetchers=metadata_fetchers, delimiter=",")
        train_windows, train_metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT, metadata_operations=metadata_operations)

        # Step 2: Extract features from offline data
        fe = FeatureExtractor()
        feature_list = fe.get_feature_groups()['HTD']
        training_features = fe.extract_features(feature_list, train_windows, array=True)

        # Step 3: Dataset creation
        data_set = {}
        data_set['training_features'] = training_features
        data_set['training_labels'] = train_metadata[labels_key]

        # Step 4: Create the EMG Classifier
        model = self.model_str.get()
        if self.regression_selected():
            # Regression
            emg_model = EMGRegressor(model=model)
            emg_model.fit(feature_dictionary=data_set)
            self.model = OnlineEMGRegressor(emg_model, WINDOW_SIZE, WINDOW_INCREMENT, self.odh, feature_list)
        else:
            # Classification
            emg_model = EMGClassifier(model=model)
            emg_model.fit(feature_dictionary=data_set)
            self.model = OnlineEMGClassifier(emg_model, WINDOW_SIZE, WINDOW_INCREMENT, self.odh, feature_list)

        # Step 5: Create online EMG classifier and start classifying.
        self.model.run(block=False) # block set to false so it will run in a seperate process.

    def on_closing(self):
        # Clean up all the processes that have been started
        self.window.destroy()

if __name__ == "__main__":
    menu = Menu()
    # maybe add a CLI here to determine classification or regression?
    # need to upload some regression videos or something... either in repo or setup so it'll automatically download them
