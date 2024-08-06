from tkinter import *
from libemg.streamers import myo_streamer
from libemg.gui import GUI
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler, RegexFilter
from libemg.utils import make_regex
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import OnlineEMGClassifier, EMGClassifier
from isofitts import FittsLawTest

class Menu:
    def __init__(self):
        streamer, sm = myo_streamer()

        # Create online data handler to listen for the data
        self.shared_memory = sm

        self.classifier = None
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
        frame = Frame(self.window)
        Label(self.window, text="Model:", font=("Arial bold", 18)).pack(in_=frame, side=LEFT, padx=(0,10))
        Entry(self.window, font=("Arial", 18), textvariable=self.model_str).pack(in_=frame, side=LEFT)
        frame.pack(pady=(20,10))
            

    def start_test(self):
        self.window.destroy()
        self.set_up_classifier()
        FittsLawTest(num_trials=8, num_circles=8, savefile=self.model_str.get() + ".pkl").run()
        # Its important to stop the classifier after the game has ended
        # Otherwise it will continuously run in a seperate process
        self.classifier.stop_running()
        self.initialize_ui()

    def launch_training(self):
        self.window.destroy()
        training_ui = GUI(self.odh, width=700, height=700, gesture_height=300, gesture_width=300)
        training_ui.download_gestures([1,2,3,4,5], "images/")
        training_ui.start_gui()
        self.initialize_ui()

    def set_up_classifier(self):
        WINDOW_SIZE = 40
        WINDOW_INCREMENT = 20

        # Step 1: Parse offline training data
        dataset_folder = 'data/'
        regex_filters = [
            RegexFilter(left_bound = "C_", right_bound="_R", values = ["0","1","2","3","4"], description='classes'),
            RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = ["0", "1", "2"], description='reps'),
        ]

        odh = OfflineDataHandler()
        odh.get_data(dataset_folder, regex_filters, delimiter=",")
        train_windows, train_metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)

        # Step 2: Extract features from offline data
        fe = FeatureExtractor()
        feature_list = fe.get_feature_groups()['HTD']
        training_features = fe.extract_features(feature_list, train_windows)

        # Step 3: Dataset creation
        data_set = {}
        data_set['training_features'] = training_features
        data_set['training_labels'] = train_metadata['classes']

        # Step 4: Create the EMG Classifier
        o_classifier = EMGClassifier(model=self.model_str.get())
        o_classifier.fit(feature_dictionary=data_set)

        # Step 5: Create online EMG classifier and start classifying.
        self.classifier = OnlineEMGClassifier(o_classifier, WINDOW_SIZE, WINDOW_INCREMENT, OnlineDataHandler(self.shared_memory), feature_list)
        self.classifier.run(block=False) # block set to false so it will run in a seperate process.

    def on_closing(self):
        # Clean up all the processes that have been started
        self.window.destroy()

if __name__ == "__main__":
    menu = Menu()