import pickle
import sys
import os

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import seaborn as sns

feature_list = ['Machine', 'NumberOfSections', 
    'PointerToSymbolTable', 'NumberOfSymbols', 
    'SizeOfOptionalHeader', 'Characteristics', 
    'Magic', 'MajorLinkerVersion', 'MinorLinkerVersion', 
    'SizeOfCode', 'SizeOfInitializedData', 
    'SizeOfUninitializedData', 'AddressOfEntryPoint', 
    'BaseOfCode', 'BaseOfData', 'ImageBase', 
    'SectionAlignment', 'FileAlignment', 'MajorOperatingSystemVersion', 
    'MinorOperatingSystemVersion', 'MajorImageVersion', 
    'MinorImageVersion', 'MajorSubsystemVersion', 
    'MinorSubsystemVersion', 'Reserved1', 'SizeOfImage', 
    'SizeOfHeaders', 'CheckSum', 'Subsystem', 'DllCharacteristics', 
    'SizeOfStackReserve', 'SizeOfStackCommit', 'SizeOfHeapReserve', 
    'SizeOfHeapCommit', 'LoaderFlags', 'NumberOfRvaAndSizes', 
    'IMAGE_DIRECTORY_ENTRY_EXPORT_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_EXPORT_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_IMPORT_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_IMPORT_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_RESOURCE_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_RESOURCE_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_EXCEPTION_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_EXCEPTION_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_SECURITY_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_SECURITY_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_BASERELOC_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_BASERELOC_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_DEBUG_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_DEBUG_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_COPYRIGHT_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_COPYRIGHT_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_GLOBALPTR_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_GLOBALPTR_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_TLS_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_TLS_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT_SIZE', 
    'IMAGE_DIRECTORY_ENTRY_IAT_VIRTUAL_ADDRESS', 
    'IMAGE_DIRECTORY_ENTRY_IAT_SIZE', 'IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT_VIRTUAL_ADDRESS', 'IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT_SIZE', 'IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR_VIRTUAL_ADDRESS', 'IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR_SIZE', 'IMAGE_DIRECTORY_ENTRY_RESERVED_VIRTUAL_ADDRESS', 'IMAGE_DIRECTORY_ENTRY_RESERVED_SIZE']

class ML:
    def __init__(self, model_path=""):
        self.model_path = model_path
        self.model = None

    def set_model_path(self, model_path):
        self.model_path = model_path

    def _check_model_path(self):
        if self.model_path == None:
            return False
        return True

    def get_model(self):
        return self.model

    def _load_data(self, csv_path):
        df = pd.read_csv(csv_path, header=None)
        file_name_list, label = df[0].values, df[1].values
        ret = []
        for file_name in file_name_list:
            with open(file_name, 'rb') as f:
                ret.append(pickle.load(f))

        return file_name_list, label, np.array(ret)

    def save_model(self):
        if not self._check_model_path():
            print("'model_path' is not set.")
            print("Save as 'model.dat'.")
            self.model_path = "model.dat"

        if not os.path.isfile(self.model_path):
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
        else:
            print(".")

    def load_model(self, model_path=""):
        if not self._check_model_path():
            if model_path == "":
                print("Have to set model path.")
                return
            self.model_path = model_path

            return 0
        
        with open(self.model_path, 'rb') as f:
            print(f"load model {self.model_path}")
            self.model = pickle.load(f)

            return 1

    def train(self, train_csv):
        if not os.path.isfile(self.model_path):
            _, labels, features = self._load_data(train_csv)
            self.model.fit(features, labels)
        else:
            print("..")
    
    def predict(self, predict_csv): # csv
        file_names, _, features = self._load_data(predict_csv)
        return (file_names, self.model.predict(features))

    def valid(self, valid_csv):
        _, labels, features = self._load_data(valid_csv)
        prediction = self.model.predict(features)
        print("Accuracy :", accuracy_score(labels, prediction))
        print("Precision :", precision_score(labels, prediction))
        print("Recall :", recall_score(labels, prediction))
    
    def draw_feature_importance(self, title_name, save_path):
        feature_imp = pd.DataFrame(sorted(zip(self.model.feature_importances_,feature_list)), columns=['Value','Feature'])

        plt.figure(figsize=(20, 10))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.title(title_name)
        plt.tight_layout()
        plt.savefig(save_path)

class LightGbm(ML):
    """Gbm

    example:

    """
    def __init__(self, model_path="", params=""):
        super().__init__(model_path=model_path)

        if os.path.isfile(model_path):
            super().load_model(model_path)# -> 바꾸기
        else:
            self.build_model()

    
    def build_model(self):
        self.model = lgb.LGBMClassifier()
    
    
class XGboost(ML):
    """XGboost

    example:

    """
    def __init__(self, model_path=""):
        super().__init__(model_path=model_path)

        if os.path.isfile(model_path):
            super().load_model(model_path)# -> 바꾸기
        else:
            self.build_model()
    
    def build_model(self):
        self.model = XGBClassifier()
    

class Forest(ML):
    """Forest

    example:

    """
    def __init__(self, model_path=""):
        super().__init__(model_path=model_path)

        if os.path.isfile(model_path):
            super().load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        self.model = RandomForestClassifier()



if __name__ == "__main__":
    # sample code
    train_cvs_path = sys.argv[1]
    test_csv_path = sys.argv[1]

    model_path = ""
    rf = LightGbm(model_path=model_path)
    rf.train(train_cvs_path)
    rf.valid(train_cvs_path)

