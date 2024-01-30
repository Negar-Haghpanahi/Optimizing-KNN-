from input_class import Context ,Separate
import scipy.io
import pandas as pd

class ReadFile():

    def __init__(self , path):
        
        self.context_sampel = Separate()
        self.mat_file = scipy.io.loadmat(path)           
        self.variables = self.mat_file.keys()
        # print(self.variables)
        self.context_sampel.df = pd.DataFrame(self.mat_file['X'])
        y = pd.DataFrame(self.mat_file['Y'])
        y = y.rename(columns={0: 'label'})
        self.context_sampel.ylabel = y
        # self.context_sampel.concat_df = pd.concat([self.context_sampel.df ,y ] ,axis=1)
        
        
      
    def get_obj(self):
            
        return self.context_sampel