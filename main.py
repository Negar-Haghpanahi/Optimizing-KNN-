from readfile import ReadFile
from preparation import Prepare
from input_class import Separate
from Model import Knn

def Read():
    
    df_file=ReadFile('E:\\Project\\KNN\\BASEHOCK.mat')  
    return df_file.get_obj()
   
def Preparation(prep_object : Separate) :
    
    pre_class =Prepare(prep_object)
    # pre_class.BoxPlot()
    null_value=pre_class.isnull()
    if null_value == 0:
        pre_class.Check_Label()
        # pre_class.LowVarianceFilter()
        pre_class.Split()
        # pre_class.Standard()  
        # pre_class.RandomForest()
        # pre_class.Optimal_Componenct()  
        # pre_class.Dimention_Reduction_PCA()
        
    return pre_class.get_obj()



if __name__ == "__main__":
    
    sample_obj =Read()
    final_obj =Preparation(sample_obj)
    knn_obj = Knn(final_obj)
    knn_obj.phase_1()
    knn_obj.phase_2()
    knn_obj.phase_3()
    knn_obj.phase_4()
