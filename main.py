from Logic import *
from PreProcessing import *
import warnings

if __name__ == "__main__":
     warnings.filterwarnings("ignore")

     regression = True

     #regression
     pre = PreProccessing('Movies_testing.csv','Movies_testing_classification.csv',regression=regression)
     data_set = pre.moviesFile
     istest = True
     Logic(data_set, pre,istest, regression= regression )

     #classification
     pre = PreProccessing('Movies_testing.csv','Movies_testing_classification.csv',regression=not regression)
     data_set = pre.moviesFile
     Logic(data_set, pre, istest, regression=not regression)



