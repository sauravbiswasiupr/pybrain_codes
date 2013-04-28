#!/usr/bin/python 
'''Script to read in each MNIST image and make a time sequence of each image abd try to recognize it using the ouptut of the last time step in that time sequence '''
__author__="Saurav Biswas" 
__version__="1.0" 


#fixing imports 
from pylab import plot , hold , show 
from pybrain.datasets import SequenceClassificationDataSet 
from pybrain.structure.modules import LSTMLayer , SoftmaxLayer 
from pybrain.supervised import BackpropTrainer 
from pybrain.tools.validation import testOnSequenceData
from pybrain.tools.shortcuts import buildNetwork 
from read_mnist import * 
import cPickle 
from read_mnist import * 
import time 
import sys 

def  read_data_MNIST(timesteps):
   f =  open("mnist.pkl","rb") 
   print "MNIST data unpickled ....." 
   print "Proceeding to create training and testing datasets ..." 
   training , validation , test = cPickle.load(f) 
   trndata = create_training_data(training,timesteps)
   validdata = create_training_data(validation,timesteps)
   tstdata = create_test_data(test)
   trndata._convertToOneOfMany() 
   validdata._convertToOneOfMany()
   tstdata._convertToOneOfMany() 
   return (trndata,validdata,tstdata) 

def create_network(timesteps):
    trndata ,validdata, tstdata= read_data_MNIST(timesteps)
    rnn =  buildNetwork(trndata.indim , 20 , trndata.outdim , hiddenclass=LSTMLayer , outclass = SoftmaxLayer , outputbias = True , recurrent = True ) 
    #20 is the number of LSTM blocks in the hidden layer 
    #we use the BPTT algo to train 
    
    trainer = BackpropTrainer(rnn,dataset=trndata ,verbose=True , momentum = 0.9 , learningrate=0.00001)
    print "Training started ..." 
    t1 = time.clock() 
    # trainer.trainEpochs(10)
    trainer.trainUntilConvergence( maxEpochs=1000)
    t2 = time.clock() 
    print "Training 1000 epochs took :  ", (t2-t1)/60.0 , "minutes "  
    #train for 1000 epochs 
    trnresult = 100. * (1.0 - testOnSequenceData(rnn , trndata))
    tstresult = 100. * (1.0 - testOnSequenceData(rnn, tstdata))
    print "Train Error : %5.2f%%" %trnresult , " , test error :%5.2f%%" %tstresult 
  

if __name__ =="__main__":
   
   timesteps = int(sys.argv[1])
   print "Using " , timesteps , "timesteps ... " 
   create_network(timesteps)
      
