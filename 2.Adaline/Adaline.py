##για διαγραφή
import numpy as np
import matplotlib.pyplot as plt

class Adaline(object):
    
    def __init__(self, lr = 0.2, epochs = 50):
        self.epochs=epochs
        self.lr=lr        
        
        self.firstClassX = None
        self.firstClassY = None
        self.secondClassX = None
        self.secondClassY = None

    
    def fit(self,input,values, G):
        
        self.w_ = np.ones(input.shape[1])
        self.cost_ = []
        
        keepLooping=True
        i = 0
        
        
        while(keepLooping):
            i+=1
            
            output = self.net_input(input)
            
            
            
            
            errors = values-output
            self.w_ += self.lr * input.T.dot(errors)
            
            
            
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
            
            
            
            
            if (cost<0.01):
                keepLooping=False
            
            if (i>self.epochs):
                keepLooping=False
            
            if (G==2):
                x1,x2 = 0,1
                y1,y2 = self.calculateLine(x1), self.calculateLine(x2)
                
                self.firstClassX, self.firstClassY = np.array([]),np.array([])
                self.secondClassX, self.secondClassY = np.array([]), np.array([])
                
                for j in range(len(input)):
                    testX,testY = input[j,[1,2]]
                    predY = self.calculateLine(testX)
                    if (testY>predY):
                        self.secondClassX,self.secondClassY = np.append(self.secondClassX,testX),np.append(self.secondClassY,testY)
                    else:
                        self.firstClassX,self.firstClassY = np.append(self.firstClassX,testX),np.append(self.firstClassY,testY)
                
                self.generateplotsgraph2(input,i,[x1,x2,y1,y2])
            
            
        
        return self 
    
    def net_input(self, input):
        
        """Calculating net input"""
        
        return np.dot(input, self.w_)
    
    def activation(self, input):
        
        """ Computer linear activation"""
        
        return self.net_input(input)
    
    def predict(self, input):
        return  np.where(self.activation(input) >= 0.0, 1,0)
    
    def calculateLine(self,aot):
        return (-self.w_[1]*aot)/self.w_[2] + self.w_[0]/self.w_[2]
    
    def generateplotsgraph2(self, X, epoch,line):
        
        fig = plt.figure()
        fig = plt.title("Εποχή: "+str(epoch))
        fig = plt.plot(self.firstClassX,self.firstClassY,"r.",label="Κλάση 1")
        fig = plt.plot(self.secondClassX,self.secondClassY,"bo",label="Κλάση 2")
        fig = plt.plot(line[0:2],line[2:4],'k-')
        
        plt.legend()
        plt.xlim([0,1])
        plt.ylim([0,1])

        return None
    
    def generateplotsgraph3(self, SRA,SA,SRB,SB,IRA,IA,IRB,IB):
        fig3 = plt.figure()
        fig3 = plt.plot(SRA,SA,"mx",label="Πρόβλεψε 0")
        fig3 = plt.plot(SRB,SB,"gx",label="Πρόβλεψε 1")
        fig3 = plt.plot(IRA,IA,"mo",label="Είναι 0",MarkerFaceColor='none')
        fig3 = plt.plot(IRB,IB,"go",label="Είναι 1",MarkerFaceColor='none')
        
        
        plt.legend()
    
    