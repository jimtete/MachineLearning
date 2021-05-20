import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
        self.firstClassX = None
        self.firstClassY = None
        self.secondClassX = None
        self.secondClassY = None

    def fit(self, X, y,G):
        n_samples, n_features = X.shape
        # init parameters
        
        self.weights = np.zeros(n_features)+0.1
        self.bias = -1
        epoch=0
        keepLooping=True
        y_ = np.array([1 if i > 0 else 0 for i in y])

        while(keepLooping):
            epoch+=1
            
            
            
            for idx, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)    
                if (y_predicted!=y_).any():
                    # Perceptron update rule
                    update = self.lr * (y_[idx] - y_predicted)
                        
                    weigtsOld = self.weights
                    self.weights += update * x_i
                    self.bias += update
                    
            if (G==2):
                ##Δημιουργία διαχωριστικής ευθείας
            
                x1,x2 = np.amin(X[:,0]),np.amax(X[:,0])
                y1,y2=self.calculateLine(x1),self.calculateLine(x2)
                
                ##Εκκίνηση αποθήκευσης
                self.firstClassX,self.firstClassY = np.array([]),np.array([])
                self.secondClassX,self.secondClassY = np.array([]),np.array([])
                for i in range(len(X)):
                    testX,testY = X[i]
                    predY = self.calculateLine(testX)
                    if (testY>predY):
                        self.secondClassX,self.secondClassY = np.append(self.secondClassX,testX),np.append(self.secondClassY,testY)
                    else:
                        self.firstClassX,self.firstClassY = np.append(self.firstClassX,testX),np.append(self.firstClassY,testY)
            
                        
                self.generateplotsgraph2(X,epoch,[x1,x2,y1,y2]) 
            
            if (G==3):
                predictions = self.predict(X)
                showedARanger,showedBRanger,showedA,showedB = np.array([]),np.array([]),np.array([]),np.array([])
                isARanger,isBRanger,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])
                for i in range(len(X)):
                    temp = predictions[i]
                    tempReal = y[i]
                    
                    if (tempReal==0):
                        isARanger,isA = np.append(isARanger,i),np.append(isA,tempReal)
                        showedARanger,showedA = np.append(showedARanger,i),np.append(showedA,temp)            
                    else:
                        isBRanger,isB = np.append(isBRanger,i),np.append(isB,tempReal)
                        showedBRanger,showedB = np.append(showedBRanger,i),np.append(showedB,temp)
                        
                self.generateplotsgraph3(showedARanger,showedA,showedBRanger,
                                         showedB,isARanger,isA,isBRanger,isB)
            
            if (epoch>self.n_iters):
                keepLooping=False
            
            
            
                
                

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)
    
    def calculateLine(self,aot):
        return (-self.weights[0]*aot-self.bias) / self.weights[1]
    
    def generateplotsgraph3(self, SRA,SA,SRB,SB,IRA,IA,IRB,IB):
        fig3 = plt.figure()
        fig3 = plt.plot(SRA,SA,"mx",label="Πρόβλεψε 0")
        fig3 = plt.plot(SRB,SB,"gx",label="Πρόβλεψε 1")
        fig3 = plt.plot(IRA,IA,"mo",label="Είναι 0",MarkerFaceColor='none')
        fig3 = plt.plot(IRB,IB,"go",label="Είναι 1",MarkerFaceColor='none')
        
        
        plt.legend()
        
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
