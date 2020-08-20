#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import exp, isnan
import numpy as np

class DynamicModel1(object):
    def __init__(self):
        self._d_nrTimeSteps = 0
        self.currentStep = 0
     
    def currentTimeStep(self):
        return self.currentStep

    def _setCurrentTimeStep(self,
                            step):
        self.currentStep = step

    def nrTimeSteps(self):
        return self._d_nrTimeSteps

    def _setNrTimeSteps(self,
                        timeSteps):
        self._d_nrTimeSteps = timeSteps

    def initial(self):
        print("Implement 'initial' method")

    def dynamic(self):
        print("Implement 'dynamic' method")


class DynamicFramework1(object):
  
    def __init__(self,
                 userModel,
                 lastTimeStep=0,
                 firstTimestep=1):
        self._d_model = userModel
        self._userModel()._setNrTimeSteps(lastTimeStep)
        
    def _runInitial(self):
        self._userModel().initial()

    def _runDynamic(self):
        self._userModel().dynamic()

    def _userModel(self):
        """
        Return the model instance provided by the user.
        """
        return self._d_model

    def run(self):
        self._runInitial()
        i = 0
        for i in range(self._userModel().nrTimeSteps()):
            self._userModel()._setCurrentTimeStep(i + 1)
            self._runDynamic()

        return 0

class MyFirstModel(DynamicModel1):
    def __init__(self, params_x):
        DynamicModel1.__init__(self)
        self.params_x = params_x
        # initial values
    def initial(self):        
        self.maxSW1 = 76.0
        self.maxSW2 = 82.0
        self.SW1 = self.inisw1 = float(60)
        self.SW2 = self.inisw2 = float(76)
        self.P = 0.0
        self.P1 = 0.0
        self.S=0
        self.concs1mo= -7.13  # initial snow interception storate isotope composition
        self.concs1imo = -11.59 # initial snow interception storate isotope composition
        self.concs1to = -9.36 # initial snow interception storate isotope composition
        self.concs2mo = -9.52    # initial snow interception storate isotope composition
        self.concs2imo = -12.18  # initial snow interception storate isotope composition
        self.concs2to = -10.85   # initial snow interception storate isotope composition       
        self.concg = float(-8)# isotope concentration groundwater upper store
        self.conci = float(-8) # isotope concentration discharge            
        self.OF = float(0)
        self.WP1 = float(40)  # withering point at Layer 1
        self.WP2 = float(50)  # withering point at Layer 2
        self.FC1 = float(64)
        self.FC2 = float(80)        
        self.Pacc = float(0)
        self.Piacc = float(0)
        self.PreF1 = float(0)
        self.PreF1a = float(0)
        self.PreF1l = float(0)
        self.PisF1 = float(0)
        self.PisF1a = float(0)
        self.PisF1l = float(0)
        self.PisF1o = float(0)
        self.PreF2 = float(0)
        self.PisF2 = float(0)
        self.PreF2a = float(0)
        self.PreF2l = float(0)
        self.PisF2a = float(0)
        self.PisF2l = float(0)
        self.PisF2o = float(0)
        self.IF = float(0)
        self.UF = float(0)
        self.IFT = float(0)
        self.UFT = float(0)
        self.Error=float(0)
        self.Pi = float(0)
        self.gap = float(0)
        self.b = 1
        self.c = 0
        self.d = 0       
        self.vecMoisture1 = []
        self.vecMoisture2 = []
        self.vecPreF1 = []
        self.vecPisF1 = []
        self.vecPreF2 = []
        self.vecPisF2 = []
        self.vecRunoffg = []
        self.vecIsotopeg = []
        self.vecRunoffi = []
        self.vecIsotopei = []
        self.vecconcs1m = []
        self.vecconcs1im = []
        self.vecconcs1t = []
        self.vecconcs2m = []
        self.vecconcs2im = []
        self.vecconcs2t = [] 
        self.vecError = [] 
        self.maxQTimeStep = -1
        # initial precipitation amount 
        self.P0 = np.array([4.7, 14.9, 3.3, 0.4, 0.1, 0, 0,
                            0, 0, 0, 0, 0], dtype=float)
        # initial precipitation isotope, 0 mm rain use the last value  
        self.Pi0 = np.array([-11.22, -13.1, -13.34, -12.81, -10.35, -10.35, -10.35, -10.35, -10.35, -10.35, -10.35, -10.35], dtype=float)
        # define parameters
    def dynamic(self):
        x_a1, x_e1, x_a2, x_e2, x_f1, x_f2, x_f3, x_f4, x_b1, x_b2, x_c1, x_c2,x_c3= self.params_x
        # add rainwater to soil storage and accumulated rainfall
        self.P = self.P0[self.currentTimeStep() - 1]
        self.Pacc = self.Pacc+self.P
        # calculation of storage which can not detect by soil water probe 
        if self.P>0:
            self.S=self.S+x_c3*self.P
            self.P1=(1-x_c3)*self.P
        else:
            self.P1=self.P
            self.S=self.S          
        self.SW1 = self.SW1+self.P1
        # calculate the accumulation values of isotope in rainfall
        self.d = self.currentTimeStep()
        self.Pi = self.Pi0[self.currentTimeStep() - 1]
        self.Piacc = self.Piacc+self.P*self.Pi  
        # time step
        self.b = 1.0
        self.d=self.d+1
        # time step for effective rainfall
        if self.P > 0.2:
            self.c = float(0)
        else:
            self.c = self.c+self.b 
        # simplified infiltration rate
        infiltration =float(10)*exp(-self.c)+float(7)
        
        self.OF = max(self.P-infiltration, 0.0)  # overland flow
        self.SW1 = self.SW1-self.OF# remove quick flow from water entering the soil        
        self.P = max(self.P-self.OF, 0.0)# rainfall except overland flow       
        self.SW1 = max(self.SW1, 0.0) #make sure the value is not negative
        # calculation of preferential flow in shallow soil      
        if self.P < 0.3:
            self.maxQTimeStep = self.currentTimeStep() - 2
        if self.P < 0.3:
            preF1_ = self.vecPreF1[self.maxQTimeStep]  
            self.PreF1 = preF1_*exp(-self.c/x_b1)
        else:
            self.PreF1 = x_a1*(self.Pacc)            
        
        self.PreF1 = max(self.PreF1, 0.0)  #make sure the value is not negative
        self.PreF1 = min(self.PreF1, infiltration)#make sure the value did not exceed infiltration
        # subtract preferential flow from shallow soil water         
        if self.SW1 > self.inisw1:
            wu_slow = (self.SW1-self.inisw1)*x_e1
        else:
            wu_slow =0       
        # calculation matric flow (or piston flow) in shallow soil
        if self.P < 0.3:
            pisF1_ = self.vecPisF1[self.maxQTimeStep]  #
            self.PisF1 = pisF1_*exp(-self.c/x_b2)
        else:
            self.PisF1 = wu_slow
        
        if self.P < 0.3:
            pisF1_ = self.vecPisF1[self.maxQTimeStep]  #
            self.PisF1 = pisF1_*exp(-self.c/x_b2)
        else:
            self.PisF1 = wu_slow        
        if isnan(self.PisF1):
            self.PisF1 = 0.0
        self.PisF1 = max(self.PisF1, 0.0) #make sure the value is not negative
        self.PisF1 = min(self.PisF1, infiltration)
        #  calculation of preferential flow in deep soil    
        if self.P < 0.3:
            preF2_ = self.vecPreF2[self.maxQTimeStep]  #
            self.PreF2 = preF2_*exp(-self.c/x_b1)
        else:
            self.PreF2 = x_a2*self.PreF1

        if isnan(self.PreF2):
            self.PreF2 = 0.0
        self.PreF2 = max(self.PreF2, 0.0)  #make sure the value is not negative
        if self.P>0:
            wd_slow= (self.SW2-self.inisw2)*x_e2
        else:
            wd_slow =0
            
        # calculation of piston flow in deep soil 
        if self.P < 0.3:
            pisF2_ = self.vecPisF2[self.maxQTimeStep]  
            self.PisF2 = pisF2_*exp(-self.c/x_b2)
        else:
            self.PisF2 = wd_slow
        if isnan(self.PisF2):
            self.PisF2 = 0.0
        self.PisF2 = max(self.PisF2, 0.0)
        self.PisF2o=self.PisF2 # Pisflow lagged one time step
               
        
        # calculation interflow
        self.IF =x_f1*(self.PreF1)+x_f2*(self.PreF2+self.PisF2o)
        # calculation underflow
        self.UF =(1-x_f2)*(self.PreF2+self.PisF2o)
       
        # calculation accumulative flow in respective flow path
        self.PreF1a=self.PreF1a+self.PreF1
        self.PisF1a=self.PisF1a+self.PisF1
        self.PreF2a=self.PreF2a+self.PreF2
        self.PisF2a=self.PisF2a+self.PisF2      
        #soil water calculation
        if self.P>0:
            self.SW2 = self.SW2+self.PisF1+(1-x_f1)*self.PreF1 
            self.PreF1l=self.PreF1l
            self.PisF1l=self.PisF1l
        else:
            self.SW2 = self.SW2
            self.PreF1l=self.PreF1l+self.PreF1
            self.PisF1l=self.PisF1l+self.PisF1
        if self.P==0 and self.SW1<=self.FC1:
            self.SW1=self.SW1
            self.PreF1l=self.PreF1l+self.PreF1
            self.PisF1l=self.PreF1l+self.PisF1
        else:     
            self.PreF1l=self.PreF1l
            self.PisF1l=self.PisF1l 
            self.SW1 -= self.PisF1
            self.SW1 -= self.PreF1  
        
        self.SW2 -= self.PreF2
        self.SW2 -= self.PisF2  
                
        self.Error=self.S-self.PreF1l-self.PisF1l
        
        ###### shallow soil water isotope ##
        self.concs1m =(self.Pacc)/(x_f3*(self.inisw1-self.PreF1a-self.PisF1a)+self.P+self.Pacc)*(self.Piacc/self.Pacc)+x_f3*(self.inisw1-self.PreF1a-self.PisF1a)/(x_f3*(self.inisw1-self.PreF1a-self.PisF1a)+self.P+self.Pacc)*self.concs1mo+(self.P)/(x_f3*(self.inisw1-self.PreF1a-self.PisF1a)+self.P+self.Pacc)*(self.Pi)
        
        self.concs1im = ((1-x_f3)*(self.inisw1-self.PisF1a-self.PreF1a))*self.concs1imo/((1-x_f3)*(self.inisw1-self.PisF1a-self.PreF1a)+self.Pacc) +self.Piacc/((1-x_f3)*(self.inisw1-self.PisF1a-self.PreF1a)+self.Pacc)
        self.concs1t=x_f3*self.concs1m+(1-x_f3)*self.concs1im
        ###### deep soil water isotope ##
        self.concs2m=(x_f4*self.inisw2*self.concs2mo+(self.PreF1a+self.Pacc)*(self.Piacc/self.Pacc)+self.P*self.Pi)/(x_f4*self.inisw2+self.PreF1a+self.Pacc+self.P)  
#(x_a2*self.inisw2*self.concs2mo+(self.PreF1a)*self.Piacc/self.Pacc+self.P*self.Piacc/self.Pacc

#)/(x_a2*self.inisw2+self.PreF1a+self.P)        
        self.concs2im=((self.PisF1a-self.PisF2a)*self.concs1imo+self.Piacc+(1-x_f4)*self.inisw2*self.concs2imo)/(self.PisF1a-self.PisF2a+(1-x_f4)*self.inisw2+self.Pacc)   
        self.concs2t=(1-x_f4)*self.concs2im+x_f4*self.concs2m        
                       
        ######################################### underflow water isotope ##
        self.concg =self.conci =(x_c1*self.P*self.Pi+x_c2*(self.UF)*self.Piacc/self.Pacc+x_a1*self.inisw1*self.concs1mo+x_a2*self.inisw2*self.concs2mo)/(x_c2*(self.UF)+x_a1*self.inisw1+x_a2*self.inisw2+x_c1*self.P)
                
        #### data outupt ##

        self.vecPreF1.append(float(self.PreF1))
        self.vecPisF1.append(float(self.PisF1))

        self.vecPreF2.append(float(self.PreF2))
        self.vecPisF2.append(float(self.PisF2))

        self.vecMoisture1.append(float(self.SW1/200))
        self.vecMoisture2.append(float(self.SW2/200))

        self.vecRunoffg.append(float(self.UF))
        self.vecIsotopeg.append(float(self.concg))
        
        self.vecRunoffi.append(float(self.IF))
        self.vecIsotopei.append(float(self.conci))
        
        self.vecconcs1m.append(float(self.concs1m))
        self.vecconcs1im.append(float(self.concs1im))
        self.vecconcs1t.append(float(self.concs1t))
        self.vecconcs2m.append(float(self.concs2m))
        self.vecconcs2im.append(float(self.concs2im))
        self.vecconcs2t.append(float(self.concs2t))
        self.vecError.append(float(self.Error))
def eval(params):
    nrOfTimeSteps = 12
    params_x = params

    myModel = MyFirstModel(params_x)
    dynamicModel = DynamicFramework1(myModel, nrOfTimeSteps)
    dynamicModel.run()
    return [np.array(myModel.vecRunoffg),
            np.array(myModel.vecIsotopeg),
            np.array(myModel.vecRunoffi),
            np.array(myModel.vecIsotopei),            
            np.array(myModel.vecMoisture1),
            np.array(myModel.vecMoisture2),
            np.array(myModel.vecPreF1),
            np.array(myModel.vecPisF1),
            np.array(myModel.vecPreF2),
            np.array(myModel.vecPisF2),
            np.array(myModel.vecconcs1m),
            np.array(myModel.vecconcs1im),
            np.array(myModel.vecconcs1t),
            np.array(myModel.vecconcs2m),
            np.array(myModel.vecconcs2im),
            np.array(myModel.vecconcs2t),
            np.array(myModel.vecError),
            ]


def eval0(params):
    nrOfTimeSteps = 12
    
    params_x = params  

    myModel = MyFirstModel(params_x)
    dynamicModel = DynamicFramework1(myModel, nrOfTimeSteps)
    dynamicModel.run()
    
    print('===============')
    print(np.array(myModel.vecRunoffg))
    print('--------------')
    print(np.array(myModel.vecIsotopeg))
    print('--------------')
    
    R_ = [np.array(myModel.vecRunoffg),
            np.array(myModel.vecIsotopeg),
            np.array(myModel.vecRunoffi),
            np.array(myModel.vecIsotopei),           
            np.array(myModel.vecMoisture1),
            np.array(myModel.vecMoisture2),
            np.array(myModel.vecPreF1),
            np.array(myModel.vecPisF1),
            np.array(myModel.vecPreF2),
            np.array(myModel.vecPisF2),
            np.array(myModel.vecconcs1m),
            np.array(myModel.vecconcs1im),
            np.array(myModel.vecconcs1t),
            np.array(myModel.vecconcs2m),
            np.array(myModel.vecconcs2im),
            np.array(myModel.vecconcs2t),
            np.array(myModel.vecError),
            ]

    return R_
def nse(simulation_s, evaluation):
    nse_ = 1 - (np.sum((evaluation - simulation_s) ** 2, dtype=np.float64) /
                np.sum((evaluation - np.mean(evaluation)) ** 2, dtype=np.float64))

    return nse_
    # initial values of parameters
if __name__ == "__main__":
    eval0([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 5.0, 5.0, 0.1, 0.1,0.1])
