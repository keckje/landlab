# -*- coding: utf-8 -*-
"""
Created on Tue May 25 20:00:58 2021

@author: Jeff Keck
        Jeff.Keck@dnr.wa.gov
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


class QDepthFromP:
    '''
    determine the flow hydraulics of a channel for a specific RI  flow 
    given its slope, contributing area, mean annual precipitation and 
    hydraulilc geometry relationships for:
        the return interval flow (Q_ri)
        bankfull flow (Q_bf)
        bankfull width (Q_bf)
        bankfull roughness (n_bf)
        return interval flow roughness (n_ri_exp)


        
    Uses the compound channel method (Open Channel Hydraulics, Sturm 2010) and 
    provided bankfull width, slope and depth to determine the normal depth 
    (uniform, steady) of a given flow return interval rate 
    
    Channel wall slopes are assumed 1:1. User must also provide parameters
    for determing channel roughness and flood plain width as a function of 
    contributing area to the channel.
    '''
    
    
    def __init__(
        self,
        # these inputs changed to dictionary inputs
        
        # P_para = [daily P lognormal distribution parameters, list]
        # MAP = float, mean annual precip of basin
        # Q_ri_para =  {USGS parameters for 5 ri flows, dictionary of lists 2y:   ,5y:  ,10y:   }
        # bf_geo {USGS parameters for bankfull flow geometry, width:  , roughness:   }
        Q_ri_hg = [68.549, 0.814, 0.0151, 0.0132, 98], # Mastin et al., 2016
        Q_bf_hg = [50.93, 0.67], # Castro and Jackson, 2001
        w_bf_hg = [11.80, 0.38], # Castro and Jackson, 2001
        n_bf_hg = [0.12, -0.21], # Field based estimate for the Sauk watershed from SUGS gages
        n_q_exp = -0.286, # Feild based estimate for the Sauk watershed from USGS gages
        btf_factor = 0.5, #bankfull width is estimated as this factor of channel width
        Imperial = True): # 

        
        self.Q_ri_a = Q_ri_hg[0]
        self.Q_ri_b = Q_ri_hg[1]
        self.Q_ri_c = Q_ri_hg[2]
        self.Q_ri_d = Q_ri_hg[3]
        self.Q_ri_CAN = Q_ri_hg[4]
        self.Q_bf_a = Q_bf_hg[0]
        self.Q_bf_b = Q_bf_hg[1]
        self.w_bf_a = w_bf_hg[0]  
        self.w_bf_b = w_bf_hg[1]
        self.n_bf_a = n_bf_hg[0]
        self.n_bf_b = n_bf_hg[1]
        self.n_q_exp = n_q_exp
        self.btf_factor = btf_factor
        self.Imperial = Imperial
                
        # self._Construct_P_distribution()
        # self._construct_P_to_Q_function()
                 

    def __call__(self, S, CA): # slope [m/m], CA [km2], MAP [mm/year]
            
            # iterates over each link, updates the nmg depth field
            
            self.S = S
            self.CA = CA
            self.MAP = MAP
    
            # self._sample P
            
            # self._P_to_Q
            
            self._DetermineHydraulics()
    
    
    
    
    # def _Construct_P_distribution:
        
    # def _sample P
    
    # def _construct_P_to_Q_function
    
    # def _P_to_Q
    
    
            
    def _DetermineHydraulics(self):
        
        # from area get bankfull roughness, Q, bf width
        self.n_bf = self._channel_n_bf(self.CA)
        
        # get bankfull flow, width and depth and return interval flow
        self.Q_bf = 1# self.Q_2  #self.channel_Q_bf(self.CA)

        self.w_bf = self.channel_w_bf(self.CA)

        # from bankfull flow, width and roughtness and the channel slope, determine
        # flow depth and the base of the channel bed width
        self.d_bf,self.A_bf,self.P_bf, self.b = Depth_LowFlow_bws(self.Q_bf,self.S,self.w_bf,self.n_bf)    

        # determine Qri from CA and MAP and USGS coefficients
        self.Q_ri # defined by self._P_to_Q  #= self.channel_Q_ri(self.CA)
            
        # define compound channel geometry
        nri = self.channel_n_ri(self.Q_ri)
        y1 = self.d_bf; y2 = y1
        b1 = self.b*self.btf_factor; b2 = b1
        m1 = 1; m2 = 1; sm1 =1; sm2 = 1
        n1 = 0.15; n2 = 0.15
        self.CompoundChannel = [self.b,m1,m2,nri,y1,b1,sm1,n1,y2,b2,sm2,n2]
        
        # compute hydraulic conditions of flow rate Qri       
        self.HydCond = ComputeCompoundChannelDepth_F2(self.Q_ri, self.S, self.CompoundChannel)

    def _channel_n_bf(self,A):
        '''
        flow resistance (n) at bankfull flow
        '''
        n_bf = self.n_bf_a*A**(self.n_bf_b)
        
        return n_bf

    def channel_Q_bf(self,A):
        if self.Imperial:
            Qbf = ((0.3861**self.Q_bf_b)*self.Q_bf_a*A**self.Q_bf_b)/(3.281**3)
        else:
            Qbf = self.Q_bf_a*A**self.Q_bf_b
        return Qbf
    
    def channel_w_bf(self, A):
        if self.Imperial:
            wbf = ((0.3861**self.w_bf_b)*self.w_bf_a*A**self.w_bf_b)/3.281
        else:
            wbf = self.w_bf_a*A**self.w_bf_b
        return wbf 
       
    def channel_Q_ri(self, A):
        if self.Imperial:
            Qri = ((self.Q_ri_a*(0.386**self.Q_ri_b)*(A)**self.Q_ri_b)*(10**(1/25.4))**(self.Q_ri_c*self.MAP)/(10**(self.Q_ri_d*self.Q_ri_CAN)))/(3.281**3)
        else:
            Qri = ((self.Q_ri_a*A**self.Q_ri_b)*10**(self.Q_ri_c*self.MAP))/(10**(self.Q_ri_d*self.Q_ri_CAN))
        return Qri 
    

    def channel_n_ri(self, Qri):
        '''
        flow resistance (n) at flow rate Q
        '''
        n_q = self.n_bf*(Qri/self.Q_bf)**(self.n_q_exp) 
        return n_q




def A_t(b,m1,m2,y):
    '''
    Area of trapezoid, use for all trapezoids in compound channel
    '''
    A = (y/2)*(b+b+y*(m1+m2))
    return A

def P_t(b,m1,m2,y):
    '''
    Wetted perimeter of trapezoid, below flood plains
    '''
    P = b+y*((1+m1**2)**(1/2)+(1+m2**2)**(1/2))
    return P

def T_t(b,m1,m2,y):
    '''
    Surface width of trapezoid
    '''
    T = b+y*(m1+m2)
    return T


def P_fp(b,fpm,y):
    '''
    Wetted perimeter of trapezoid,  flood plain
    vertical wetted perimeter with main channel excluded
    '''
    P = b+y*((1+fpm**2)**(1/2))  
    return P


def P_mc_af(m1,m2,y):
    '''
    Wetted perimeter of trapezoid in main channel when flow above flood plain
    base wetted perimeter excluded, channel bank and vertical wetted perimiter 
    between flood plain included
    '''
    P = y*((1+m1**2)**(1/2)+(1+m2**2)**(1/2))
    return P

def A_mc_f1(b,m1,m2,y,y1):
    '''
    Area, main channel, flow above flood plain 1
    '''
    yd1 = y-y1
    b1 = b+y1*(m1+m2)
    A1 = A_t(b,m1,m2,y1)
    A2 = A_t(b1,m2,0,yd1) #use slope of main channel trapezoid opposite flood plain 1
    A = A1+A2
    return A

def A_mc_f2(b,m1,m2,y,y1,y2):
    '''
    Area, main channel, flow above flood plain 2
    '''
    yd1 = y2-y1
    yd2 = y-y2
    b1 = b+y1*(m1+m2)
    b2 = b1+yd1*(m2+0)
    A1 = A_t(b,m1,m2,y1)
    A2 = A_t(b1,m2,0,yd1)
    A3 = A_t(b2,0,0,yd2)
    A = A1+A2+A3
    return A

def P_mc_f1(b,m1,m2,y,y1):
    '''
    Perimeter, main channel, flow above flood plain 1
    '''
    yd1 = y-y1
    P1 = P_t(b,m1,m2,y1)
    P2 = P_mc_af(m2,0,yd1)
    P = P1+P2
    return P

def P_mc_f2(b,m1,m2,y,y1,y2):
    '''
    Perimeter, main channel, flow above flood plain 2
    '''
    yd1 = y2-y1
    yd2 = y-y2
    P1 = P_t(b,m1,m2,y1)
    P2 = P_mc_af(m2,0,yd1)
    P3 = P_mc_af(0,0,yd2)
    P = P1+P2+P3
    return P


def Depth_LowFlow(Q,S,b,m1,m2,n):
    
    '''
    A main channel = A_t(b,m1,m2,y)
    P main channel = P_t(b,m1,m2,y)
    '''
    def h(y):
        return np.abs(Q-(1/n)*A_t(b,m1,m2,y)*((A_t(b,m1,m2,y)/P_t(b,m1,m2,y))**(2/3))*S**(1/2))

    r = scipy.optimize.minimize_scalar(h,method='bounded',bounds = [.0001,10])
    
    y = r.x
    A = A_t(b,m1,m2,y) #flow area
    P = P_t(b,m1,m2,y) #wetted perimeter
    T = T_t(b,m1,m2,y)
    return (y,A,P,T)


def Depth_FlowOnFloodPlain1(Q,S,b,m1,m2,n,y1,b1,sm1,n1):
    '''
    A main channel = A_mc_f1(b,m1,m2,y,y1)
    P main channel = P_mc_f1(b,m1,m2,y,y1)
    
    A flood plain 1 = A_t(b1,m1,0,y-y1)
    P flood plain 1 = P_fp(b1,m1,y-y1)                
    '''
    
    def i(y):
        return np.abs(Q-((1/n)*A_mc_f1(b,m1,m2,y,y1)*((A_mc_f1(b,m1,m2,y,y1)/P_mc_f1(b,m1,m2,y,y1))**(2/3))*S**(1/2)
                      +(1/n1)*A_t(b1,m1,0,y-y1)*((A_t(b1,m1,0,y-y1)/P_fp(b1,m1,y-y1))**(2/3))*S**(1/2)))
        

    r = scipy.optimize.minimize_scalar(i,method='bounded',bounds = [y1,y1+10]) #method bounded few problems finding minimum, give upper limit to reduce search range
    
    y = r.x
    Amc = A_mc_f1(b,m1,m2,y,y1)
    Pmc = P_mc_f1(b,m1,m2,y,y1)
    Afp1 = A_t(b1,m1,0,y-y1)
    Pfp1 = P_fp(b1,m1,y-y1)
    A = Amc+Afp1 #checked with manual compuation
    P = Pmc+Pfp1-(y-y1) #checked with manual compuation
    T = T_t(T_t(b,m1,m2,y1),0,m2,y-y1)+T_t(b1,0,sm1,y-y1)#checked with manual compuation    
    Qmc = (1/n)*Amc*((Amc/Pmc)**(2/3))*S**(1/2)
    Vmc = Qmc/Amc
    Qfp1 = (1/n)*Afp1*((Afp1/Pfp1)**(2/3))*S**(1/2)
    Vfp1 = Qfp1/Afp1        
    return (y,A,P,T,Qmc,Vmc,Qfp1,Vfp1)


def Depth_FlowOnFloodPlain2(Q,S,b,m1,m2,n,y1,b1,sm1,n1,y2,b2,sm2,n2):
    '''
    A main channel = A_mc_f2(b,m1,m2,y,y1,y2)
    P main channel = P_mc_f2(b,m1,m2,y,y1,y2)
    
    A flood plain 1 = A_t(b1,m1,0,y2-y1)
    P flood plain 1 = P_fp(b1,m1,y2-y1)
          
    A flood plain 2 = A_t(b2,m2,0,y-y2)
    P flood plain 2 = P_fp(b2,m2,y-y2)
    '''       
    
    def j(y):
        return np.abs(Q-((1/n)*A_mc_f2(b,m1,m2,y,y1,y2)*((A_mc_f2(b,m1,m2,y,y1,y2)/P_mc_f2(b,m1,m2,y,y1,y2))**(2/3))*S**(1/2)
                      +(1/n1)*A_t(b1,m1,0,y2-y1)*((A_t(b1,m1,0,y2-y1)/P_fp(b1,m1,y2-y1))**(2/3))*S**(1/2)
                      +(1/n2)*A_t(b2,m2,0,y-y2)*((A_t(b2,m2,0,y-y2)/P_fp(b2,m2,y-y2))**(2/3))*S**(1/2)))
        

    r = scipy.optimize.minimize_scalar(j,method='bounded',bounds = [y2,y2+20])
    
    y = r.x #depth in main channel
    Amc = A_mc_f2(b,m1,m2,y,y1,y2) #area main channel
    Pmc = P_mc_f2(b,m1,m2,y,y1,y2) #wetter perimiter main channel - includes boundary between main channel and flood plains
    Afp1 = A_t(b1,m1,0,y-y1) #flood plain 1
    Pfp1 = P_fp(b1,m1,y-y1)
    Afp2 = A_t(b2,m2,0,y-y2) #flood plain 2
    Pfp2 = P_fp(b2,m2,y-y2)
    A = Amc+Afp1+Afp2 #checked with manual compuation
    P = Pmc+Pfp1+Pfp2-(y-y1)-(y-y2) #checked with manual compuation
    T = T_t(T_t(b,m1,m2,y1),0,m2,y2-y1)+T_t(b1,0,sm1,y-y1)+T_t(b2,0,sm2,y-y2) #checked with manual computation
    Qmc = (1/n)*Amc*((Amc/Pmc)**(2/3))*S**(1/2)
    Vmc = Qmc/Amc
    Qfp1 = (1/n)*Afp1*((Afp1/Pfp1)**(2/3))*S**(1/2)
    Vfp1 = Qfp1/Afp1
    Qfp2 = (1/n)*Afp2*((Afp2/Pfp2)**(2/3))*S**(1/2)
    Vfp2 = Qfp2/Afp2
    return (y,A,P,T,Qmc,Vmc,Qfp1,Vfp1,Qfp2,Vfp2)


def ComputeCompoundChannelDepth_F2(Q,S,CCGeometry):

    b,m1,m2,n,y1,b1,sm1,n1,y2,b2,sm2,n2 = CCGeometry
    
    out1 = Depth_LowFlow(Q,S,b,m1,m2,n)
    
    
    
    y = out1[0]
    
    if y <= y1:
    
        A = out1[1]
        P = out1[2]
        T = out1[3]
        Qmc = Q
        Vmc = Q/A
        Qfp1 = 0
        Vfp1 = 0
        Qfp2 = 0
        Vfp2 = 0
        c = 'low flow'
        
    elif y > y1:
        out2 = Depth_FlowOnFloodPlain1(Q,S,b,m1,m2,n,y1,b1,sm1,n1)
        y = out2[0]
    
        if y > y2:
            out3 = Depth_FlowOnFloodPlain2(Q,S,b,m1,m2,n,y1,b1,sm1,n1,y2,b2,sm2,n2)
            y = out3[0]
            A = out3[1]
            P = out3[2]
            T = out3[3]
            Qmc = out3[4]
            Vmc = out3[5]
            Qfp1 = out3[6]
            Vfp1 = out3[7]
            Qfp2 = out3[8]
            Vfp2 = out3[9]
            c = 'flood plain 2'

        else:
            A = out2[1]
            P = out2[2]
            T = out2[3]
            Qmc = out2[4]
            Vmc = out2[5]
            Qfp1 = out2[6]
            Vfp1 = out2[7]
            Qfp2 = 0
            Vfp2 = 0
            c = 'flood plain 1'



    R = A/P #hydraulic radius
    V = Q/A #flow velocity
    D = A/T # hydraulic depth (average depth across cross-section)
    Fn = V/((9.81*D)**(0.5)) #froude number, ratio of momentum to gravity ~ velcity/wave speed => needs to be changed to overbank flow froude number, p 43 of Sturm                                                                     
    
    RDict = {'flood extent':c,'normal depth [m]':y,'hydraulic radius [m]':R,
             'flow area [m2]':A,'flow velocity [m/s]':V,'Mean(hydraulic) Depth':A/T,'Froude #':Fn,
             'mc flow [m3/s]':Qmc,'mc flow velocity [m/s]':Vmc}

    return RDict



def A_t_bws(b_ws,m1,m2,y):
    '''
    Area of trapezoid, use for all trapezoids in compound channel, given 
    water surface width instead of channel bed width
    '''
    A = (y/2)*(b_ws+b_ws-y*(m1+m2))
    return A

def P_t_bws(b_ws,m1,m2,y):
    '''
    Wetted perimeter of trapezoid, below flood plains, given water surface width
    instead of channel bed width
    '''
    P = b_ws-y*(m1+m2)+y*((1+m1**2)**(1/2)+(1+m2**2)**(1/2))
    return P

def b_t_bws(b_ws,m1,m2,y):
    '''
    Wetted perimeter of trapezoid, below flood plains, given water surface width
    instead of channel bed width
    '''
    b = b_ws-y*(m1+m2)
    return b


def Depth_LowFlow_bws(Q,S,b_ws,n,m1=1,m2=1):
    
    '''
    A main channel, using water surface width = A_t_bws(b,m1,m2,y)
    P main channel, using water surface width = P_t_bws(b,m1,m2,y)
    '''
    def h(y):
        return np.abs(Q-(1/n)*A_t_bws(b_ws,m1,m2,y)*((A_t_bws(b_ws,m1,m2,y)/P_t_bws(b_ws,m1,m2,y))**(2/3))*S**(1/2))

    r = scipy.optimize.minimize_scalar(h,method='bounded',bounds = [.0001,b_ws/(m1*2)]) # y can not be higher than half the width of b_ws
    
    y = r.x
    A = A_t_bws(b_ws,m1,m2,y) #flow area
    P = P_t_bws(b_ws,m1,m2,y) #wetted perimeter
    b = b_t_bws(b_ws,m1,m2,y)
    return (y,A,P,b)



#interplate function used by other functions
def intp(x,y,x1): 
    
    f = interpolate.interp1d(x,y)   
    
    out = f(x1)
    return out



def RI_Events_DIST(AMS, dist = 'LP3', 
                   RI=[1.5,2,5,25,50,100], plotting_position = 'Weibull'):
    '''
    Fits distribution to annual maximum series or partial duration series using
    methods and formulas described in: 
   
         Maidment, 1992, Handbook of Hydrology, Chapter 18
       
    Parameters
    ----------
    AMS : pd series, index is pd.datetime
        annual maximum series (or partial duration series if partial
                               duration series is smame length as annual maximum
                               series)
    dist : string
        type of distribution fit to the annual maximum series. cane be either
        lognormal, gumbel, GEV, pearson type III or log-pearson type III:
        'LN', 'GUMB', 'GEV', 'P3', 'LP3' , default is 'LP3'
        NOTE: some distributions may not fit data
    
    RI : list
        flow magnitude is returned for all return interval values listed in RI 
        return interval values must be greater than 1
        The default is [1.5,2,5,25,50,100].
    
    plotting_position : string
        plotting position forumula. Default is 'Weibull'
    
    Returns
    -------
    Fx : np.array
        cdf quantile
    x1 : np.array
        cdf value
    T : np.array
        quantile equivalent return interval [yrs]
    Q_ri : dictionary
        key is each value in RI, value is magnitude


    '''
    # compute moments
    
    mn = MS.mean() # m1
    vr = MS.var() # m2
    cs = stats.skew(MS.values) 
    m2 = lm(MS.values,moment = 2)
    m3 = lm(MS.values,moment = 3)
    m4 = lm(MS.values,moment = 4)
    lskew = m3
    L3 = m3*m2 #third Lmoment
    lkurt = m4
    L4 = m4*m2 #fourth Lmoment    

    
    if dist == 'LN':
    
    #   parameters, maidment 1992, p18.14 
        sigy = (np.log(1+vr/(mn**2)))**0.5 #maidment 1992, p18.14 
        muy = np.log(mn)-0.5*sigy**2 #maidment 1992, p18.14 
        
        x1 = np.linspace(1,2*max(MS),10000)  #solve pdf at fine resolution
        
        #for comparison to emperical estimate using plotting position
        x2 = np.sort(MS, axis=0) # values in MS, sorted small to large (pp is quantile)
        
        #probability density function, maidment 1992, p18.12 
        X = [x1,x2]
        
        fx = {}
        for i,x in enumerate(X):

            fp = 1/(x*(2*3.1416*sigy**2)**0.5) 
            sp = np.exp(-1/2*((np.log(x)-muy)/(sigy))**2)
            fx[i] = fp*sp
    
    if dist =='GUMB':
        # parameters
        alhat =m[1]/np.log(2) # 18.2.15
        xi = mn-0.5772*alhat # 18.2.17
        x1 = np.linspace(1,2*max(MS),10000)
        
        x2 = np.sort(MS, axis=0)
        X = [x1,x2]
        
        fx = {}
        
        for i,x in enumerate(X):
            
            fp =-(x-xi)/alhat
            sp = np.exp(-(x-xi)/alhat)
            fx[i] = (1/alhat)*np.exp(fp-sp)
            
    if dist == 'GEV':
    
        c =(2*m[1])/((L3+3*m[1]))-(np.log(2)/np.log(3))
        k = 7.8590*c+2.9554*c**2
        
        g = gamma(1+k); 
        alphap = (k*m[1])/(g*(1-(2**-k))) 
        xi = m[0]+(alphap*(g-1))/(k)
        
        x1 = np.linspace(1,2*max(MS),10000)
        
        x2 = np.sort(MS, axis=0) 

        Fx = np.exp(-(1-(k*(x1-xi))/alphap)**(1/k))
        
     
    if dist == 'P3':    
        #parameters
        alphap = 4/cs**2#18.20
        betap = 2/((vr**0.5)*cs)
        xi = mn-alphap/betap
        
        
        x1 = np.linspace(np.ceil(xi),1.5*max(MS),10000) 

        x2 = np.sort(MS, axis=0) 
        
        X = [x1,x2]
        
        fx = {}
        for i,x in enumerate(X):
            
            fp = np.absolute(betap)*(betap*(x-xi))**(alphap-1)
            sp = np.exp(-betap*(x-xi))/((gamma(alphap)))
            fx[i] = fp*sp
    
    
    if dist == 'LP3':
        # natural log of flow data is pearson type 3 distributed
        x = np.log(MS.values)
        
        mn = x.mean() #compute first three moments of log(data)
        vr = x.var()
        cs = stats.skew(x)
        
        #parameters
        alphap = 4/cs**2#18.20
        betap = 2/((vr**0.5)*cs)
        xi = mn-alphap/betap
        
        E3 = np.exp(3*xi)*(betap/(betap-3))**alphap #18.33
        E2 = np.exp(2*xi)*(betap/(betap-2))**alphap
        muq = np.exp(1*xi)*(betap/(betap-1))**alphap
        vrq= np.exp(2*xi)*(((betap/(betap-2))**alphap)-(betap/(betap-1))**(2*alphap))
        csq = (E3-3*muq*E2+2*muq**3)/((vrq**.5)**3)
        
        alphapq = 4/csq**2
        alphapq = alphapq
        
        betapq = 2/((vrq**.5)*csq)
        betapq=betapq
        
        xiq = muq-alphapq/betapq
        xiq=xiq
        
        if betap <0:
            if 1.5*max(MS) < np.exp(xi):
                x1=np.linspace(1,1.5*max(MS),num = 10000)
            else:
                x1=np.linspace(1,np.exp(xi),num = 10000) #table 18.2.1
        if betap >0:
            x1=np.linspace(np.exp(xi),max(MS)*1.5,num = 10000) 
         
       
        #for comparison to emperical estimate using plotting position
        x2 = np.sort(MS, axis=0) # values in MS, sorted small to large (pp is quantile)
        
        X = [x1,x2]
        
        fx = {}
        for i,x in enumerate(X):
            
            fp = (1/x)*np.absolute(betap)*(betap*(np.log(x)-xi))**(alphap-1)
            sp = np.exp(-betap*(np.log(x)-xi))/((gamma(alphap)))
            fx[i] = fp*sp
  
        
    
    #histogram of data
    if dist != 'GEV':
        fig, ax=plt.subplots(1,1,figsize=(6,6))
        plt.hist((MS), bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.xlim(0,1.5*max(MS))
        plt.show()
        
        #parameterized pdf
        fig, ax=plt.subplots(1,1,figsize=(6,6))
        plt.plot(x1, fx[0],'b-',markersize=8,linewidth=3)
        plt.tick_params(labelsize=16)
        plt.grid(which='both')
        #plt.title(title,fontsize=18)
        plt.show()
        
        
    #cdf created by summing area under pdf
    fig, ax=plt.subplots(1,1,figsize=(6,6))
    Fx = sc.integrate.cumtrapz(fx[0], x1, initial=0)
    plt.plot(x1,Fx) #examine cdf
    plt.show()
    

    #summary plot
    n = MS.shape[0]
    ranks = np.array(range(1,n+1))
    #pp = (ranks-.4)/(n+0.2) #Cunnane
    # plottting position (exceedance probability if ordered large to small)
    if plotting_position == 'Cunnane':
        PP = (ranks-.4)/(n+.2) #compute plotting position 
    elif plotting_position == 'Weibull':
        PP = ranks/(n+1)
    elif plotting_position == 'Blom':
        PP = (ranks - (3/8))/(n+(1/4))
    else:
        PP = ranks/(n+1)
    
    T = 1/(1-Fx)
    
    x = T
    y = x1
    
    f = interpolate.interp1d(x,y) 
    
    ri = RI
    Q_ri = {}
    
    for i in ri:
        Q_ri[i] = f(i)
    
    
    fig, ax=plt.subplots(1,1,figsize=(12,6))
    for i,v in enumerate(ri):
        
       plt.plot([-.5,1.5],[Q[v],Q[v]],'k--',linewidth = 1+5/len(ri)*i,alpha = 1-1/len(ri)*i, label = str(ri[i]))
    
    plt.plot(Fx,x1,'r',label = 'fitted distribution');
    plt.plot(pp,x2,'.', label ='emperical estimate');
    plt.ylim([0,max([Q[v],x2.max()])*1.05])
    plt.xlim([-0.02,1.02])
    ax.legend(loc = 'upper center')
    plt.show()

    # interpolated flow rates for RI, flow rate for each quantile, return period of each quantile, 
    # cdf quantile, cdf flow rate, return interval of quantile, flow rates of return interval flows listed in RI
    return Fx, x1, T, Q_ri 


def AnnualMaximumSeries(time_series, plotting_position = 'Cunnane'):
    '''
    Parameters
    ----------
    time_series : pd series
        time series of data from which annual maximum series will be computed
        index is pd.DateTime
    
    plotting_position : string
        plotting position forumula. Default is 'Weibull'

    Returns
    -------
    AMS : pandas dataframe
        a dataframe of the annual maximum series magnitude and return interval [yrs]

    '''
    
    #sep=30 #separation between storm events to be considered independent - flow
    #sep =7 #precipitation
    
    ranks = {}
    RIp = {}
    MagpS ={}
    pds_l = {}
    Qri_l = {}
    
    time_series.name = 'value'
    time_series_a = time_series.resample('1Y').max() # resample to annual maximum value
    AMS = time_series_a.sort_values(ascending=False).to_frame() #sort large to small
    
    n = AMS.shape[0]
    ranks = np.array(range(1,n+1))
    
    # plottting position (exceedance probability if ordered large to small)
    if plotting_position == 'Cunnane':
        PP = (ranks-.4)/(n+.2) #compute plotting position 
    elif plotting_position == 'Weibull':
        PP = ranks/(n+1)
    elif plotting_position == 'Blom':
        PP = (ranks - (3/8))/(n+(1/4))
    else:
        PP = ranks/(n+1)
      
    EP = PP 

    T = 1/PP # return interval
    
    AMS['RI [yrs]']=T
    
    return AMS


def PartialDurationSeries(time_series, plotting_position = 'Cunnane',sep=30):
    '''
    Uses a peaks-over-threshold approach to create a partial duration series
    from a time series of data. The PDS is truncated to include only RI>=1 events
    
    Based on methods described in:
            Malamud B.D. and Turcotte D.L., 2006, The Applicability of power-law 
            frequency statstics to floods, Journal of Hydrology, v.322, p.168-180

    Parameters
    ----------
    time_series : pd series
        time series of data from which annual maximum series will be computed
        index is pd.DateTime
    
    plotting_position : string
        plotting position forumula. Default is 'Weibull'
        
    sep = integer
        time separation [days] used to extract independent events
    
    Returns
    -------
    PDS : pandas dataframe
        a dataframe of the partial duration series magnitude and return interval [yrs]

    '''
    
    #sep=30 #separation between storm events to be considered independent - flow
    #sep =7 #precipitation
    
    ranks = {}
    RIp = {}
    MagpS ={}
    pds_l = {}
    Qri_l = {}


    Qdt = time_series.copy() # make copy so as not to write over original

    
    ll = Qdt.shape[0]
    c=1       
    pds = {}
    mq = Qdt.max()
    pds[Qdt[Qdt==mq].index[0]] = mq
    
    while  mq>0:
    
        Rf = Qdt[Qdt==mq].index+ datetime.timedelta(days=sep)
        Rp = Qdt[Qdt==mq].index- datetime.timedelta(days=sep)
        mask = (Qdt.index > Rp[0]) & (Qdt.index <= Rf[0])
        Qdt.loc[mask]=0
    
        mq = Qdt.max() 

        if mq >0: # last value of algorith is 0, dont add zero to pds
            pds[Qdt[Qdt==mq].index[0]] = mq

    pds_l = pds
    
    PDS = pd.DataFrame.from_dict(pds_l, orient='index')
    PDS.columns = ['value']
        
    Yrs = max(Qdt.index.year)-min(Qdt.index.year)+1 # number of years in time series
    PDS = PDS.iloc[0:Yrs] # truncate to have length equal to Yrs
    n = PDS.shape[0]            
    ranks = np.array(range(1,n+1)) 
    #compute plotting position or exceedance probability (probablity of equal to or larger) if largest to smallest. NOTE: quantile if smallest to largest
    # plottting position (exceedance probability if ordered large to small)
    if plotting_position == 'Cunnane':
        PP = (ranks-.4)/(Yrs+.2) #compute plotting position 
    elif plotting_position == 'Weibull':
        PP = ranks/(Yrs+1)
    elif plotting_position == 'Blom':
        PP = (ranks - (3/8))/(Yrs+(1/4))
    else:
        PP = ranks/(n+1)
    
    #exceedance probability, eqaul to plotting position if ordered large to small
    EP = PP
    T = 1/PP                        
    PDS['RI [yrs]'] = T                    
    
    return PDS


    

#model results for all grid points, PNNL, horly
# os.chdir('D:/UW_PhD/PreeventsProject/Hydrometeorology and Floods/FlowModeling/ModeltoObsComparison2002to2015')
# m1 = pd.read_csv('Streamflow.Only_AllPtsPNNLWRF2002to2015', delim_whitespace=True)#, delim_whitespace=True) m3/h
# m1['DATE'] = pd.to_datetime(m1['DATE'])
# m1 = m1.set_index('DATE')
# Qm1 = (m1['12189500'])/3600 #convert m3/h to m3/s
# Qm1[0] = 200 #assume value of first day in model output
# Qm1_d = Qm1.to_frame()

# time_series = Qm1_d['12189500']
# # ams = AnnualMaximumSeries(time_series)
# ams = PartialDurationSeries(time_series)
# RI_Events_DIST(ams['value'],dist = 'LP3',RI=[1.051,2,5,25,50,100])


# bS = {'flow':[Qm1_d['12189500']]}
# dnm = ['flow']
# out = PartialDurationSeries(dnm,bS,Stype='AMS',RI=[.2,2,5,10,13],events = False,sep=30)
# MS = out[0]['flow']['12189500']
# RI_Events_DIST(MS,dist = 'LP3',RI=[1.051,2,5,25,50,100])


