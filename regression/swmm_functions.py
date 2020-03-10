import pandas as pd;
from pyswmm import Simulation;
from pyswmm import Subcatchments;
from pyswmm import Nodes;
from pyswmm import SystemStats;
from pyswmm import Simulation;
from swmmtoolbox import swmmtoolbox;
import seaborn as sns; 
import matplotlib.pyplot as plt

#impervious upslope
def update_GA(read,write,perm_line, H_i,K,IMD):
    fin = open(read)
    fout = open(write, "wt")
    for perm_line in fin: #note that these values have to be updated if the read_file inputs change
        fout.write(perm_line.replace('S2              	12.6      	0.01      	0.097     ', 
                                'S2              	'+ str(H_i) + '      	'+ str(K) + '      	'+ str(IMD) + ''))
    fin.close()
    fout.close()

def run_sim_sens(
    g,soil,read,write,perm_line,x,
    A1,A2,
    W,So,n_perv,n_imp,d_perv,d_imp,P):
    #soil is the soil dictionary
    # x is the row in the soils dictionary
    # p is pervious fraction
    # W1 is width, held constant
    # S is slope
    # P is precip intensity (cm/hr)
    #sigma indicates impervious or pervious upslope (-1 or 1)
    
    """run pyswmm
    """
    #Set parameters
    K=soil["K"][x] #pulls the K value from soil dictionary
    H_i=soil["H_i"][x] #suction head
    IMD=soil["IMD"][x] 

    perm_line = perm_line
    update_GA(read,write, perm_line, H_i,K,IMD)
    
#     with open(write) as f:
#         for i, line in enumerate(f, 1):
#             if i ==perm_line:
#                 break
#         print(line)
        
    #run simulation, looping over soils dictionary

    with Simulation(write,'r') as sim:
        d=dict()
        
        s1 =  Subcatchments(sim)["S1"]
        s2 =  Subcatchments(sim)["S2"]
        
        s1.Imperv_N=n_imp
        s2.Imperv_N=n_imp
        
        s1.Perv_N = n_perv
        s2.Perv_N = n_perv
        
        s1.Imperv_S = d_imp
        s2.Imperv_S = d_imp
    
        s1.Perv_S = d_perv
        s2.Perv_S = d_perv
        print(s2.Perv_S)
        with open(write) as f:
            for i, line in enumerate(f, 1):
                if i ==62:
                    break
            print(line)
            
        s1.area = A1
        s2.area = A2
        
        s1.width=W
        s2.width=W

        s1.slope=So
        s2.slope=So

        for step in sim:
            pass
        system_routing = SystemStats(sim)
        sim.report()
        d[A1,A2,K,W,So,n_perv,n_imp,d_perv,d_imp,P,H_i,IMD]=s2.statistics  # sets dictionary keys 
        g.update(d)
        return g


def run_sim(g,soil,read,write,perm_line,x,A1,A2,W1,W2,S,P):
    #   x is the row in the soils dictionary
    #   A1 is impervious subcatchment area
    #   A2 is pervious subcatchment area
    #   W is catchment width
    
    """run pyswmm
    """
    #   Set soil parameters
    perm_line = perm_line
    K=soil["K"][x] #    pulls the K value from soil dictionary
    H_i=soil["H_i"][x] #    suction head
    IMD=soil["IMD"][x] #    initial moisture deficit
    
    update_GA(read,write, perm_line,H_i,K,IMD)
#     with open(write) as f:
#         for i, line in enumerate(f, 1):
#             if i ==perm_line:
#                 break
#     print(line)
    with Simulation(write,'r') as sim:

        d=dict()
        s1 =  Subcatchments(sim)["S1"]
        s2 =  Subcatchments(sim)["S2"]
        s1.area = A1
        s2.area = A2
        s1.width = W1
        s2.width = W2
        s1.slope = S/100.
        s2.slope = S/100.
        for step in sim:
            pass
        system_routing = SystemStats(sim)
        sim.report()
        d[K,W1,A1,A2,S,P,H_i,IMD]=s2.statistics # sets dictionary keys
        g.update(d)
        return g
    return g

def convert_to_df(x,e):
    fileID=pd.DataFrame.from_dict(e,'index')
    fileID=fileID.sort_index().reset_index()
    return fileID

def rename_df(fileID,sim_df):
    sim_df = pd.DataFrame(fileID)
    sim_df=core1.reset_index()
    sim_df.rename({'level_0': 'Ks','level_1': 'W', 'level_2': 'A1','level_3': 'A2','level_4': 'S','level_5': 'P','level_6': 'H_i'}, axis=1, inplace=True)
    sim_df['scenario']=run
    sim_df['A']=sim_df['A2']+sim_df['A1']
    sim_df['fV']=sim_df['A2']/sim_df['A']

def scatter_IF():
    sns.set(color_codes=True)
    sns.set(font_scale=2)  
    plt.figure(1, figsize=(10,8), )
    sns.set_context("talk",rc={"lines.linewidth": 2.5})
    sns.set(style="whitegrid", font_scale=2,rc={'figure.figsize':(12,8)})
    sns.lmplot( x="fV", y="IF", data=df_core, fit_reg=False, hue='Ks', logx =False, height = 10, legend= False)

    plt.legend(title = '$K$ (in/hr)', loc = 'best')
    plt.xlabel('Pervious fraction ($fV$)')
    plt.ylabel('Infiltration fraction ($IF$)')
    plt.title("IF across range of pervious fractions")
    
