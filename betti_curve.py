from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from copy import deepcopy
import random

#class for functions related to betti curve 
class Betti_Curve:
    def __init__(self, BC, data):
        self.data = data
        self.BC = BC
    
    def load_pcloud(self, data_path):
        pcloud = torch.stack([torch.load(file, map_location="cpu") for file in data_path], dim = 0)

        # array
        pcloud = pcloud.detach().cpu().numpy()
        return pcloud
    
    # calculate the persistence diagram
    def persistence_diagram(self):
        # list of 2D point clouds
        pc_list = [self.load_pcloud(paths) for paths in self.data]
                  
        VR = VietorisRipsPersistence(homology_dimensions= (0,1), metric= "euclidean", n_jobs = None)
        
        Persistence_Diagram = VR.fit_transform(pc_list)
        return Persistence_Diagram
        
    #graph betti curves
    def graph(self):
        #torch size([n, 4096])
        PD = self.persistence_diagram()
        
        betti = self.BC.fit_transform(PD)
        
        figs = []
        for i in range(len(betti)):
            figs.append(self.BC.plot(betti, sample=i))
        
        return figs
    
    def save_PD(self):
        PD = self.persistence_diagram()
        
        for i in range(PD.shape[0]):
            np.savetxt(f"pd_sample{i}.csv", PD[i], delimiter=",", header="birth,death,dim", comments="")

# Plot the Betti Curves on the same graph for easier comparison 
def plot_BC(BC):
    print("Plotting...")
    figs = BC.graph()
    
    fig = go.Figure()
    
    # combine two graph in one 

    for tr in figs[0].data:
        tr2 = deepcopy(tr)
        tr2.name = f" Sampled gnh {tr2.name}"
        tr2.legendgroup = "Sampled gnh"
        fig.add_trace(tr2)

    for tr in figs[1].data:
        tr2 = deepcopy(tr)
        tr2.name = f"gph {tr2.name}"
        tr2.legendgroup = "gph"
        tr2.line = dict(dash="dash")   #just to visually separate groups
        fig.add_trace(tr2)

    fig.update_layout(
        title="Betti curves: Sampled gnh vs gph",
        xaxis_title="Filtration parameter",
        yaxis_title="Betti number",
    )
    fig.show()
    
def gnh_vs_gph(gph_tensor_files):
    print("Choosing random points")
    
    random.seed(0)
    gnh_tensor_files = random.sample(gnh_tensor_files, k=209)

    
    dataset = [gnh_tensor_files, gph_tensor_files]
    
    BC = BettiCurve()
    
    #BC_Graph(gnh_tensor_files)
    declare_BC = Betti_Curve(BC, dataset)
    
    #plot_BC(declare_BC= declare_BC)
    declare_BC.save_PD()

def bootstrapping_gnh(gnh_tensor_files):
    m = 209
    B = 10
    # seeing the gnh's distribution stability by random sampling 209 points ten times and viewing their betti curve 
    random.seed(0)
    dataset = [random.sample(gnh_tensor_files, k=m) for _ in range(B)]
    print("Random Sampling Complete")
    BC = BettiCurve()
    
    #BC_Graph(gnh_tensor_files)
    declare_BC = Betti_Curve(BC, dataset)
    
    print("Plotting...")
    figs = declare_BC.graph()
    
    fig = go.Figure()
    
    # combine two graph in one 
    for i, f in enumerate(figs):
        for tr in f.data:
            tr2 = deepcopy(tr)
            tr2.opacity = 0.75         # make bootstraps faint so overlap is readable
            fig.add_trace(tr2)

    fig.update_layout(
        title=f"Bootstrapped Gram-Negative Hosts Betti curves (B={B}, n={m})",
        xaxis_title="Filtration parameter ε",
        yaxis_title="Betti number",
        hovermode="x unified"
    )
    fig.update_yaxes(rangemode="tozero")

    fig.show()

def main():
    # Path to the gram_neg host data 
    print("Starting Code ")
    gram_neg_hosts = Path("evo2_data/gram_neg/hosts")
    gram_pos_hosts = Path("evo2_data/gram_pos/hosts")

    gnh_tensor_files = sorted(gram_neg_hosts.glob("*.pt"))
    gph_tensor_files = sorted(gram_pos_hosts.glob("*.pt"))
    
    # pass the input as an array for easier comparison later on
    # easily can include more datasets, just by adding to this array
    dataset = [gnh_tensor_files, gph_tensor_files]
    
    BC = BettiCurve()
    
    # declares an object of the class
    declare_BC = Betti_Curve(BC, dataset)
    plot_BC(declare_BC)
   
    
if __name__ == "__main__":
    main()


