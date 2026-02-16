from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from copy import deepcopy



class Betti_Curve:
    def __init__(self, BC, data):
        self.data = data
        self.BC = BC
    
    def load_pcloud(self, data_path):
        pcloud = torch.stack([torch.load(file, map_location="cpu") for file in data_path], dim = 0)

        # array
        pcloud = pcloud.detach().cpu().numpy()
        return pcloud
        
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
    

def main():
    # Path to the gram_neg host data 
    gram_neg_hosts = Path("evo2_data/gram_neg/hosts")
    gram_pos_hosts = Path("evo2_data/gram_pos/hosts")

    gnh_tensor_files = sorted(gram_neg_hosts.glob("*.pt"))
    gph_tensor_files = sorted(gram_pos_hosts.glob("*.pt"))
    
    dataset = [gnh_tensor_files, gph_tensor_files]
    
    BC = BettiCurve()
    
    #BC_Graph(gnh_tensor_files)
    declare_BC = Betti_Curve(BC, dataset)
            
    print("Plotting...")
    #figs = declare_BC.graph()
    figs = declare_BC.graph()
    
    fig = go.Figure()
    
    # combine two graph in one 

    for tr in figs[0].data:
        tr2 = deepcopy(tr)
        tr2.name = f"gnh {tr2.name}"
        tr2.legendgroup = "gnh"
        fig.add_trace(tr2)

    for tr in figs[1].data:
        tr2 = deepcopy(tr)
        tr2.name = f"gph {tr2.name}"
        tr2.legendgroup = "gph"
        tr2.line = dict(dash="dash")   # just to visually separate groups
        fig.add_trace(tr2)

    fig.update_layout(
        title="Betti curves: gnh vs gph",
        xaxis_title="Filtration parameter",
        yaxis_title="Betti number",
    )
    fig.show()
    
    
if __name__ == "__main__":
    main()


