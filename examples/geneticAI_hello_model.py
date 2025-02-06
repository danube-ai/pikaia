import numpy as np

import pikaia
import pikaia.alg
          

if __name__ == "__main__":

    # setup numeric printout format
    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    rawdata = np.zeros([3,3])
    rawdata[:,:] = [[ 300, 10, 2],
                    [ 600,  5, 2],
                    [1500,  4, 1]]
    gvfitnessrules = ["inv_percentage", "inv_percentage", "inv_percentage"]
    data = pikaia.alg.Population(rawdata, gvfitnessrules)
        
    strategy = ["GS Dominant", "OS Balanced"]
    
    iterations = 1

    # creating the genetic model
    model = pikaia.alg.Model(data, strategy)

    initialgenefitness = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    model.complete_run(initialgenefitness, iterations)
    
