"""
    Read parameters and run Hallue model
"""


# ---- Import package
import os, sys
sys.path.append(os.path.normpath(r"E:\EauxSCAR\adeqwat"))
from pymarthe import MartheModel
from pymarthe import marthe_utils

# --- Set directories
par_dir = 'param'
sim_dir = 'sim'

# ---- Load model
mm = MartheModel('hallue.rma')

# ---- Load initial pumping data
mm.add_pump(pastp_file = 'hallue.pastp', mode = 'aquifer')

# ---- Set pumping data from parameter file

par_files = [os.path.join(par_dir,f) for f in os.listdir(par_dir)]
mm.aqpump.set_param(par_files)

# ---- Write pumping data in Marthe files
mm.aqpump.write_data()

# ---- Run model
mm.run_model(exe_name='Marth_R8', silent = True, verbose = False)

# ---- Post-processing to export simulated head
head_df = marthe_utils.read_mi_prn('historiq.prn').xs('Charge',level='type', axis=1)

for wn in head_df.columns:
    # ---- Write in 'sim' directory
    head_df[wn].to_csv(os.path.join(sim_dir, f'{wn}.dat'),
                       sep = '\t', header=False, index=True)

