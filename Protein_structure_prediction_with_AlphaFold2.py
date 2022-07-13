#!/usr/bin/env python
# coding: utf-8

# # Protein structure prediction with AlphaFold2
# ## Please Note: The AlphaFold parameters used in this notebook are available only for non-commercial use under the terms of the CC BY-NC 4.0 license (NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation).
# ## This is a simple version requires query_sequence and MSA (a3m format) for structure prediction. 
# 
# ## <font color='red'>Imprtant notice</font>: Due to the high demands on system resources this notebook should be run while paying attention to system load on 'hugorm' - please reach to DataMan or Bioinf / Design before running several predicitons in parallel. 
# 
# If have any questions, contact PTI@novozymes.coom

# In[1]:

from typing import Dict
import os
#import mock
import numpy as np
import pickle
#import py3Dmol

import os
import sys

os.system('cp -r /z/home/pti/projects/alpha_fold_pred/alphafold .')
os.system('cp -r /z/home/pti/projects/alpha_fold_pred/params/ .')

import alphafold
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model


import sys

raw_sequence = sys.argv[1]
query_sequence = raw_sequence #+ 'GS'*30 + raw_sequence + 'GS'*30 + raw_sequence + 'GS'*30 + raw_sequence
#query_sequence


# # Input your favourite protein sequence :)

#query_sequence = 'TPPGTPSKSAAQSQLNALTVKTEGSMSGYSRDLFPHWISQGSGCDTRQVVLKRDADSYSGNCPVTSGSWYSYYDGVTFTNPSDLDIDHIVPLAEAWRSGASSWTTSKRQDFANDLSGPQLIAVSASTNRSKGDQDPSTWQPPRSGAACGYSKWWISTKYKWGLSLQSSEKTALQGMLNSCSY' #@param {type:"string"} #pde wild type
#jobname = '5xys_4mer'
jobname =  sys.argv[2]  #'5xys_4mer' 

#!cp /z/home/pti/projects/alpha_fold_pred/{jobname}.a3m .

# remove whitespaces
jobname="".join(jobname.split())

with open(f"{jobname}.fasta", "w") as text_file:
    text_file.write(">1\n%s" % query_sequence)

# number of models to use
num_models = 5 #[1,2,3,4,5]
use_amber = True 
use_msa = True 


# # Run homologs search
# This will take around 20 mins - 1 hour

# In[5]:


hhblits_database = "/z/linux/hhsuite/databases/UniRef30_2020_02/UniRef30_2020_02"
print(f'/z/linux/download/hh-suite3/hh-suite/build/bin/hhblits -i {jobname}.fasta  -oa3m {jobname}.a3m -d {hhblits_database} -maxfilt 100000 -realign_max 100000  -all -B 100000 -Z 100000 -n 4 -e 0.001')
os.system(f'/z/linux/download/hh-suite3/hh-suite/build/bin/hhblits -i {jobname}.fasta  -oa3m {jobname}.a3m -d {hhblits_database} -maxfilt 100000 -realign_max 100000  -all -B 100000 -Z 100000 -n 4 -e 0.001')


# # Setup the model

# In[6]:


#This part is mainly written by Martin Steinegger and Sergey Ovchinnikov
if "model" not in dir():
  import warnings
  warnings.filterwarnings('ignore')
  import os
  import sys
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  import tensorflow as tf
  import numpy as np
  import pickle
  import py3Dmol
  import matplotlib.pyplot as plt
  from alphafold.common import protein
  from alphafold.data import pipeline
  from alphafold.data import templates
  from alphafold.model import data
  from alphafold.model import config
  from alphafold.model import model

  import ipywidgets
  from ipywidgets import interact, fixed

  tf.get_logger().setLevel('ERROR')

if use_amber and "relax" not in dir():
  sys.path.insert(0, '/usr/local/lib/python3.7/site-packages/')
  from alphafold.relax import relax

if "model_params" not in dir(): model_params = {}
for model_name in ["model_1","model_2","model_3","model_4","model_5"][:num_models]:
  if model_name not in model_params:
    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = 1
    model_params[model_name] = data.get_model_haiku_params(model_name=model_name, data_dir=".")
    if model_name == "model_1":
      model_runner_1 = model.RunModel(model_config, model_params[model_name])
    if model_name == "model_3":
      model_runner_3 = model.RunModel(model_config, model_params[model_name])

def mk_mock_template(query_sequence):
  # since alphafold's model requires a template input
  # we create a blank example w/ zero input, confidence -1
  ln = len(query_sequence)
  output_templates_sequence = "-"*ln
  output_confidence_scores = np.full(ln,-1)
  templates_all_atom_positions = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
  templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
  templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
                                                                    templates.residue_constants.HHBLITS_AA_TO_ID)
  template_features = {'template_all_atom_positions': templates_all_atom_positions[None],
                       'template_all_atom_masks': templates_all_atom_masks[None],
                       'template_sequence': [f'none'.encode()],
                       'template_aatype': np.array(templates_aatype)[None],
                       'template_confidence_scores': output_confidence_scores[None],
                       'template_domain_names': [f'none'.encode()],
                       'template_release_date': [f'none'.encode()]}
  return template_features

def set_bfactor(pdb_filename, bfac):
  I = open(pdb_filename,"r").readlines()
  O = open(pdb_filename,"w")
  for line in I:
    if line[0:6] == "ATOM  ":
      seq_id = int(line[23:26].strip()) - 1
      O.write("{prefix}{bfac:6.2f}{suffix}".format(prefix=line[:60], bfac=bfac[seq_id], suffix=line[66:]))
  O.close()

def predict_structure(prefix, feature_dict, do_relax=True, random_seed=0):  
  """Predicts structure using AlphaFold for the given sequence."""

  # Run the models.
  plddts = []
  unrelaxed_pdb_lines = []
  relaxed_pdb_lines = []

  for model_name, params in model_params.items():
    print(f"running {model_name}")
    # swap params to avoid recompiling
    # note: models 1,2 have diff number of params compared to models 3,4,5
    if any(str(m) in model_name for m in [1,2]): model_runner = model_runner_1
    if any(str(m) in model_name for m in [3,4,5]): model_runner = model_runner_3
    model_runner.params = params
    
    processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
    prediction_result = model_runner.predict(processed_feature_dict)
    unrelaxed_protein = protein.from_prediction(processed_feature_dict,prediction_result)
    unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
    plddts.append(prediction_result['plddt'])

    if do_relax:
      # Relax the prediction.
      amber_relaxer = relax.AmberRelaxation(max_iterations=0,tolerance=2.39,
                                            stiffness=10.0,exclude_residues=[],
                                            max_outer_iterations=20)      
      relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
      relaxed_pdb_lines.append(relaxed_pdb_str)

  # rerank models based on predicted lddt
  lddt_rank = np.mean(plddts,-1).argsort()[::-1]
  plddts_ranked = {}
  for n,r in enumerate(lddt_rank):
    print(f"model_{n+1} {np.mean(plddts[r])}")

    unrelaxed_pdb_path = f'{prefix}_unrelaxed_model_{n+1}.pdb'    
    with open(unrelaxed_pdb_path, 'w') as f: f.write(unrelaxed_pdb_lines[r])
    set_bfactor(unrelaxed_pdb_path,plddts[r]/100)

    if do_relax:
      relaxed_pdb_path = f'{prefix}_relaxed_model_{n+1}.pdb'
      with open(relaxed_pdb_path, 'w') as f: f.write(relaxed_pdb_lines[r])
      set_bfactor(relaxed_pdb_path,plddts[r]/100)

    plddts_ranked[f"model_{n+1}"] = plddts[r]

  return plddts_ranked


# # Predict structure
# 
# This will take around half hour (100 AA protein) to 9 hours (1300 AA protein).
# 
# Two pdbs will be generated, relaxed (with molecular dynamics simulations refinement at the end) and unrelaxed*.pdb

# In[7]:


a3m_lines = "".join(open(f"{jobname}.a3m","r").readlines())
msa, deletion_matrix = pipeline.parsers.parse_a3m(a3m_lines)
query_sequence = msa[0]
feature_dict = {
    **pipeline.make_sequence_features(sequence=query_sequence,
                                      description="none",
                                      num_res=len(query_sequence)),
    **pipeline.make_msa_features(msas=[msa],deletion_matrices=[deletion_matrix]),
    **mk_mock_template(query_sequence)
}
plddts = predict_structure(jobname, feature_dict, do_relax=use_amber)


# # Display the prediction, and find the pdb file in your current directory

# In[8]:


#p = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
#p.addModel(open(f"{jobname}_relaxed_model_1.pdb",'r').read(),'pdb')
#p.setStyle({'cartoon': {'color':'spectrum'}})
#p.zoomTo()
#p.show()


# In[ ]:





# In[ ]:





# In[ ]:




