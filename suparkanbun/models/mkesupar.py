#! /usr/bin/python3
import glob,torch,pickle,safetensors.torch
import dill as esupar_dill
class EsuparUnpickler(esupar_dill.Unpickler):
  def find_class(self,module,name):
    if module.startswith("supar"):
      module="esupar."+module
    return pickle.Unpickler.find_class(self,module,name)
esupar_dill.Unpickler=EsuparUnpickler

for f in glob.glob("*/*.supar"):
  s=torch.load(f,pickle_module=esupar_dill,weights_only=False)
  d=s.pop("state_dict")
  d["esupar.config"]=torch.tensor([c for c in pickle.dumps(s)],dtype=torch.uint8)
  safetensors.torch.save_file(d,f.replace(".supar",".esupar"))
