#! /usr/bin/python3 -i
# coding=utf-8

import os

try:
  from transformers.utils import cached_file
except:
  from transformers.file_utils import cached_path,hf_bucket_url
  cached_file=lambda x,y:cached_path(hf_bucket_url(x,y))

def download(rootdir,file,dir="."):
  import shutil
  t=os.path.join(dir,"filesize.txt")
  shutil.copy(cached_file(rootdir,os.path.dirname(file)+"/filesize.txt"),t)
  with open(t,"r") as f:
    r=f.read()
  f=os.path.basename(file)
  ft=0
  for t in r.split("\n"):
    s=t.split()
    if len(s)==2:
      if s[0]==f:
        ft=int(s[1])
  if ft==0:
    return
  shutil.copy(cached_file(rootdir,file),os.path.join(dir,f))

def checkdownload(rootdir,model,dir="."):
  while True:
    t=os.path.join(dir,"filesize.txt")
    with open(t,"r") as f:
      r=f.read()
    for t in r.split("\n"):
      s=t.split()
      if len(s)==2:
        f=os.path.join(dir,s[0])
        i=int(s[1])
        try:
          j=os.path.getsize(f)
        except:
          j=-1
        if i!=j:
          download(rootdir,model+s[0],dir)
          break
    else:
      return

