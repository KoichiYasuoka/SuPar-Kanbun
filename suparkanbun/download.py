#! /usr/bin/python3 -i
# coding=utf-8

import os

def download(url,file,dir="."):
  import shutil
  from transformers.file_utils import cached_path
  t=os.path.join(dir,"filesize.txt")
  shutil.copy(cached_path(url+"filesize.txt"),t)
  with open(t,"r") as f:
    r=f.read()
  ft=0
  for t in r.split("\n"):
    s=t.split()
    if len(s)==2:
      if s[0]==file:
        ft=int(s[1])
  if ft==0:
    return
  shutil.copy(cached_path(url+file),os.path.join(dir,file))

def checkdownload(url,dir="."):
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
          download(url,s[0],dir)
          break
    else:
      return

