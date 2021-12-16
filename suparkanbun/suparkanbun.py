#! /usr/bin/python3 -i
# coding=utf-8

import os
PACKAGE_DIR=os.path.abspath(os.path.dirname(__file__))
DOWNLOAD_DIR=os.path.join(PACKAGE_DIR,"models")

from transformers.file_utils import hf_bucket_url
MODEL_URL=hf_bucket_url("KoichiYasuoka/SuPar-Kanbun","suparkanbun/models/")

import numpy
from spacy.language import Language
from spacy.symbols import LANG,NORM,LEMMA,POS,TAG,DEP,HEAD
from spacy.tokens import Doc,Span,Token
from spacy.util import get_lang_class

class SuParKanbunLanguage(Language):
  lang="lzh"
  max_length=10**6
  def __init__(self,BERT,Danku):
    self.Defaults.lex_attr_getters[LANG]=lambda _text:"lzh"
    try:
      self.vocab=self.Defaults.create_vocab()
      self.pipeline=[]
    except:
      from spacy.vocab import create_vocab
      self.vocab=create_vocab("lzh",self.Defaults)
      self._components=[]
      self._disabled=set()
    self.tokenizer=SuParKanbunTokenizer(BERT,Danku,self.vocab)
    self._meta={
      "author":"Koichi Yasuoka",
      "description":"derived from SuParKanbun",
      "lang":"SuParKanbun_lzh",
      "license":"MIT",
      "name":"SuParKanbun_lzh",
      "parent_package":"suparkanbun",
      "pipeline":"Tokenizer, POS-Tagger, Parser",
      "spacy_version":">=2.1.0"
    }
    self._path=None

class SuParKanbunTokenizer(object):
  to_disk=lambda self,*args,**kwargs:None
  from_disk=lambda self,*args,**kwargs:None
  to_bytes=lambda self,*args,**kwargs:None
  from_bytes=lambda self,*args,**kwargs:None
  def __init__(self,bert,danku,vocab):
    from supar import Parser
    self.bert=bert
    self.vocab=vocab
    self.simplify={}
    if bert.startswith("guwenbert"):
      from suparkanbun.simplify import simplify
      self.simplify=simplify
    d=os.path.join(DOWNLOAD_DIR,bert+".pos")
    self.tagger=AutoModelTagger(d)
    f=os.path.join(d,bert+".supar")
    self.supar=Parser.load(f)
    if danku:
      d=os.path.join(DOWNLOAD_DIR,bert+".danku")
      self.danku=AutoModelTagger(d,["B","E","E2","E3","M","S"])
    else:
      self.danku=None
    self.gloss=MakeGloss()
  def __call__(self,text):
    from suparkanbun.tradify import tradify
    t=""
    for c in text:
      if c in self.simplify:
        t+=self.simplify[c]
      else:
        t+=c
    if self.danku!=None:
      u=t.replace("\n","")
      t=""
      while len(u)>500:
        s=self.danku(u[0:500])
        r=""
        for c,p in s:
          r+=c
          if p=="S" or p=="E":
            r+="\n"
        r="\n".join(r.split("\n")[0:-2])+"\n"
        t+=r
        u=u[len(r.replace("\n","")):]
      s=self.danku(u)
      for c,p in s:
        t+=c
        if p=="S" or p=="E":
          t+="\n"
    if len(t)<500:
      p=self.tagger(t.replace("\n",""))
    else:
      p=[]
      u=""
      for s in t.strip().split("\n"):
        u+=s
        if len(u)>400:
          p+=self.tagger(u)
          u=""
      if len(u)>0:
        p+=self.tagger(u)
    u=self.supar.predict([[c for c in s] for s in t.strip().split("\n")],lang=None)
    t=text.replace("\n","")
    i=0
    w=[]
    for s in u.sentences:
      v=[]
      for h,d in zip(s.values[6],s.values[7]):
        j=t[i]
        k=tradify[j] if j in tradify else j
        v.append({"form":j,"lemma":k,"pos":p[i][1],"head":h,"deprel":d})
        i+=1
      for j in reversed(range(0,len(v)-1)):
        if v[j]["deprel"]=="compound" and v[j]["head"]==j+2 and v[j]["pos"]==v[j+1]["pos"]:
          k=v.pop(j)
          v[j]["form"]=k["form"]+v[j]["form"]
          v[j]["lemma"]=k["lemma"]+v[j]["lemma"]
          for k in range(0,len(v)):
            if v[k]["head"]>j+1:
              v[k]["head"]-=1
      w.append(list(v))
    vs=self.vocab.strings
    r=vs.add("ROOT")
    words=[]
    lemmas=[]
    pos=[]
    tags=[]
    feats=[]
    heads=[]
    deps=[]
    spaces=[]
    norms=[]
    for s in w:
      for i,t in enumerate(s):
        form=t["form"]
        words.append(form)
        lemmas.append(vs.add(t["lemma"]))
        p=t["pos"].split(",")
        xpos=",".join(p[0:4])
        pos.append(vs.add(p[4]))
        tags.append(vs.add(xpos))
        feats.append(p[5])
        if t["deprel"]=="root":
          heads.append(0)
          deps.append(r)
        else:
          heads.append(t["head"]-i-1)
          deps.append(vs.add(t["deprel"]))
        spaces.append(False)
        g=self.gloss(form,xpos)
        if g!=None:
          norms.append(vs.add(g))
        else:
          norms.append(vs.add(form))
    doc=Doc(self.vocab,words=words,spaces=spaces)
    a=numpy.array(list(zip(lemmas,pos,tags,deps,heads,norms)),dtype="uint64")
    doc.from_array([LEMMA,POS,TAG,DEP,HEAD,NORM],a)
    try:
      doc.is_tagged=True
      doc.is_parsed=True
    except:
      for i,j in enumerate(feats):
        if j!="_" and j!="":
          doc[i].set_morph(j)
    return doc

class AutoModelTagger(object):
  def __init__(self,dir,label=None):
    from suparkanbun.download import checkdownload
    from transformers import AutoModelForTokenClassification,AutoTokenizer
    checkdownload(MODEL_URL+os.path.basename(dir)+"/",dir)
    self.model=AutoModelForTokenClassification.from_pretrained(dir)
    self.tokenizer=AutoTokenizer.from_pretrained(dir)
    self.label=label if label else self.model.config.id2label
  def __call__(self,text):
    import torch
    input=self.tokenizer.encode(text,return_tensors="pt")
    output=self.model(input)
    predict=torch.argmax(output[0],dim=2)
    return [(t,self.label[p]) for t,p in zip(text,predict[0].tolist()[1:])]

class MakeGloss(object):
  def __init__(self,file=None):
    if file==None:
      file=os.path.join(DOWNLOAD_DIR,"gloss.orig.txt")
    with open(file,"r",encoding="utf-8") as f:
      r=f.read()
    self.gloss={}
    for s in r.split("\n"):
      t=s.split()
      if len(t)==4:
        self.gloss[(t[0],t[2])]=t[3]
      elif len(t)==5:
        self.gloss[(t[0],t[3])]=t[4]
    self.extra={
      "n,名詞,人,姓氏":"[surname]",
      "n,名詞,人,名":"[given-name]",
      "n,名詞,主体,書物":"[book-name]",
      "n,名詞,主体,国名":"[country-name]",
      "n,名詞,固定物,地名":"[place-name]"
    }
  def __call__(self,form,xpos):
    if (form,xpos) in self.gloss:
      return self.gloss[(form,xpos)]
    if xpos in self.extra:
      return self.extra[xpos]
    if xpos=="n,名詞,時,*":
      if len(form)>1:
        return "[era-name]"
    return None

def load(BERT="roberta-classical-chinese-base-char",Danku=False):
  return SuParKanbunLanguage(BERT,Danku)

def to_conllu(item,offset=1):
  if type(item)==Doc:
    return "".join(to_conllu(s)+"\n" for s in item.sents)
  elif type(item)==Span:
    return "# text = "+str(item)+"\n"+"".join(to_conllu(t,1-item.start)+"\n" for t in item)
  elif type(item)==Token:
    m="_" if item.whitespace_ else "SpaceAfter=No"
    if item.norm_!="":
      if item.norm_!=item.orth_:
        m="Gloss="+item.norm_+"|"+m
        m=m.replace("|_","")
    l=item.lemma_
    if l=="":
      l="_"
    t=item.tag_
    if t=="":
      t="_"
    try:
      f=str(item.morph)
      if f.startswith("<spacy") or f=="":
        f="_"
    except:
      f="_"
    return "\t".join([str(item.i+offset),item.orth_,l,item.pos_,t,f,str(0 if item.head==item else item.head.i+offset),item.dep_.lower(),"_",m])
  return "".join(to_conllu(s)+"\n" for s in item)

