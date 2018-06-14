import pandas as pd
import gzip
import numpy as np

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('meta_Beauty.json.gz')

data=df[['asin','categories','price','brand']]
data=np.array(data)
np.save('meta_required.npy',data)