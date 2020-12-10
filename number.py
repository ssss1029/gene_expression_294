import pandas as pd
import numpy as np
data = pd.read_csv('dataset/E004/classification/test.csv')
data.columns = np.arange(8)
print(data.groupby(by=7).count())
