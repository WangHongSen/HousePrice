import pandas as pd 
from sklearn.impute import SimpleImputer
df = pd.read_csv('train.csv')
# columns = df.columns
# total = df.shape[0]
# f = open("statics.txt","w",encoding='utf-8')
# def count():
#     for i in columns:      
#         l = df[i].value_counts()
#         f.write(str(l))
#         f.write("\nmiss {}".format(total - sum(list(l))))
#         f.write("\n*************************************\n")
# count()
imp = SimpleImputer(missing_values = 'NaN',
              strategy ='mean')
imp.fit_transform(df['区'])