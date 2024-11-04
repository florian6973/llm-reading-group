import glob
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt

# dataset_path = '/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/11k.csv'
dataset_path = '/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'

notes = pd.read_csv(dataset_path, index_col=0)['text']
# print(notes)

complaints = []
social_history = []
for i, note in enumerate(notes):
    # print(note)
    # exit()
    try:
        cc_idx = note.find("Chief Complaint:")
        cc_nl_idx = note.find('\n', cc_idx)
        cc = note[cc_nl_idx + 1:note.find('\n', cc_nl_idx + 1)].lower()
    except:
        cc = "ERROR"
    complaints.append(cc)

    try:
        cc_idx = note.lower().find("Social History:".lower())
        cc_nl_idx = note.find('\n', cc_idx)
        cc = note[cc_nl_idx + 1:note.find('\n', cc_nl_idx + 1)].lower()
        if cc != "___":
            print(cc, i)
    except:
        cc = "ERROR"
    social_history.append(cc)
    # print(cc)
    # exit()

# print(notes[-1])
df = pd.DataFrame({"complaints": complaints})
df2 = df.groupby(['complaints'])['complaints'].count().sort_values(ascending=False)
print(df)
print(df2)

plt.figure()
df['complaints'].value_counts(normalize=True)[:20].plot(kind='barh')
plt.gca().invert_yaxis()
plt.xlabel("Proportion")
plt.title("From the 330,000 original notes")
plt.xlim(0.0005, 1.1)
plt.xscale('log')
plt.tight_layout()
plt.savefig("orig_complaints.png")

df = pd.DataFrame({"social_history": social_history})
df2 = df.groupby(['social_history'])['social_history'].count().sort_values(ascending=False)
print(df)
print(df2)
df2.to_csv('sh_distrib.csv')


print(len(notes))



plt.figure()
df['social_history_rep'] = [sh if sh in ['___', ''] else '[some text]' for sh in df['social_history']]
df['social_history_rep'].value_counts(normalize=True)[:5].plot(kind='barh')
plt.gca().invert_yaxis()
plt.xlabel("Proportion")
plt.title("From the 330,000 original notes")
plt.xscale('log')
plt.xlim(0.0005, 1.1)
plt.tight_layout()
plt.savefig("orig_social.png")