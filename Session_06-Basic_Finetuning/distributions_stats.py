import glob
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


notes = []
complaints = []
social_history = []
for note_path in tqdm(glob.glob('gens/*.txt')):
    with open(note_path, 'r') as f:
        notes.append(f.read())

    note = notes[-1]
    try:
        cc_idx = note.find("Chief Complaint:")
        cc_nl_idx = note.find('\n', cc_idx)
        cc = note[cc_nl_idx + 1:note.find('\n', cc_nl_idx + 1)].lower()
    except:
        cc = "ERROR"
    complaints.append(cc)

    try:
        cc_idx = note.find("Social History:")
        if cc_idx != -1:
            cc_nl_idx = note.find('\n', cc_idx)
            cc = note[cc_nl_idx + 1:note.find('\n', cc_nl_idx + 1)].lower()
            cc = cc[:min(len(cc), 10)]
        else:
            cc = '[not present]'
        if cc != "___":
            print(cc, note_path)
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
plt.title("From 1000 generated notes\n(temperature 3, beam search 5, sampling)")
plt.xscale('log')
plt.xlim(0.0005, 1.1)
plt.tight_layout()
plt.savefig("gen_complaints.png")

df = pd.DataFrame({"social_history": social_history})
df2 = df.groupby(['social_history'])['social_history'].count().sort_values(ascending=False)
print(df)
print(df2)


plt.figure()
df['social_history_rep'] = [sh if sh in ['___', '', '[not present]'] else '[some text]' for sh in df['social_history']]
df['social_history_rep'].value_counts(normalize=True)[:5].plot(kind='barh')
plt.gca().invert_yaxis()
plt.xlabel("Proportion")
plt.title("From 1000 generated notes\n(temperature 3, beam search 5, sampling)")
plt.xscale('log')
plt.xlim(0.0005, 1.1)
plt.tight_layout()
plt.savefig("gen_social.png")

