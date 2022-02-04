import ast
from  collections import defaultdict
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def load_log(log_path):
    dict = defaultdict(list)
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            dict_line = ast.literal_eval(line)
            for key, value in dict_line.items():
                dict[key].append(value)
    return dict

log_dir = "/home/pyuan2/Projects2021/mae/jobdir/finetune_lung_mr75/vit_large_patch16_e100_input32_luna_blr1e3"
log_path = os.path.join(log_dir, "log.txt")

dict = load_log(log_path)
df = pd.DataFrame(dict)

sns.set_style("whitegrid")

plt.figure()
df_loss = df[["train_loss", "test_loss", "epoch"]].melt("epoch", value_name="loss")
sns.lineplot(x="epoch", y="loss", hue="variable", data=df_loss)
plt.tight_layout()

plt.figure()
df_lr = df[["train_lr", "epoch"]]
sns.lineplot(x="epoch", y="train_lr", data=df_lr)
plt.tight_layout()

plt.figure()
df_acc = df[["test_acc1", "test_acc5", "epoch"]].melt("epoch", value_name="acc")
sns.lineplot(x="epoch", y="acc", hue="variable", data=df_acc)
plt.tight_layout()

plt.show()

print("")