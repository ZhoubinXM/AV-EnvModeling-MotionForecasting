import pandas as pd

# filename = "/data/jerome.zhou/database/accident/normal_db_cutin_adms_result_with_tag_0310.csv"
# filename = "/data/jerome.zhou/database/accident/normal_db_cutin_adms_result_with_tag_0310.csv"
filename = "/data/jerome.zhou/database/accident/risk_db_cutin_adms_result_with_tag_0310_2000.csv"

root = "/data/guifeng/crash_dataset/risk/"

df = pd.read_csv(filename, low_memory=False)
groups = df.groupby(df.c_dlb_uuid)

for dlb in groups:
    dlb[1].to_csv(root + dlb[0] + ".csv")

