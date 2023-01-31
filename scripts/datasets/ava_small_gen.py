import os
import pandas as pd

def main():
    ava_info_folder = os.environ['AVA_info_folder']
    info_df = pd.read_csv(f'{ava_info_folder}/info.csv', index_col=0)

    reduced_df = info_df.sample(frac=1, random_state=1).head(10000)
    reduced_df.to_csv('info.csv')

if __name__ == "__main__":
    main()