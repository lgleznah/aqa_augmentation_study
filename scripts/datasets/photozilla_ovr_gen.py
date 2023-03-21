import os
import sys
import glob
import pandas as pd

labels_dict = {
    'aerial': 0,
    'architecture': 1,
    'event': 2,
    'fashion': 3,
    'food': 4,
    'nature': 5,
    'sports': 6,
    'street': 7,
    'wedding': 8,
    'wildlife': 9
}

def main():
    target_images = 10000

    # Load AVA to compute unbalance amount
    ava_info_folder = os.environ['AVA_info_folder']
    ava_df = pd.read_csv(f'{ava_info_folder}/info.csv', index_col=0)

    positive_frac = len(ava_df[ava_df['VotesMean'] > 5.0]) / len(ava_df)
    negative_frac = 1 - positive_frac

    # Load Photozilla, and sample images from the positive and negative classes with the same proportion as in AVA
    photozilla_info_folder = os.environ['Photozilla_info_folder']
    photozilla_df = pd.read_csv(f'{photozilla_info_folder}/info.csv', index_col=0)

    positive_class = labels_dict[sys.argv[1]]
    positive_df= photozilla_df[photozilla_df['label'] == positive_class].sample(int(positive_frac * target_images), random_state=1)
    negative_df = photozilla_df[photozilla_df['label'] != positive_class].sample(int(negative_frac * target_images), random_state=1)

    positive_df = positive_df.assign(label=1)
    negative_df = negative_df.assign(label=0)

    photozilla_ovr_df = pd.concat([positive_df, negative_df]).sample(frac=1, random_state=1)
    photozilla_ovr_df.to_csv('info.csv')

if __name__ == "__main__":
    main()