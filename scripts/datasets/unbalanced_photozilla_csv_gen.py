import os
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
    ava_info_folder = os.environ['AVA_info_folder']
    info_csv = pd.read_csv(f'{ava_info_folder}/info.csv', index_col=0)

    maxvote_counts = info_csv.iloc[:,1:11].idxmax(axis="columns").apply(lambda x: int(x[4:]) - 1).value_counts()
    maxvote_counts_normalized = (maxvote_counts / maxvote_counts.sum()).to_dict()

    relative_paths = []

    # Generate paths relative to the root folder of the images in the dataset
    photozilla_images_folder = os.environ['Photozilla_images_folder']
    for filename in glob.iglob(photozilla_images_folder + '**/**', recursive=True):
        path = os.path.normpath(filename)
        path = path.split(os.sep)
        relative_paths.append(os.sep.join(path[-2:]))

    # Filter out unnecessary paths
    relative_paths = list(filter(lambda path: path.split(os.sep)[0] in labels_dict, relative_paths))

    # Construct dictionary with paths and labels, and create Dataframe from it
    df_dict = {
        'id': relative_paths,
        'label': list(map(lambda path: labels_dict[path.split(os.sep)[0]], relative_paths))
    }

    # Only keep around 50000 images, for testing
    df = pd.DataFrame.from_dict(df_dict)
    df = df.groupby('label', group_keys=True).apply(lambda x : x.sample(int(maxvote_counts_normalized[x.name] * 50000))).droplevel('label')
    df.to_csv('info.csv')

if __name__ == "__main__":
    main()