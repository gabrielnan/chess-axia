# This program downloads the chess game zip file, loads into pandas, and does some light cleaning
# To download the file from Google Drive, it uses the gdown library.  If you don't want to pip install it, 
# just download the zip file into the working directory beforehand
# This requires 750 MB of disk space for zip file
# See the following URL for a description of what the columns mean
# https://chess-research-project.readthedocs.io/en/latest/#description-of-the-simplified-format

import pandas as pd
from os.path import exists
import numpy as np

# Number of rows to read
# Set to None to read all rows
numRowsRead = 100

# URL of chess records file
zipfileUrl = 'https://drive.google.com/uc?export=download&confirm=BYO0&id=0Bw0y3jV73lx_aXE3RnhmeE5Rb1E'

# Name of zip file
zipName = 'all_with_filtered_anotations.zip'

# Downloads zip file.  750 MB!
if not exists(zipName):
    import gdown
    gdown.download(zipfileUrl, zipName, quiet=False)

# Read column names
columns = list(pd.read_csv(zipName,compression='zip', sep=' ', skiprows=4, nrows=1))
columns = list(map(lambda x: x.split('.')[1], columns[1:-1]))

# sets up function to clean up booleans in data
booleanColumns = columns[6:]
booleanConvert = lambda x: True if 'true' in x else False if 'false' in x else np.nan

# Read metadata for each game
metadata = pd.read_csv(zipName, compression='zip', sep=' ', skiprows=5, names=columns, index_col=False, nrows=numRowsRead, comment='#', converters={i : booleanConvert for i in booleanColumns})

# read moves for each game
moves = pd.read_csv(zipName, compression='zip', sep='###', skiprows=5, names=['moves'], nrows=numRowsRead).reset_index()
moves['t'] = moves['index'].apply(lambda x: x.split(' ')[0])
moves.drop(['index'], axis=1, inplace=True)

# Join metadata and moves
games = metadata.join(moves, sort=True, rsuffix='r')
games.drop('tr',axis=1, inplace=True)
del metadata
del moves

# Some cleaning still needs to be done, like throwing out games with corrupt data, or games that start at abnormal states (Fischer Random Chess)

print(games.info(),'\n\n',games.head())