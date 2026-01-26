from collections import Counter
from dataset.isic2019_dataset import ISIC2019Dataset

ds = ISIC2019Dataset(client_id=0, train=True)
print(Counter(ds.labels))
