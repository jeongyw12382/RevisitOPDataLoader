from torch.utils.data import DataLoader
import DataSet
from torchvision import transforms
import numpy as np

image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

def collate_fn(samples, dataset):


    ret = {
        'q_tf_image' : np.stack([np.array(np.expand_dims(sample['q_tf_image'], axis=0)) for sample in samples]), 
        'pos_tf_image' : np.stack([np.array(np.expand_dims(sample['pos_tf_image'], axis=0)) for sample in samples]),
        'neg_tf_image' : np.stack([np.stack(sample['neg_tf_image']) for sample in samples])
    }

    print(
        ret['q_tf_image'].shape, 
        ret['pos_tf_image'].shape,
        ret['neg_tf_image'].shape,
    )

    return ret


class QueryLoader(DataLoader):
    
    def __init__(self, dataset, batch_size=4, collate_fn=collate_fn, transform=image_transform):
        self.data_set = DataSet.QuerySet(dataset, transform)
        super().__init__(self.data_set, batch_size=batch_size, collate_fn=lambda x : collate_fn(x, self.data_set))


if __name__=='__main__':
    d = QueryLoader('rparis6k')