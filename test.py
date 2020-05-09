import kaolin as kal
from kaolin.datasets import S3DIS_Points

S3DIS_ROOT = '/scratch/pushkalkatara/data/Stanford3dDataset_v1.2_Aligned_Version'
CACHE_DIR = '/scratch/pushkalkatara/tests/datasets/cache'

points = S3DIS_Points(
        root=S3DIS_ROOT,
        cache_dir=CACHE_DIR
)
