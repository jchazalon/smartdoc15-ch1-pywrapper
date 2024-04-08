from smartdoc15_ch1 import Dataset, evaluate_segmentation


d = Dataset(data_home="./docker-automation/data/competitions/2015-ICDAR-smartdoc/challenge1/99-computable-version-2017-test",
           download_if_missing=True)
print(len(d))