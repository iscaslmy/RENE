import yaml
import torch


config_file = open('config.yml', 'r', encoding='utf-8')
params = yaml.load(config_file, Loader=yaml.FullLoader)
print(params.get('pretraining', 0))


str = '您好。我是中国人。'
print(' '.join(str))

list = [[1], [1, 3, 5, 6, 7], [1, 3]]
print(sorted(list, key=(lambda x: len(x)), reverse=True))

pretrained_dict = torch.load('model/lm_model.pth')
print(pretrained_dict)
