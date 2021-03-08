import torch
from torchvision import transforms
import numpy as np

from data.ModelNet40Loader import ModelNet40Cls
import data.data_utils as d_utils
from models.pointnet2_ssg_sem import PointNet2SemSegSSG

if __name__ == '__main__':
    
    # device = torch.device('cuda:0')
    torch.cuda.set_device(1)
    num_points_per_shape = 1024

    # build dataloaders
    transforms_with_aug = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    ) # it performs random rotate, scale, shift, jitter (add random noise)
    transform = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
        ]
    )
    trainset = ModelNet40Cls(num_points_per_shape, train=True, transforms=transforms_with_aug)
    testset = ModelNet40Cls(num_points_per_shape, train=False, transforms=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # build model
    param = {}
    param["model.use_xyz"] = True
    param['in_fea_dim'] = 0
    param['out_dim'] = 3
    model = PointNet2SemSegSSG(param)
    # model.to(device)
    model.cuda()

    # train the model
    model.train()
    for i, (data, label) in enumerate(trainloader):
        print('Trainset [%d/%d] %.3f' % (i, len(trainloader), i/len(trainloader)))
        data = data[:,:,0:3].cuda() #to(device)
        print('batch data shape:', data.shape)
        out = model(data)
        print('batch out shape:', out.shape)
        print('batch label shape:', label.shape)

    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(testloader):
            print('Testset [%d/%d] %.3f' % (i, len(testloader), i/len(testloader)))
            data = data[:,:,0:3].cuda() #to(device)
            print('batch data shape:', data.shape)
            out = model(data)
            print('batch out shape:', out.shape)
            print('batch label shape:', label.shape)