from torchvision import datasets, transforms

def transformationen(mean, std, groesse=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(groesse, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(groesse * 256 / 224)),
        transforms.CenterCrop(groesse),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf