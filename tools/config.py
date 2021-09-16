
imagenet_dir = "/home/tongtong/dataset/imagenet"
place_dir = "/home/tongtong/dataset/places365_standard"

def get_dataset_dir(name):
    if name == "imagenet":
        return imagenet_dir
    elif name == "place":
        return place_dir