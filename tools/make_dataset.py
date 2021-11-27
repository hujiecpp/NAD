import shutil
import os

## For val dataset:
## ori: val/*.jpg -> dst: val/classname/*.jpg

## For train dataset:
## ori: train/a-z/classname/*.jpg -> dst: train/classname/*.jpg

txt_file = open('./val.txt', 'r')
try:
    for line in txt_file:
        print(line)
        name = line.split('/')
        print(name)
        class_dir = os.path.join(name[0], name[1])
        print(class_dir)
        os.makedirs(class_dir, exist_ok = True)
        img_path = os.path.join(name[0], name[2].rstrip('\n'))
        if os.path.exists(img_path):
            shutil.move(img_path, class_dir + '/')
        # break
finally:
    txt_file.close()


## For the train dir, you can remove the file-folds by:
## mv -r ./train/a/* ./train/
## mv -r ./train/b/* ./train/
## ...

## or by: (may take a lot of minutes)
# txt_file = open('./train.txt', 'r')
# try:
#     for line in txt_file:
#         print(line)
#         name = line.split('/')
#         print(name)
#         class_dir = os.path.join(name[0], name[1])
#         print(class_dir)
#         os.makedirs(class_dir, exist_ok = True)
#         print(name[1][0])
#         img_path = os.path.join(name[0], name[1][0], name[1], name[2].rstrip('\n'))
#         print(img_path)
#         if os.path.exists(img_path):
#             shutil.move(img_path, class_dir + '/')
#         # break
# finally:
#     txt_file.close()