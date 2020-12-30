import os
from shutil import copyfile

inner_path = "C:/Users/zgl/AppData/Local/Packages/Microsoft.Windows.ContentDeliveryManager_cw5n1h2txyewy/LocalState/Assets"
file_list = os.listdir(inner_path)
for item in file_list:
    if not item.endswith('.jpg'):
        src = os.path.join(os.path.abspath(inner_path), item)
        itemjpg = item + '.jpg'
        dst = os.path.join(os.path.abspath(inner_path), itemjpg)
        os.rename(src, dst)
