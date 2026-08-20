import os
import shutil
import random

classes = ['Straight', 'Wavy', 'curly', 'kinky', 'dreadlocks']
src = 'dataset/hairtexture'
test_dir = 'test/hairtexture_test'

for cls in classes:
    src_cls = os.path.join(src, cls)
    test_cls = os.path.join(test_dir, cls)
    os.makedirs(test_cls, exist_ok=True)
    
    images = os.listdir(src_cls)
    random.shuffle(images)
    test_images = images[:int(0.2 * len(images))]
    
    for img in test_images:
        shutil.move(os.path.join(src_cls, img),
                   os.path.join(test_cls, img))

print("Done! Test data created!")