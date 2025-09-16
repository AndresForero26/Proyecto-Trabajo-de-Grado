import json
import os
from sklearn.model_selection import train_test_split
from shutil import copy2
from pathlib import Path

def split_folder_by_class(src_root, dst_root, test_size=0.15, val_size=0.15, seed=42):
    """
    Asume estructura:
    src_root/
      Anthracnose/
      FruitFly/
      Healthy/
    Crea:
    dst_root/
      train/{clases}
      val/{clases}
      test/{clases}
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    for part in ['train','val','test']:
        for cls in [d.name for d in src_root.iterdir() if d.is_dir()]:
            (dst_root/part/cls).mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in src_root.iterdir() if d.is_dir()]
    for cls in classes:
        files = [p for p in (src_root/cls).glob('*') if p.is_file()]
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=seed)
        train_files, val_files  = train_test_split(train_files, test_size=val_size/(1-test_size), random_state=seed)

        for p in train_files:
            copy2(p, dst_root/'train'/cls/p.name)
        for p in val_files:
            copy2(p, dst_root/'val'/cls/p.name)
        for p in test_files:
            copy2(p, dst_root/'test'/cls/p.name)

    # guardar class_indices
    mapping = {cls: i for i, cls in enumerate(sorted(classes))}
    with open(dst_root/'class_indices.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    return classes
