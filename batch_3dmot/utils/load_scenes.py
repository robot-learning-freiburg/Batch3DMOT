import nuscenes
from nuscenes.utils.splits import create_splits_scenes


def load_scene_meta_list(data_path: str, dataset: str = 'nuscenes', version: str = "v1.0-trainval"):
    print("nuscenes split:", version)
    assert dataset in ["nuscenes"], "Error, please pass a valid dataset name"
    assert version in ["v1.0-mini", "v1.0-trainval", "v1.0-test"], "Error: The given split description is not configured."

    # Get splits & all available scenes
    split = create_splits_scenes(verbose=False)

    try:
        if dataset == "nuscenes":
            if version == "v1.0-mini":
                
                # Load nuscenes instance
                nusc = nuscenes.nuscenes.NuScenes(version=version, dataroot=data_path, verbose=True)
                all_scenes = nusc.scene

                # Define the mini-train/val split and then select metadata based on scene name accordingly
                mini_train = split['mini_train']
                mini_val = split['mini_val']

                mini_train_scene_meta_list = [x for x in all_scenes if x['name'] in mini_train]
                mini_val_scene_meta_list = [x for x in all_scenes if x['name'] in mini_val]

                return nusc, [mini_train_scene_meta_list, mini_val_scene_meta_list]

            elif version == "v1.0-trainval":

                # Load nuscenes instance
                nusc = nuscenes.nuscenes.NuScenes(version=version, dataroot=data_path, verbose=True)
                all_scenes = nusc.scene

                # Define the train/val split and then select metadata based on scene name accordingly
                train = split['train']
                val = split['val']
                train_scene_meta_list = [x for x in all_scenes if x['name'] in train]
                val_scene_meta_list = [x for x in all_scenes if x['name'] in val]

                return nusc, [train_scene_meta_list, val_scene_meta_list]
            
            elif version == "v1.0-test":

                # Load nuscenes instance
                nusc = nuscenes.nuscenes.NuScenes(version=version, dataroot=data_path, verbose=True)
                all_scenes = nusc.scene
                test = split['test']
                
                # Define the train/val split and then select metadata based on scene name accordingly
                test_scene_meta_list = [x for x in all_scenes if x['name'] in test]

                return nusc, [test_scene_meta_list]
        else:
            raise NotImplementedError

    except Exception as exc:
        raise exc
