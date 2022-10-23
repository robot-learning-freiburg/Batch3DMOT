import yaml
import argparse
import os
import sys


class ParamNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def overwrite(self, args: argparse.Namespace):
        for k, v in vars(args).items():
            if k in self.__dict__.keys() and v is not None:
                self.__dict__[k] = v


class ParamLib:
    def __init__(self, config_path: str):
        self.config_path = config_path

        # Create all parameter dictionaries
        self.main = ParamNamespace()
        self.paths = ParamNamespace()
        self.resnet = ParamNamespace()
        self.pointnet = ParamNamespace()
        self.radarnet = ParamNamespace()
        self.gnn = ParamNamespace()
        self.preprocessing = ParamNamespace()
        self.graph_construction = ParamNamespace()
        self.detections = ParamNamespace()
        self.predict = ParamNamespace()
        self.classes = ParamNamespace()
        self.eval = ParamNamespace()

        # Load config file with parametrization, create paths and do sys.path.inserts
        self.load_config_file(self.config_path)
        #self.create_dir_structure()
        self.add_system_paths()

    def load_config_file(self, path: str):
        """
        Loads a config YAML file and sets the different dictionaries.
        Args:
            path: path to some configuration file in yaml format

        Returns:
        """

        with open(path, 'r') as stream:
            try:
                config_file = yaml.safe_load(stream)
            except yaml.YAMLError as exception:
                print(exception)

        # Copy yaml content to the different dictionaries.
        vars(self.main).update(config_file['main'])
        vars(self.paths).update(config_file['paths'])
        vars(self.resnet).update(config_file['resnet'])
        vars(self.pointnet).update(config_file['pointnet'])
        vars(self.radarnet).update(config_file['radarnet'])
        vars(self.gnn).update(config_file['gnn'])
        vars(self.preprocessing).update(config_file['preprocessing'])
        vars(self.graph_construction).update(config_file['graph_construction'])
        vars(self.detections).update(config_file['detections'])
        vars(self.predict).update(config_file['predict'])
        vars(self.classes).update(config_file['classes'])
        vars(self.eval).update(config_file['eval'])

        # Set some secondary paths that are important
        if self.main.dataset == "nuscenes":

            # paths to preprocessed data
            self.paths.preprocessed_data = os.path.join(self.paths.tmp, self.main.dataset, 'preprocessed/')
            self.paths.preprocessed_data_img = os.path.join(self.paths.tmp, self.main.dataset, 'preprocessed/img/')
            self.paths.preprocessed_data_lidar = os.path.join(self.paths.tmp, self.main.dataset, 'preprocessed/lidar/')
            self.paths.preprocessed_data_radar = os.path.join(self.paths.tmp, self.main.dataset, 'preprocessed/radar/')

            # processed annotations
            self.paths.scene_meta = os.path.join(self.paths.tmp, self.main.dataset, 'scene_meta.json')
            self.paths.image_anns = os.path.join(self.paths.data, self.main.version, 'image_annotations.json')
            self.paths.processed_img_anns = os.path.join(self.paths.tmp, self.main.dataset, 'processed_img_anns.json')
            self.paths.processed_lidar_anns = os.path.join(self.paths.tmp, self.main.dataset, 'processed_lidar_anns.json')
            self.paths.processed_radar_anns = os.path.join(self.paths.tmp, self.main.dataset, 'processed_radar_anns.json')

            # paths to processed graph data
            self.paths.graphs = os.path.join(self.paths.tmp, self.main.dataset, 'graphs/')
            self.paths.graphs_pose_megvii_disj_len5 = os.path.join(self.paths.graphs, 'pose_megvii_disj_len5/')
            self.paths.graphs_pose_centerpoint_disj_len5 = os.path.join(self.paths.graphs, 'pose_centerpoint_disj_len5/')
            self.paths.graphs_clr_megvii_disj_len5 = os.path.join(self.paths.graphs, 'clr_megvii_disj_len5/')
            self.paths.graphs_clr_centerpoint_disj_len5 = os.path.join(self.paths.graphs, 'clr_centerpoint_disj_len5/')

            # additional paths for training, eval, etc.
            self.paths.eval = os.path.join(self.paths.tmp, self.main.dataset, 'eval/')
            self.paths.models = os.path.join(self.paths.top_level, 'models/')
            self.paths.detections = os.path.join(self.paths.tmp, self.main.dataset, 'detections/')

        else:
            raise NotImplementedError

    def create_dir_structure(self):
        """
        Loops through the paths dictionary in order to create
        the paths if they do not exist.
        Args:
            paths_dict: some para

        Returns:
            -
        """
        for name, path in vars(self.paths).items():
            # exclude all paths to files
            if len(path.split('.')) == 1:
                if not os.path.exists(path):
                    os.makedirs(path)

    def add_system_paths(self):
        """
        Loops through the paths dictionary in order to create
        the paths if they do not exist.
        Args:
            paths_dict: some para

        Returns:
            -
        """
        sys.path.insert(0, self.paths.package)
        sys.path.insert(0, os.path.join(self.paths.package, 'utils'))
        sys.path.insert(0, os.path.join(self.paths.package, 'eval'))
        sys.path.insert(0, os.path.join(self.paths.package, 'models'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load config.yaml to batch_3dmot project',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', type=str, help='Provide path to config file.')
    parser.add_argument('--num_epochs', type=int, help='Provide path to config file.')
    opt = parser.parse_args()

    params = ParamLib(opt.file)
    params.gnn.overwrite(opt)

    print(params.classes.nuscenes_tracking_eval)