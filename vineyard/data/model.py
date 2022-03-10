import os


class DataSet:
    """
    Simple data.py structure to store data.py information used during the training.
    """

    def __init__(self, samples_train_folder,
                 labels_train_folder,
                 samples_test_folder,
                 labels_test_folder,
                 sample_size, label_size,
                 image_depth, color_mode,
                 samples_eval_folder=None,
                 labels_eval_folder=None):
        self.samples_train_folder = samples_train_folder
        self.labels_train_folder = labels_train_folder
        self.samples_test_folder = samples_test_folder
        self.labels_test_folder = labels_test_folder
        self.samples_eval_folder = samples_eval_folder
        self.labels_eval_folder = labels_eval_folder

        self.sample_size = sample_size
        self.label_size = label_size
        self.image_depth = image_depth
        self.color_mode = color_mode

        self.channelwise_centering = False
        self.channel_means = 0
        self.channelwise_std_normalization = False
        self.channel_stds = 0

        self._calc_metrics()

    def _calc_metrics(self):
        # number of images
        self.num_train_images = self._count_imgs_in_folder(self.samples_train_folder)
        self.num_test_images = self._count_imgs_in_folder(self.samples_test_folder)
        # calc mean image values

    def _count_imgs_in_folder(self, folder_path):
        return len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])

    def __str__(self):
        return "{}-{}".format(self.sample_size[0], self.label_size[0])


class DatasetDef():
    """
    Dataset metadata and metrics obtained from data.py images
    """

    def __init__(self, patch_size, base_folder, patch_is_gt=True, gt_downscale=1, stride=None,
                 max_img_folder=None):
        # if the original imagen needs to be scaled before it is used as ground truth
        # ej if the base image has a 0.25m/px GSD and we need a 1m/px GSD, the scale has to be 4
        self.gt_downscale = gt_downscale
        # patch side measured in px
        self.patch_size = patch_size
        self.max_img_folder = max_img_folder
        self.base_folder = base_folder
        # window slide stripe during patch extrating process
        self.stride = stride // 3 if not stride else stride

        self.n_train_images = 0
        self.n_test_images = 0

        self.channelwise_centering = False
        self.channel_means = 0
        self.channelwise_std_normalization = False
        self.channel_stds = 0

    def get_train_folder(self):
        return os.path.join(self.base_folder, "train")

    def get_test_folder(self):
        return os.path.join(self.base_folder, "test")

    def get_eval_folder(self):
        return os.path.join(self.base_folder, "eval")

    def get_samples_folder(self, type, size):
        if type != "train" and type != "test" and type != "eval":
            raise ValueError("Invalid data.py folder: " + type)
        return os.path.join(self.base_folder, type, "lr_{}".format(size))

    def get_labels_folder(self, type):
        return os.path.join(self.base_folder, "train", "lr_{}".format(self.patch_size))

    def get_total_images(self):
        return self.n_test_images + self.n_train_images

    # def _calc(self):
    #     dts_def.n_train_images =
    #     dts_def.n_test_images =

    def save(self, file_path=None):
        """
        Save data.py definition to base_folder
        :return:
        """
        import jsonpickle
        json = jsonpickle.encode(self)
        if file_path is None:
            file_path = os.path.join(self.base_folder, "data.py.json")
        # save json content
        fo = open(file_path, "w")
        fo.write(json)
        fo.close()

    @staticmethod
    def read(file_path):
        """
        :param file_path:
        :return: data.py
        :rtype: DatasetDef
        """
        import jsonpickle

        json = None
        with open(file_path, 'r') as mf:
            json = mf.read().replace('\n', '')

        m = jsonpickle.decode(json)

        return m


def get_dts(config_file, sample_size, label_size, image_depth=3, color_mode="rgb", channelwise_centering=False,
            channelwise_std_normalization=False):
    tsample_size = (sample_size, sample_size)
    tlabel_size = (label_size, label_size)
    # read data.py information
    dts_meta = DatasetDef.read(config_file)

    # locate folders wrt patches sizes:
    samples_train_folder = os.path.join(dts_meta.base_folder, "train", "lr_{}".format(sample_size))
    labels_train_folder = os.path.join(dts_meta.base_folder, "train", "lr_{}".format(label_size))
    samples_test_folder = os.path.join(dts_meta.base_folder, "test", "lr_{}".format(sample_size))
    labels_test_folder = os.path.join(dts_meta.base_folder, "test", "lr_{}".format(label_size))
    samples_eval_folder = os.path.join(dts_meta.base_folder, "eval", "lr_{}".format(sample_size))
    labels_eval_folder = os.path.join(dts_meta.base_folder, "eval", "lr_{}".format(label_size))

    dts = DataSet(samples_train_folder,
                  labels_train_folder,
                  samples_test_folder,
                  labels_test_folder,
                  tsample_size, tlabel_size,
                  samples_eval_folder=samples_eval_folder,
                  labels_eval_folder=labels_eval_folder,
                  image_depth=image_depth,
                  color_mode=color_mode)
    dts.channelwise_centering = channelwise_centering
    if not channelwise_centering:
        dts.channel_means = 0
    else:
        dts.channel_means = dts_meta.channel_means

    dts.channelwise_std_normalization = channelwise_std_normalization
    if not channelwise_std_normalization:
        dts.channel_stds = 0
    else:
        dts.channel_stds = dts_meta.channel_stds

    return dts
