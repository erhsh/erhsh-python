def create_net():
    import mindspore.common.dtype as mstype
    from mindspore import nn
    from mindspore.common.initializer import initializer

    def _make_layer(base, batch_norm):
        layers = []
        in_channels = 3
        for v in base:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                weight_shape = (v, in_channels, 3, 3)
                weight = initializer('XavierUniform', shape=weight_shape, dtype=mstype.float32)
                conv2d = nn.Conv2d(in_channels=in_channels,
                                   out_channels=v,
                                   kernel_size=3,
                                   padding=0,
                                   pad_mode="same",
                                   has_bias=False,
                                   weight_init=weight)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        return nn.SequentialCell(layers)

    class Vgg(nn.Cell):
        def __init__(self, base, num_classes=1000, batch_norm=False, has_dropout=True, phase="train",
                     include_top=True):
            super(Vgg, self).__init__()
            self.layers = _make_layer(base, batch_norm=batch_norm)
            self.include_top = include_top
            self.flatten = nn.Flatten()
            dropout_ratio = 0.5
            if not has_dropout or phase == "test":
                dropout_ratio = 1.0
            self.classifier = nn.SequentialCell([
                nn.Dense(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Dense(4096, 4096),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Dense(4096, num_classes),
            ])

        def construct(self, x):
            x = self.layers(x)
            if self.include_top:
                x = self.flatten(x)
                x = self.classifier(x)
            return x

    cfg = {
        '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    return Vgg(cfg['16'], num_classes=10)


if __name__ == '__main__':
    net = create_net()
    print(net)
