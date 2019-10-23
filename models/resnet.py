

"""
    For diagram of Nets: https://arxiv.org/pdf/1512.03385.pdf

    [filter size, stride, padding]
    Assume the two dimensions are the same
    Each kernel requires the following parameters:
     - k_i: kernel size
     - s_i: stride
     - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow) 
"""


basic_block = [[3, 1, 1], [3, 1, 1]]
basic_block_downsample = [[3, 2, 1], [3, 1, 1]]
bottleneck = [[1, 1, 0], [3, 1, 1], [1, 1, 0]]
bottleneck_downsample = [[1, 2, 0], [3, 1, 1], [1, 1, 0]]


resnet_50 = [
    [7, 2, 3],
    [3, 2, 1],
    *bottleneck * 3,

    *bottleneck_downsample,
    *bottleneck * 3,

    *bottleneck_downsample,
    *bottleneck * 5,

    *bottleneck_downsample,
    *bottleneck * 2,

    [7, 1, 0]
]


resnet_50_names = [
    'conv1',
    'max_pool',
    *[f'conv2_{i}_{j}' for i in range(1, 4) for j in range(1, 4)],
    *[f'conv3_{i}_{j}' for i in range(1, 5) for j in range(1, 4)],
    *[f'conv4_{i}_{j}' for i in range(1, 7) for j in range(1, 4)],
    *[f'conv5_{i}_{j}' for i in range(1, 4) for j in range(1, 4)],
    'avg_pool'
]


resnet_18 = [
    [7, 2, 3],
    [3, 2, 1],
    *basic_block * 2,
    *basic_block_downsample,
    *basic_block,
    *basic_block_downsample,
    *basic_block,
    *basic_block_downsample,
    *basic_block,
    [7, 1, 0]
]

resnet_18_names = [
    'conv1',
    'max_pool',
    *[f'conv2_{i}_{j}' for i in range(1, 3) for j in range(1, 3)],
    *[f'conv3_{i}_{j}' for i in range(1, 3) for j in range(1, 3)],
    *[f'conv4_{i}_{j}' for i in range(1, 3) for j in range(1, 3)],
    *[f'conv5_{i}_{j}' for i in range(1, 3) for j in range(1, 3)],
    'avg_pool'
]


resnet_34 = [
    [7, 2, 3],
    [3, 2, 1],
    *basic_block * 3,
    *basic_block_downsample,
    *basic_block * 3,
    *basic_block_downsample,
    *basic_block * 5,
    *basic_block_downsample,
    *basic_block * 2,
    [7, 1, 0]
]

resnet_34_names = [
    'conv1',
    'max_pool',
    *[f'conv2_{i}_{j}' for i in range(1, 4) for j in range(1, 3)],
    *[f'conv3_{i}_{j}' for i in range(1, 5) for j in range(1, 3)],
    *[f'conv4_{i}_{j}' for i in range(1, 7) for j in range(1, 3)],
    *[f'conv5_{i}_{j}' for i in range(1, 4) for j in range(1, 3)],
    'avg_pool'
]


def get_model(name):

    if '18' in name:
        return resnet_18, resnet_18_names
    elif '34' in name:
        return resnet_34, resnet_34_names
    elif '50' in name:
        return resnet_50, resnet_50_names
    else:
        raise Exception('Unknown model type')


if __name__ == '__main__':
    assert len(resnet_34) == len(resnet_34_names), (
        f'len_model={len(resnet_34)}, len_names={len(resnet_34_names)}'
    )
    assert len(resnet_18) == len(resnet_18_names), (
        f'len_model={len(resnet_18)}, len_names={len(resnet_18_names)}'
    )
    assert len(resnet_50) == len(resnet_50_names), (
        f'len_model={len(resnet_50)}, len_names={len(resnet_50_names)}'
    )
