

"""
    For diagram of Nets: http://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf

    [filter size, stride, padding]
    Assume the two dimensions are the same
    Each kernel requires the following parameters:
     - k_i: kernel size
     - s_i: stride
     - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow) 
"""


bottleneck = [[1, 1, 0], [3, 1, 1], [1, 1, 0]]
bottleneck_downsample = [[1, 1, 0], [3, 2, 1], [1, 1, 0]]


mobile_net_v2 = [
    [3, 2, 1],

    *bottleneck,

    *bottleneck_downsample,
    *bottleneck,

    *bottleneck_downsample,
    *bottleneck * 2,

    *bottleneck_downsample,
    *bottleneck * 3,

    *bottleneck * 3,

    *bottleneck_downsample,
    *bottleneck * 2,

    *bottleneck,

    [1, 1, 0],

    [7, 1, 0]
]


mobile_net_v2_names = [
    'conv2d',
    *[f'bottleneck_1_1_{j}' for j in range(1, 4)],
    *[f'bottleneck_2_{i}_{j}' for i in range(1, 3) for j in range(1, 4)],
    *[f'bottleneck_3_{i}_{j}' for i in range(1, 4) for j in range(1, 4)],
    *[f'bottleneck_4_{i}_{j}' for i in range(1, 5) for j in range(1, 4)],
    *[f'bottleneck_5_{i}_{j}' for i in range(1, 4) for j in range(1, 4)],
    *[f'bottleneck_6_{i}_{j}' for i in range(1, 4) for j in range(1, 4)],
    *[f'bottleneck_7_1_{j}' for j in range(1, 4)],
    'conv2d 1x1',
    'avg_pool'
]



def get_model(name):
    if 'v2' in name:
        return mobile_net_v2, mobile_net_v2_names


if __name__ == '__main__':
    assert len(mobile_net_v2) == len(mobile_net_v2_names), (
        f'len_model={len(mobile_net_v2)}, len_names={len(mobile_net_v2_names)}'
    )
