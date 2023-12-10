import numpy


def lr_cityblock_cluster_cardinality(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    1.228800e-01,
                    1.227127e-01,
                    7.808845e-02,
                    4.154137e-02,
                    5.657729e-02,
                    3.525646e-02,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_cityblock_cluster_cardinality(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 1.391490e-02:
        if lfd_ema <= 6.940212e-01:
            if radius <= 1.341821e-02:
                return 5.454838e-01
            else:
                return 3.113860e-01
        else:
            if cardinality <= 2.564081e-02:
                return 3.508578e-01
            else:
                return 7.560313e-01
    else:
        if cardinality_ema <= 3.965033e-01:
            if cardinality <= 1.411780e-01:
                return 8.795414e-01
            else:
                return 9.287172e-01
        else:
            if cardinality_ema <= 7.348774e-01:
                return 9.599685e-01
            else:
                return 9.889817e-01


def lr_cityblock_component_cardinality(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    1.429870e-02,
                    -1.323484e-02,
                    -1.150261e-02,
                    3.896381e-02,
                    4.082664e-02,
                    -1.364604e-02,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_cityblock_component_cardinality(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 5.148810e-03:
        return 7.500000e-01
    else:
        if cardinality_ema <= 6.438965e-01:
            if lfd_ema <= 9.120842e-01:
                return 9.709770e-01
            else:
                return 9.303896e-01
        else:
            if lfd_ema <= 8.965069e-01:
                return 9.982470e-01
            else:
                return 9.858773e-01


def lr_cityblock_graph_neighborhood(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    8.262898e-03,
                    1.537685e-02,
                    9.422306e-03,
                    3.740549e-02,
                    3.891843e-02,
                    -1.250707e-02,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_cityblock_graph_neighborhood(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 5.148810e-03:
        if cardinality <= 2.540494e-02:
            return 4.437313e-01
        else:
            return 7.500000e-01
    else:
        if cardinality_ema <= 6.354841e-01:
            if cardinality_ema <= 6.253555e-01:
                return 9.625763e-01
            else:
                return 8.007593e-01
        else:
            if cardinality_ema <= 8.838843e-01:
                return 9.887261e-01
            else:
                return 9.979190e-01


def lr_cityblock_parent_cardinality(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    8.433946e-02,
                    6.050625e-02,
                    5.882554e-02,
                    3.593437e-03,
                    4.128473e-02,
                    -4.736846e-02,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_cityblock_parent_cardinality(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 1.472459e-02:
        if cardinality <= 2.505734e-02:
            if radius <= 1.655350e-04:
                return 3.151504e-01
            else:
                return 6.447252e-01
        else:
            if cardinality <= 1.076636e-01:
                return 7.414646e-01
            else:
                return 9.229898e-01
    else:
        if cardinality <= 5.942344e-01:
            if lfd_ema <= 9.527803e-01:
                return 9.468785e-01
            else:
                return 8.911853e-01
        else:
            if cardinality_ema <= 9.790418e-01:
                return 9.705729e-01
            else:
                return 9.989975e-01


def lr_cityblock_stationary_probabilities(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    4.659433e-02,
                    -5.014006e-02,
                    -6.017402e-02,
                    5.812719e-02,
                    1.466290e-01,
                    1.266893e-03,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_cityblock_stationary_probabilities(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 2.358828e-04:
        if radius_ema <= 1.177303e-02:
            if radius_ema <= 8.709679e-05:
                return 1.400713e-01
            else:
                return 1.698153e-01
        else:
            return 4.245149e-01
    else:
        if cardinality_ema <= 1.807206e-02:
            if radius_ema <= 7.669807e-01:
                return 5.307418e-01
            else:
                return 9.984011e-01
        else:
            if cardinality_ema <= 4.135659e-01:
                return 9.372930e-01
            else:
                return 9.907336e-01


def lr_cityblock_vertex_degree(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    1.032663e-01,
                    1.232432e-01,
                    7.317461e-02,
                    1.084027e-02,
                    9.541312e-02,
                    1.760110e-02,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_cityblock_vertex_degree(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 1.391490e-02:
        if lfd_ema <= 6.932603e-01:
            if cardinality <= 1.076636e-01:
                return 4.198158e-01
            else:
                return 8.463385e-01
        else:
            if cardinality <= 2.564081e-02:
                return 3.269299e-01
            else:
                return 7.921413e-01
    else:
        if cardinality_ema <= 7.950329e-01:
            if cardinality <= 4.997393e-01:
                return 9.251187e-01
            else:
                return 9.570666e-01
        else:
            if radius_ema <= 4.155880e-01:
                return 9.516691e-01
            else:
                return 9.928017e-01


def lr_euclidean_cluster_cardinality(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    1.313924e-01,
                    1.326884e-01,
                    9.136274e-02,
                    2.134787e-02,
                    3.100747e-02,
                    3.298891e-02,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_euclidean_cluster_cardinality(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 1.067198e-02:
        if cardinality <= 1.079019e-01:
            if lfd_ema <= 9.999740e-01:
                return 5.265128e-01
            else:
                return 8.834613e-01
        else:
            if cardinality <= 3.330053e-01:
                return 8.695605e-01
            else:
                return 9.694478e-01
    else:
        if cardinality <= 4.997283e-01:
            if lfd <= 2.929797e-01:
                return 8.696701e-01
            else:
                return 9.258335e-01
        else:
            if cardinality_ema <= 7.698584e-01:
                return 9.665650e-01
            else:
                return 9.917949e-01


def lr_euclidean_component_cardinality(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    5.582536e-02,
                    2.442987e-02,
                    8.037801e-03,
                    1.539072e-02,
                    3.654952e-02,
                    1.429881e-02,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_euclidean_component_cardinality(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius_ema <= 5.111776e-03:
        if radius <= 9.307770e-05:
            return 7.500000e-01
        else:
            return 6.997932e-01
    else:
        if cardinality <= 1.494950e-01:
            if cardinality_ema <= 4.477562e-02:
                return 8.816036e-01
            else:
                return 9.351654e-01
        else:
            if cardinality_ema <= 6.924688e-01:
                return 9.702070e-01
            else:
                return 9.932433e-01


def lr_euclidean_graph_neighborhood(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    9.585250e-02,
                    6.025230e-02,
                    3.753800e-02,
                    -3.118970e-03,
                    7.559676e-02,
                    1.875789e-02,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_euclidean_graph_neighborhood(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius_ema <= 2.359367e-05:
        if lfd <= 3.415343e-01:
            if cardinality <= 4.490425e-02:
                return 1.299296e-01
            else:
                return 1.299296e-01
        else:
            return 4.405503e-01
    else:
        if cardinality_ema <= 1.619149e-02:
            if cardinality_ema <= 3.227095e-03:
                return 5.649648e-01
            else:
                return 7.366586e-01
        else:
            if radius_ema <= 5.773949e-02:
                return 8.322585e-01
            else:
                return 9.700278e-01


def lr_euclidean_parent_cardinality(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    9.211533e-02,
                    7.063269e-02,
                    2.331588e-02,
                    7.911250e-04,
                    1.672235e-02,
                    3.891130e-03,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_euclidean_parent_cardinality(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 1.067198e-02:
        if cardinality <= 2.828062e-02:
            if radius <= 9.307770e-05:
                return 3.440542e-01
            else:
                return 6.220238e-01
        else:
            if lfd <= 2.580366e-01:
                return 5.624770e-01
            else:
                return 8.176447e-01
    else:
        if cardinality <= 3.860220e-02:
            if cardinality <= 3.826937e-02:
                return 8.636209e-01
            else:
                return 5.281209e-01
        else:
            if cardinality <= 4.997400e-01:
                return 9.494346e-01
            else:
                return 9.803474e-01


def lr_euclidean_stationary_probabilities(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    8.131170e-02,
                    1.629200e-02,
                    -4.917042e-02,
                    3.507954e-02,
                    -1.800446e-03,
                    -5.963697e-03,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_euclidean_stationary_probabilities(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius_ema <= 3.118177e-02:
        if lfd <= 8.090920e-01:
            if lfd_ema <= 2.858897e-01:
                return 4.526594e-01
            else:
                return 8.478856e-01
        else:
            if lfd_ema <= 6.024930e-01:
                return 5.471349e-03
            else:
                return 2.958559e-01
    else:
        if cardinality <= 1.494950e-01:
            if radius_ema <= 9.417503e-01:
                return 9.296579e-01
            else:
                return 7.623177e-01
        else:
            if cardinality_ema <= 1.749870e-02:
                return 7.646179e-01
            else:
                return 9.863799e-01


def lr_euclidean_vertex_degree(ratios: numpy.array) -> float:
    return float(
        numpy.dot(
            numpy.asarray(
                a=[
                    1.096380e-01,
                    1.658747e-01,
                    1.045492e-01,
                    -1.207694e-03,
                    6.186889e-02,
                    2.726535e-02,
                ],
                dtype=float,
            ),
            ratios,
        ),
    )


def dt_euclidean_vertex_degree(ratios: numpy.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 1.067198e-02:
        if lfd <= 9.634169e-01:
            if lfd_ema <= 9.999111e-01:
                return 4.713993e-01
            else:
                return 9.783193e-01
        else:
            if radius <= 1.020328e-02:
                return 9.089947e-01
            else:
                return 4.464702e-01
    else:
        if cardinality <= 4.997769e-01:
            if lfd <= 9.999996e-01:
                return 9.335233e-01
            else:
                return 7.793101e-01
        else:
            if radius_ema <= 7.430525e-01:
                return 9.627948e-01
            else:
                return 9.921575e-01
