from skopt.space import Integer, Space
# from skopt.samplter import Lhs
def StripFreeParams(point):
    fixed_params = []
    for low, high in FIXED_RANGES:
        fixed_params += point[low:high]
    return fixed_params

def StripFixedParams(point):
    stripped_point = []
    pos = 0
    for low, high in FIXED_RANGES:
        stripped_point += point[:low-pos]
        point = point[high-pos:]
        pos = high
    _, high = FIXED_RANGES[-1]
    stripped_point += point[high-pos:]
    return stripped_point

def CreateDiscreteSpace():
    dZgap = 10
    zGap = dZgap / 2  # halflengh of gap
    dimensions = 8 * [
        Integer(170 + zGap, 300 + zGap)  # magnet lengths
        ] + 8 * (
            2 * [
                Integer(10, 100)  # dXIn, dXOut
            ] + 2 * [
                Integer(20, 200)  # dYIn, dYOut
            ] + 2 * [
                Integer(2, 70)  # gapIn, gapOut
            ])
    return Space(StripFixedParams(dimensions))

def CreateReducedSpace(minimum, variation=0.1):
    dZgap = 10
    zGap = dZgap / 2  # halflengh of gap
    dimensions = 8 * [
        Integer(170 + zGap, 300 + zGap)  # magnet lengths
        ] + 8 * (
            2 * [
                Integer(10, 100)  # dXIn, dXOut
            ] + 2 * [
                Integer(20, 200)  # dYIn, dYOut
            ] + 2 * [
                Integer(2, 70)  # gapIn, gapOut
            ])
    reduced_dims = [Integer(max(int((1-variation)*m), dim.bounds[0]), min(int((1+variation)*m), dim.bounds[1]))
                    for m, dim in zip(minimum, dimensions)]
    return Space(StripFixedParams(reduced_dims))

DEFAULT_POINT = [
    70.0,
    170.0,
    205.,
    205.,
    280.,
    245.,
    305.,
    240.,
    40.0,
    40.0,
    150.0,
    150.0,
    2.0,
    2.0,
    80.0,
    80.0,
    150.0,
    150.0,
    2.0,
    2.0,
    87.,
    65.,
    35.,
    121,
    11.,
    2.,
    65.,
    43.,
    121.,
    207.,
    11.,
    2.,
    6.,
    33.,
    32.,
    13.,
    70.,
    11.,
    5.,
    16.,
    112.,
    5.,
    4.,
    2.,
    15.,
    34.,
    235.,
    32.,
    5.,
    8.,
    31.,
    90.,
    186.,
    310.,
    2.,
    55.,
]
FIXED_RANGES = [(0, 2), (8, 20)]
FIXED_PARAMS = StripFreeParams(DEFAULT_POINT)