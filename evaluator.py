

def compute_next_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1


def eval_conv(in_size, in_chnl, out_chnl, kernel_size, stride=1, padding=0):
    out_size = compute_next_size(in_size, kernel_size, stride, padding)
    cal_per_elem = 2 * kernel_size * kernel_size * in_chnl
    ret = cal_per_elem * out_size * out_size * out_chnl
    return ret, out_size


def eval_fc(in_features, out_features):
    return 2 * in_features * out_features


def eval_big_net():
    params = [0, 28]

    def block(in_chnl, out_chnl):
        cal, size = eval_conv(params[1], in_chnl, out_chnl, 3)
        size = compute_next_size(size, 2, 2)
        params[0] += cal
        params[1] = size
    block(1, 32)
    block(32, 64)
    block(64, 128)

    params[0] += eval_fc(128, 625)
    params[0] += eval_fc(625, 10)
    print('big net FLOP: %d' % params[0])
    return params[0]


def eval_small_net():
    # res = 0
    res, _ = eval_conv(28, 1, 16, 5, 2)
    # max pooling
    res += eval_fc(6 * 6 * 16, 128)
    res += eval_fc(128, 10)
    res += eval_fc(128, 128)
    res += eval_fc(128, 1)
    print('small net FLOP: %d' % res)
    return res


eval_big_net()
eval_small_net()


