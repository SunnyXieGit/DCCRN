#返回包含网络参数的字典。参数包括编码器、LSTM层、全连接层和解码器的通道数、卷积核大小、步长、填充等。
def get_net_params():
    params = {}
    encoder_dim_start = 32
    params["encoder_channels"] = [
        1,
        encoder_dim_start,
        encoder_dim_start * 2,
        encoder_dim_start * 4,
        encoder_dim_start * 8,
        encoder_dim_start * 8,
        encoder_dim_start * 8
    ]
    #每层卷积核的尺寸。
    params["encoder_kernel_sizes"] = [
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2)
    ]
    #卷积操作的步长 。
    params["encoder_strides"] = [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1)
    ]
    #卷积操作的填充。
    params["encoder_paddings"] = [
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0)
    ]
    # ----------lstm---------
    params["lstm_dim"] = [
        1280, 128
    ]
    params["dense"] = [
        128, 1280
    ]
    params["lstm_layer_num"] = 2
    # --------decoder--------
    params["decoder_channels"] = [
        0,
        encoder_dim_start * 8,
        encoder_dim_start * 8,
        encoder_dim_start * 4,
        encoder_dim_start * 2,
        encoder_dim_start,
        1
    ]
    params["decoder_kernel_sizes"] = [
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2)
    ]
    params["decoder_strides"] = [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1)
    ]
    params["decoder_paddings"] = [
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0)
    ]
    params["encoder_chw"] = [
        (32, 129, 6),
        (64, 65, 5),
        (128, 33, 4),
        (256, 17, 3),
        (256, 9, 2),
        (256, 5, 1)
    ]
    params["decoder_chw"] = [
        (256, 9, 2),
        (256, 17, 3),
        (128, 33, 4),
        (64, 65, 5),
        (32, 129, 6),
        (1, 257, 7)
    ]
    return params
