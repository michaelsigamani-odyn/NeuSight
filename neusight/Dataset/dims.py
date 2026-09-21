import random

def generate_train_bmm(dims_out):
    random.seed(42)
    
    grids = [0,32,64,128,256,512,1024]

    b_grid_idx = 0
    m_grid_idx = 0
    n_grid_idx = 0
    k_grid_idx = 0

    grid_num = len(grids) - 1

    points_per_grid = 15

    points = []

    for b_grid_idx in range(grid_num):
        for m_grid_idx in range(grid_num):
            for n_grid_idx in range(grid_num):
                for k_grid_idx in range(grid_num):
                    for i in range(points_per_grid):
                        B = random.randint(grids[b_grid_idx]+1, grids[b_grid_idx+1]+1)
                        M = random.randint(grids[m_grid_idx]+1, grids[m_grid_idx+1]+1)
                        N = random.randint(grids[n_grid_idx]+1, grids[n_grid_idx+1]+1)
                        K = random.randint(grids[k_grid_idx]+1, grids[k_grid_idx+1]+1)
                        points.append((B,M,N,K))

    with open(dims_out, 'w') as f:
        print(dims_out)
        for dim in points:
            f.write(";".join([str(d) for d in dim]) + "\n")

def generate_train_linear(dims_out):
    random.seed(42)

    m_grids = [0,256,512,1024,2048,4096,8192]
    n_grids = [0,256,512,1024,2048,4096,8192]
    k_grids = [0,256,512,1024,2048,4096,8192,16384,32768,65536]

    m_grid_idx = 0
    n_grid_idx = 0
    k_grid_idx = 0

    m_grid_num = len(m_grids) - 1
    n_grid_num = len(n_grids) - 1
    k_grid_num = len(k_grids) - 1

    points_per_grid = 15

    points = []

    for m_grid_idx in range(m_grid_num):
        for n_grid_idx in range(n_grid_num):
            for k_grid_idx in range(k_grid_num):
                for i in range(points_per_grid):
                    B = 1
                    M = random.randint(m_grids[m_grid_idx]+1, m_grids[m_grid_idx+1]+1)
                    N = random.randint(n_grids[n_grid_idx]+1, n_grids[n_grid_idx+1]+1)
                    K = random.randint(k_grids[k_grid_idx]+1, k_grids[k_grid_idx+1]+1)
                    points.append((B,M,N,K))

    # for backward
    m_grids = [0,256,512,1024,2048,4096,8192]
    n_grids = [8192,16384,32768,65536]
    k_grids = [0,256,512,1024,2048,4096,8192]

    m_grid_idx = 0
    n_grid_idx = 0
    k_grid_idx = 0

    m_grid_num = len(m_grids) - 1
    n_grid_num = len(n_grids) - 1
    k_grid_num = len(k_grids) - 1

    points_per_grid = 15

    for m_grid_idx in range(m_grid_num):
        for n_grid_idx in range(n_grid_num):
            for k_grid_idx in range(k_grid_num):
                for i in range(points_per_grid):
                    B = 1
                    M = random.randint(m_grids[m_grid_idx]+1, m_grids[m_grid_idx+1]+1)
                    N = random.randint(n_grids[n_grid_idx]+1, n_grids[n_grid_idx+1]+1)
                    K = random.randint(k_grids[k_grid_idx]+1, k_grids[k_grid_idx+1]+1)
                    points.append((B,M,N,K))


    with open(dims_out, "w") as f:
        for dim in points:
            f.write(";".join([str(d) for d in dim]) + "\n")

def generate_train_vec(dims_out):
    random.seed(42)

    b_grids = [0,256,512,1024,2048,4096,8192,16384]
    h_grids = [0,256,512,1024,2048,4096]

    b_grid_idx = 0
    h_grid_idx = 0

    b_grid_num = len(b_grids) - 1
    h_grid_num = len(h_grids) - 1

    points_per_grid = 50

    points = []

    for b_grid_idx in range(b_grid_num):
        for h_grid_idx in range(h_grid_num):
            for i in range(points_per_grid):
                b = random.randint(b_grids[b_grid_idx]+1, b_grids[b_grid_idx+1]+1)
                h = random.randint(h_grids[h_grid_idx]+1, h_grids[h_grid_idx+1]+1)
                points.append((b,h))

    with open(dims_out, "w") as f:
        for dim in points:
            f.write(";".join([str(d) for d in dim]) + "\n")

def generate_test_bmm(dims_out):
    random.seed(42)
    
    batch_sizes = [1,2,4,8,16,32,64,128,256,512]
    number_of_heads = [12,16,20,40,96,128]
    seq_len = [196,384,512,1024,2048,3072,4096]
    hidden_size = [64,96,128,256]

    QV_dims = []
    for b in batch_sizes:
        for head in number_of_heads:
            for s in seq_len:
                for hid in hidden_size:
                    QV_dims.append((b*head,s,hid,s))

    fw_dims = []
    for dim in QV_dims:
        fw_dims.append(dim)
        fw_dims.append((dim[0],dim[1],dim[3],dim[2]))
    QV_dims = None

    fb_dims = []
    for dim in fw_dims:
        fb_dims.append(dim)
        fb_dims.append((dim[0],dim[1],dim[3],dim[2]))
        fb_dims.append((dim[0],dim[2],dim[1],dim[3]))
    fw_dims = None

    fb_dims = list(set(fb_dims))

    # print(f"size : {len(fb_dims)}")

    with open(dims_out, "w") as f:
        for dim in fb_dims:
            f.write(";".join([str(d) for d in dim]) + "\n")

def generate_test_linear(dims_out):
    random.seed(42)
    
    batch_sizes = [1,2,4,8,16,32]
    number_of_heads = [12,16,20,32]
    seq_len = [512,1024,2048]
    hidden_size = [64,80,128]
    vocab_size = [50272, 30522]

    fw_dims = []
    for b in batch_sizes:
        for head in number_of_heads:
            for s in seq_len:
                for hid in hidden_size:
                    fw_dims.append((1, b*s,head*hid,3*head*hid)) # qkv proj
                    fw_dims.append((1, b*s,head*hid,head*hid)) # qkv linear
                    fw_dims.append((1, b*s,head*hid,4*head*hid)) # ff1
                    fw_dims.append((1, b*s,4*head*hid,head*hid)) # ff2

    # lm head
    for b in batch_sizes:
        for head in number_of_heads:
            for s in seq_len:
                for hid in hidden_size:
                    for v in vocab_size:
                        fw_dims.append((1, b*s,head*hid,v)) # qkv proj


    bw_dims = []
    for dim in fw_dims:
        bw_dims.append(dim)
        bw_dims.append((dim[0],dim[1],dim[3],dim[2]))
        bw_dims.append((dim[0],dim[2],dim[1],dim[3]))
    fb_dims = fw_dims + bw_dims
    fw_dims = None
    bw_dims = None

    fb_dims = list(set(fb_dims))

    # print(f"size : {len(fb_dims)}")

    with open(dims_out, "w") as f:
        for dim in fb_dims:
            f.write(";".join([str(d) for d in dim]) + "\n")

def generate_gpt_vec(dims_out):
    random.seed(42)

    dims = {
        "BERT_Base" : (12, 64, 512, (1,2,4,8,16,32,64)),
        "BERT_Large" : (16, 64, 512, (1,2,4,8,16,32,64)),
        "GPT2_Large" : (20, 64, 1024, (1,2,4,8,16,32,64)),
        "GPT2_XL" : (20, 80, 1024, (1,2,4,8,16,32,64)),
        "OPT_13" : (32, 64, 2048, (1,2,4,8,16,32,64)),
        "GPT3_XL" : (24, 128, 2048, (1,2,4,8,16)),
        "GPT3_27" : (32, 80, 2048, (1,2,4,8)),
        "GPT3_67" : (32, 128, 2048, (1,2,4,8)),
    }

    fw_dims = []
    for model, (head, hid, s, batch) in dims.items():
        for b in batch:
            fw_dims.append((b*head*s, s))
            fw_dims.append((b*s, head*hid))
            fw_dims.append((b*s, head*hid*4))
    fw_dims = set(fw_dims)
    fw_dims = list(fw_dims)

    print(f"size : {len(fw_dims)}")

    with open(dims_out, "w") as f:
        for dim in fw_dims:
            f.write(";".join([str(d) for d in dim]) + "\n")

def generate_gpt_vec(dims_out):
    random.seed(42)

    dims = {
        "BERT_Base" : (12, 64, 512, (1,2,4,8,16,32,64), 30522),
        "BERT_Large" : (16, 64, 512, (1,2,4,8,16,32,64), 30522),
        "GPT2_Large" : (20, 64, 1024, (1,2,4,8,16,32,64), 50257),
        "GPT2_XL" : (20, 80, 1024, (1,2,4,8,16,32,64), 50257),
        "OPT_13" : (32, 64, 2048, (1,2,4,8,16,32,64), 50272),
        "GPT3_XL" : (24, 128, 2048, (1,2,4,8,16), 50257),
        "GPT3_27" : (32, 80, 2048, (1,2,4,8), 50257),
        "GPT3_67" : (32, 128, 2048, (1,2,4,8), 50257),
    }

    fw_dims = []
    for model, (head, hid, s, batch, vocab_size) in dims.items():
        for b in batch:
            fw_dims.append((b*head*s, s))
            fw_dims.append((b*s, head*hid))
            fw_dims.append((b*s, head*hid*4))
    fw_dims = set(fw_dims)
    fw_dims = list(fw_dims)

    bw_dims = []
    for model, (head, hid, s, batch, vocab_size) in dims.items():
        for b in batch:
            # accumulation
            total_hid = head*hid

            bw_dims.append((1, hid)) # 1, hid (vec)
            bw_dims.append((1, total_hid)) # 1, hid (vec)
            
            bw_dims.append((1, hid*4)) # 1, 4*hid (vec)
            bw_dims.append((1, total_hid*4)) # 1, 4*hid (vec)
            
            bw_dims.append((1, hid*hid)) # hid, hid (weight)
            bw_dims.append((1, total_hid*total_hid)) # hid, hid (weight)

            bw_dims.append((1, hid*hid*3)) # hid, 4*hid (weight)
            bw_dims.append((1, total_hid*total_hid*3)) # hid, 4*hid (weight)

            bw_dims.append((1, hid*hid*4)) # hid, 4*hid (weight)
            bw_dims.append((1, total_hid*total_hid*4)) # hid, 4*hid (weight)

            bw_dims.append((1, total_hid*vocab_size)) # embedding

            bw_dims.append((1, vocab_size)) # pte

    bw_dims = set(bw_dims)
    bw_dims = list(bw_dims)

    dims = fw_dims + bw_dims
    dims = list(set(dims))

    print(f"size : {len(dims)}")

    with open(dims_out, "w") as f:
        for dim in dims:
            f.write(";".join([str(d) for d in dim]) + "\n")


def generate_train_conv(dims_out):
    random.seed(42)

    batch = 0
    i_c = 0
    o_c = 0
    k_s = 0
    i_s = 0
    stride = 0
    padding = 0

    points = []
    while len(points) < 5000:
        batch = random.randint(1, 64)
        i_c = random.randint(3, 2048)
        o_c = random.randint(3, 2048)
        k_s = random.randint(3, 3)
        i_s = random.randint(5, 256)
        stride = random.randint(1, 1)
        padding = random.randint(1, 1)

        # sanity check
        if k_s > i_s:
            continue

        flops = (2 * i_c * o_c * k_s * k_s * i_s * i_s) / 1e9
        if flops > 1:
            # too many flops
            continue

        points.append((batch, i_c, o_c, k_s, i_s, stride, padding))

    with open(dims_out, "w") as f:
        for dim in points:
            f.write(";".join([str(d) for d in dim]) + "\n")

def generate_resnet_testcase(dims_out):
    random.seed(42)

    # i_c, o_c, k_s, i_s, stride, padding
    dims_map = {
        # "conv0":(3, 64, 7, 224, 2, 3), # output size = 112
        # maxpooling, output size = 56

        "layer1_ds":  (64, 256, 1, 56, 1, 0),
        "layer1_c1_1":(64,  64, 1, 56, 1, 0),
        "layer1_c1":  (256, 64, 1, 56, 1, 0),
        "layer1_c2":  (64,  64, 3, 56, 1, 1),
        "layer1_c3":  (64, 256, 1, 56, 1, 0),
        
        # "layer2_ds":  (256, 512, 1, 56, 2, 0), # output size = 28
        "layer2_c1_1":(256, 128, 1, 56, 1, 0),
        # "layer2_c2_1":(128, 128, 3, 56, 2, 1), # output size = 28
        "layer2_c1":  (512, 128, 1, 28, 1, 0),
        "layer2_c2":  (128, 128, 3, 28, 1, 1),
        "layer2_c3":  (128, 512, 1, 28, 1, 0),

        # "layer3_ds":  (512, 1024, 1, 28, 2, 0), # output size = 14
        "layer3_c1_1":(512,  256, 1, 28, 1, 0),
        # "layer3_c2_1":(256,  256, 3, 28, 2, 1), # output size = 14
        "layer3_c1":  (1024, 256, 1, 14, 1, 0),
        "layer3_c2":  (256,  256, 3, 14, 1, 1),
        "layer3_c3":  (256, 1024, 1, 14, 1, 0),

        # "layer4_ds":  (1024, 2048, 1, 14, 2, 0), # output size = 7
        "layer4_c1_1":(1024,  512, 1, 14, 1, 0),
        # "layer4_c2_1":(512,   512, 3, 14, 2, 1), # output size = 7
        "layer4_c1":  (2048,  512, 1,  7, 1, 0),
        "layer4_c2":  (512,   512, 3,  7, 1, 1),
        "layer4_c3":  (512,  2048, 1,  7, 1, 0),
    }

    dims = []
    batch = [1,2,4,8,16,32,48,64]
    for name, (i_c, o_c, k_s, i_s, stride, padding) in dims_map.items():
        for b in batch:
            dims.append((b, i_c, o_c, k_s, i_s, stride, padding))
    dims = set(dims)
    dims = list(dims)

    print(f"size : {len(dims)}")

    with open(dims_out, "w") as f:
        for dim in dims:
            f.write(";".join([str(d) for d in dim]) + "\n")


import numpy as np

def generate_train_ln(dims_out):
    random.seed(42)

    b_grids = [0,256,512,1024,2048,4096,8192,16384]
    h_grids = [256,512,1024,2048,4096]

    b_grids = np.array(b_grids, dtype=int) // 4
    h_grids = np.array(h_grids, dtype=int) // 4

    b_grid_idx = 0
    h_grid_idx = 0

    b_grid_num = len(b_grids) - 1
    h_grid_num = len(h_grids) - 1

    points_per_grid = 50

    points = []

    for b_grid_idx in range(b_grid_num):
        for h_grid_idx in range(h_grid_num):
            for i in range(points_per_grid):
                b = random.randint(b_grids[b_grid_idx]+1, b_grids[b_grid_idx+1]+1) * 4
                h = random.randint(h_grids[h_grid_idx]+1, h_grids[h_grid_idx+1]+1) * 4
                points.append((b,h))

    with open(dims_out, "w") as f:
        for dim in points:
            f.write(";".join([str(d) for d in dim]) + "\n")
