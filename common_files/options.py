# training arguments

import argparse
NPL = 0.0
PL =0.0
def args_parser():
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument('--image_height', type=int, default=256,
                        help="image_height")
    parser.add_argument('--image_width', type=int, default=256,
                        help="image_width")

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=10,
                        help="number of global rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=12,
                        help="the number of local epochs: E")
    parser.add_argument('--val_global_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=1,
                        help="local batch size: B")
    parser.add_argument('--lr_fixed', type=float, default=0.000098,
                        help='learning rate')
    parser.add_argument('--lr_PL', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--mode', type = str, default='fedprox',
                        help='fedavg | fedprox | fedbn')
    parser.add_argument('--mupure', type=float, default=0.01,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--mubad', type=float, default=0.07,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--pack_prob_up_C1', type=float, default=NPL,
                        help='Packet loss probability: Uplink')
    parser.add_argument('--pack_prob_down_C1', type=float, default=NPL,
                        help='Packet loss probability: Downlink')
    parser.add_argument('--pack_prob_up_C2', type=float, default=NPL,
                        help='Packet loss probability: Uplink')
    parser.add_argument('--pack_prob_down_C2', type=float, default=NPL,
                        help='Packet loss probability: Downlink')
    parser.add_argument('--pack_prob_up_C3', type=float, default=NPL,
                        help='Packet loss probability: Uplink')
    parser.add_argument('--pack_prob_down_C3', type=float, default=NPL,
                        help='Packet loss probability: Downlink')
    parser.add_argument('--pack_prob_up_C4', type=float, default=NPL,
                        help='Packet loss probability: Uplink')
    parser.add_argument('--pack_prob_down_C4', type=float, default=NPL,
                        help='Packet loss probability: Downlink')
    parser.add_argument('--pack_prob_up_C5', type=float, default=PL,
                        help='Packet loss probability: Uplink')
    parser.add_argument('--pack_prob_down_C5', type=float, default=PL,
                        help='Packet loss probability: Downlink')
    parser.add_argument('--max_retra_deep', type=float, default=8,
                        help='Maximum no.of retransmission attempts in the deep split')
    parser.add_argument('--max_retra_shallow', type=float, default=9,
                        help='Maximum no.of retransmission attempts in the shallow split')

    # other arguments
    parser.add_argument('--num_classes', type=int, default=5, help="number \
                        of classes")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
