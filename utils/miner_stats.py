def count_pair_masks(miner):
    """
    Count the number of True entries in each pair mask
    (EP, EN, HP, HN) stored inside an EasyHardMiner.

    Returns
    -------
    tuple(int, int, int, int)
        (n_EP, n_EN, n_HP, n_HN)
    """
    # easy positives  (same label & same cluster)
    n_ep = miner.EP.sum().item()
    # easy negatives  (diff label & diff cluster)
    n_en = miner.EN.sum().item()
    # hard positives  (same label & diff cluster)
    n_hp = miner.HP.sum().item()
    # hard negatives  (diff label & same cluster)
    n_hn = miner.HN.sum().item()

    return n_ep, n_en, n_hp, n_hn
