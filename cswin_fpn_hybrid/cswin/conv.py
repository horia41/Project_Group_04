def conv_filter(state_dict, patch_size=16):
    """ 
    convert patch embedding weight from manual patchify + linear proj to conv
    """
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict
