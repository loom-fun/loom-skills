import numpy as np
def order_idx(order, gh, gw):
    if order == "raster": return np.repeat(np.arange(gh), gw), np.tile(np.arange(gw), gh)
    o = [(ty * 2 + dy, tx * 2 + dx) for ty in range(gh // 2) for tx in range(gw // 2) for dy in range(2) for dx in range(2)]
    return np.array([a for a, b in o]), np.array([b for a, b in o])
