import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def debug_grid_mask():
    x = np.random.randn(12, 3, 100, 100)
    ratio = 0.5
    rotate = 1
    mode = 1
    # 随着训练的进行，进行grid_mask的概率越小
    # self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5
    # if np.random.rand() > self.prob or not self.training:
    #     return x
    n, c, h, w = x.shape
    # x = x.view(-1, h, w)
    x = x.reshape(-1, h, w)
    hh = int(1.5 * h)
    ww = int(1.5 * w)
    d = np.random.randint(2, h)
    l = min(max(int(d * ratio + 0.5), 1), d - 1)
    mask = np.ones((hh, ww), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    # if self.use_h:
    for i in range(hh // d):
        s = d * i + st_h
        t = min(s + l, hh)
        mask[s:t, :] *= 0

    # if self.use_w:
    for i in range(ww // d):
        s = d * i + st_w
        t = min(s + l, ww)
        mask[:, s:t] *= 0

    r = np.random.randint(rotate)
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask)
    mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

    # mask = torch.from_numpy(mask).to(x.dtype).cuda()
    if mode == 1:
        mask = 1 - mask
    # mask = mask.expand_as(x)
    # if self.offset:
    #     offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(x.dtype).cuda()
    #     x = x * mask + offset * (1 - mask)
    # else:
    x = x * mask
    print(np.unique(mask), mask.shape)
    plt.imshow(mask)
    plt.show()

    # return x.view(n, c, h, w)

def debug_lss_get_geometry_v1():
    # 需要搞清楚这几个参数的含义
    # lidar2img.append(img_meta['lidar2img'])     雷达到相机的外参数
    # camera2ego.append(img_meta['camera2ego'])   相机到车身坐标系
    # camera_intrinsics.append(img_meta['camera_intrinsics'])   相机内参
    # img_aug_matrix.append(img_meta['img_aug_matrix'])   图像进行数据增强的参数
    # lidar2ego.append(img_meta['lidar2ego'])      雷达到车身坐标系
    # lidar2img = np.asarray(lidar2img)            雷达到相机的外参

    rots = camera2ego[..., :3, :3]
    trans = camera2ego[..., :3, 3]
    intrins = camera_intrinsics[..., :3, :3]
    post_rots = img_aug_matrix[..., :3, :3]
    post_trans = img_aug_matrix[..., :3, 3]
    lidar2ego_rots = lidar2ego[..., :3, :3]
    lidar2ego_trans = lidar2ego[..., :3, 3]

    # tmpgeom = self.get_geometry(
    #     fH,
    #     fW,
    #     mylidar2img,
    #     img_metas,
    # )

    # 从图像预测离散的深度,通过外参得到lidar坐标系下的点
    geom = self.get_geometry_v1(
        fH,
        fW,
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        lidar2ego_rots,
        lidar2ego_trans,
        img_metas
    )





def debug_lss_mlp_input():

    # camera2ego, camera_intrinsics, post_rots, post_trans 分别为相机到ego的外参， 相机的内参，图像数据增强部分的外参
    def get_mlp_input(self, sensor2ego, intrin, post_rot, post_tran):
        B, N, _, _ = sensor2ego.shape
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],   # fx
            intrin[:, :, 1, 1],   # fy
            intrin[:, :, 0, 2],   # cx
            intrin[:, :, 1, 2],   # cy
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
        ], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

# 调试匈牙利匹配
def debug_linear_sum_assignment():
    from scipy.optimize import linear_sum_assignment
    # 按照cost进行分配，M,N的输入大小，输出的index长度为min(M, N),按照不重复分配的原则，有些是没有分配到gt的

    cost = np.array([[4, 1, 3, 3], [2, 0, 5, 5], [3, 2, 2, 6]])
    cost = cost.transpose(1, 0)
    print(cost.shape)
    row_ind, col_ind = linear_sum_assignment(cost)
    print(cost)
    print(row_ind)  # 开销矩阵对应的行索引
    print(col_ind)  # 对应行索引的最优指派的列索引
    print(cost[row_ind, col_ind])  # 提取每个行索引的最优指派列索引所在的元素，形成数组
    print(cost[row_ind, col_ind].sum())


if __name__ == "__main__":
    print("Start")
    # debug_grid_mask()
    debug_linear_sum_assignment()
    print("End")