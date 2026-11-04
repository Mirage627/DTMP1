if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import multiprocessing
import os
import shutil
import click
import pathlib
import h5py
from tqdm import tqdm
import collections
import pickle
import json
from EfficientFlow.common.robomimic_util import RobomimicAbsoluteActionConverter

#我加的

# # ================= 兼容所有版本的终极修复代码 =================
# import robosuite
# from robosuite.environments.manipulation.stack import Stack

# # 尝试获取环境注册字典
# # 1.4.x 通常在 environments.base 中
# try:
#     from robosuite.environments.base import REGISTERED_ENVS
# except ImportError:
#     # 1.2.x 或更早版本可能直接在 robosuite.environments 中
#     try:
#         from robosuite.environments import REGISTERED_ENVS
#     except ImportError:
#         # 如果都找不到，创建一个假字典防止报错（虽然这样注册会失败，但程序不会崩）
#         print("警告：无法找到环境注册表 REGISTERED_ENVS")
#         REGISTERED_ENVS = {}

# # 强行��册 Stack_D0
# REGISTERED_ENVS["Stack_D1"] = Stack
# # 如果存在 ALL_ENVIRONMENTS 列表，也加进去
# if hasattr(robosuite, "ALL_ENVIRONMENTS"):
#      if "Stack_D1" not in robosuite.ALL_ENVIRONMENTS:
#         robosuite.ALL_ENVIRONMENTS.append("Stack_D1")

# print("已通过底层字典注册 Stack_D1")
# # ==========================================================



import os
import robosuite
# 导入以确保所有的原生操纵环境已被加载并注册到 REGISTERED_ENVS 中
import robosuite.environments.manipulation
# ================= 兼容所有版本的获取字典 =================
try:
    from robosuite.environments.base import REGISTERED_ENVS
except ImportError:
    try:
        from robosuite.environments import REGISTERED_ENVS
    except ImportError:
        print("警告：无法找到环境注册表 REGISTERED_ENVS")
        REGISTERED_ENVS = {}

# ================= 动态注册环境函数 =================
def register_env_from_name(dataset_name_or_path):
    """
    根据数据集名称或文件路径动态注册环境
    例如: "Stack_D4" -> 映射到 "Stack"
          "/data/Lift_D7.hdf5" -> 映射到 "Lift"
    """
    # 1. 如果传入的是路径，提取文件名并去掉后缀 (如 "/a/b/Lift_D7.hdf5" -> "Lift_D7")
    base_name = os.path.basename(dataset_name_or_path).split('.')[0]
    
    # 2. 提取原生的环境名字 (如 "Stack_D4" -> "Stack")
    # robosuite 的标准环境名通常没有下划线，所以我们可以通过 '_D' 或者 '_' 来截断
    if "_D" in base_name:
        original_env_name = base_name.split('_D')[0]
    else:
        original_env_name = base_name.split('_')[0]

    # 3. 检查并获取原生环境的类
    if original_env_name not in REGISTERED_ENVS:
        # 添加一点容错机制，如果传了纯小写 (stack)，尝试转为首字母大写 (Stack)
        if original_env_name.capitalize() in REGISTERED_ENVS:
            original_env_name = original_env_name.capitalize()
        else:
            print(f"警告：原生环境 '{original_env_name}' 不存在于 Robosuite 中，无法映射！")
            return
            
    # 提取原生环境类
    env_class = REGISTERED_ENVS[original_env_name]
    
    # 4. 强行注册带后缀的新名字
    REGISTERED_ENVS[base_name] = env_class
    
    # 5. 更新 ALL_ENVIRONMENTS 列表
    if hasattr(robosuite, "ALL_ENVIRONMENTS"):
        if base_name not in robosuite.ALL_ENVIRONMENTS:
            robosuite.ALL_ENVIRONMENTS.append(base_name)
            
    print(f"✅ 已动态注册环境: {base_name} (底层映射自原生环境: {original_env_name})")

#===================================================================================================



def worker(x):
    path, idx, do_eval = x
    converter = RobomimicAbsoluteActionConverter(path)
    if do_eval:
        abs_actions, info = converter.convert_and_eval_idx(idx)
    else:
        abs_actions = converter.convert_idx(idx)
        info = dict()
    return abs_actions, info

@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-o', '--output', required=True, help='output hdf5 path. Parent directory must exist')
@click.option('-e', '--eval_dir', default=None, help='directory to output evaluation metrics')
@click.option('-n', '--num_workers', default=None, type=int)
def main(input, output, eval_dir, num_workers):
    # process inputs
    input = pathlib.Path(input).expanduser()
    #print(f"Input: {input}")
    assert input.is_file()
    output = pathlib.Path(output).expanduser()
    assert output.parent.is_dir()
    assert not output.is_dir()

    do_eval = False
    if eval_dir is not None:
        eval_dir = pathlib.Path(eval_dir).expanduser()
        assert eval_dir.parent.exists()
        do_eval = True
    
    converter = RobomimicAbsoluteActionConverter(input)

    # run
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker, [(input, i, do_eval) for i in range(len(converter))])
    
    # save output
    print('Copying hdf5')
    shutil.copy(str(input), str(output))

    # modify action
    with h5py.File(output, 'r+') as out_file:
        for i in tqdm(range(len(converter)), desc="Writing to output"):
            abs_actions, info = results[i]
            demo = out_file[f'data/demo_{i}']
            demo['actions'][:] = abs_actions
        # Actions now contain absolute controller goals; retain that meaning
        # when downstream tools construct an environment from this dataset.
        out_file['data'].attrs['env_args'] = json.dumps(converter.abs_env.serialize())
    
    # save eval
    if do_eval:
        eval_dir.mkdir(parents=False, exist_ok=True)

        print("Writing error_stats.pkl")
        infos = [info for _, info in results]
        pickle.dump(infos, eval_dir.joinpath('error_stats.pkl').open('wb'))

        print("Generating visualization")
        metrics = ['pos', 'rot']
        metrics_dicts = dict()
        for m in metrics:
            metrics_dicts[m] = collections.defaultdict(list)

        for i in range(len(infos)):
            info = infos[i]
            for k, v in info.items():
                for m in metrics:
                    metrics_dicts[m][k].append(v[m])

        from matplotlib import pyplot as plt
        plt.switch_backend('PDF')

        fig, ax = plt.subplots(1, len(metrics))
        for i in range(len(metrics)):
            axis = ax[i]
            data = metrics_dicts[metrics[i]]
            for key, value in data.items():
                axis.plot(value, label=key)
            axis.legend()
            axis.set_title(metrics[i])
        fig.set_size_inches(10,4)
        fig.savefig(str(eval_dir.joinpath('error_stats.pdf')))
        fig.savefig(str(eval_dir.joinpath('error_stats.png')))


if __name__ == "__main__":
    main()
