from collections import defaultdict
import os
import pprint
from typing import Any, Dict, List

import seaborn as sns
import pandas as pd
import numpy as np
from scipy import interpolate

from . import json_utils
from . import dict_utils as du


def simple_lineplot(
    dirs: List[str],
    x_dict_path: str,  # 'additional_info.ues_count' example
    y_dict_path: str,  # 'incremental_events.task_lost_rate' example
    xlabel: str,
    ylabel: str,
    group_dict_path: str = None,
    group_label: str = None,
    plot_title: str = None,
    smooth_curve: bool = False,
    smooth_config: Dict[str, Any] = {},
    save_fig_file: str = 'fig.png',
    **kwargs,
) -> None:
    """Lineplots the data from directories.

    Args:
        dirs (List[str]): Dirs from which to collect data

        x_dict_path (str): Dict path to x value.

        y_dict_path (str): Dict path to y value.

        xlabel (str): Label for the x axis

        ylabel (str): Label for the y axis

        group_dict_path (str, optional): Path to group value (is used to
                distinguish the lines). Defaults to None.

        group_label (str, optional): Label for group values. Defaults
                to None.

        plot_title (str, optional): Plot title. Defaults to None.

        smooth_curve (bool, optional): Whether to smooth curve or not.
                Defaults to False.

        smooth_config (Dict[str, Any], optional): Config for curve smoothing.
                Defaults to {}.

        save_fig_file (str, optional): Path to file where to store fig.
                Defaults to 'fig.png'.
    """
    data = load_data_dirs(dirs, x_dict_path, y_dict_path, xlabel, ylabel,
                          False, group_dict_path, group_label)
    if smooth_curve:
        data = smooth_curve_df(
                data, xlabel, ylabel, group_label, **smooth_config)

    p = sns.lineplot(data=data, x=xlabel, y=ylabel, hue=group_label,
                     style=group_label, **kwargs)

    fig = p.get_figure()

    if plot_title is not None:
        p.set_title(plot_title)

    fig.savefig(save_fig_file)


def smooth_curve(x: list, y: list, points=100, k=3, t=None, bc_type=None,
                    axis=0, check_finite=True):
    """
    Smooth the curve.

    Args:
        x (list): arraylike x data
        y (list): arraylike y data
        points (int, optional): how many point to generate on x axis.
                                Defaults to 100.
        other params: same as make_interp_spline

    Returns:
        Tuple: new x,y data
    """
    x_new = np.linspace(min(x), max(x), points)
    a_BSpline = interpolate.make_interp_spline(x, y, k=k, t=t,
                    bc_type=bc_type, axis=axis, check_finite=check_finite)
    return x_new, a_BSpline(x_new)

def smooth_curve_df(
    data: pd.DataFrame,
    xcol: str, ycol: str, groupcol: str = None,
    points=100, k=3, t=None, bc_type=None, axis=0, check_finite=True
) -> pd.DataFrame:
    """
    Smooth the curves from DataFrame.

    Args:
        data (pd.DataFrame): Data.
        xcol (str): x column label.
        ycol (str): y column label.
        groupcol (str): group column label. Used to group results.
        points (int, optional): how many point to generate on x axis.
                                Defaults to 100.
        other params: same as make_interp_spline

    Returns:
        Tuple: new x,y data
    """
    if groupcol is None:
        x,y = smooth_curve(data[xcol], data[ycol], points, k, t, bc_type, axis,
                                check_finite)
        return pd.DataFrame(
            data= {
                xcol: x,
                ycol: y
            }
        )

    groups = set(data[groupcol].tolist())
    new_data = pd.DataFrame(columns=data.columns)
    for group in groups:
        d = data[data[groupcol] == group]
        x, y = smooth_curve(
                d[xcol], d[ycol], points, k, t, bc_type, axis,check_finite)
        dict_ = {
            xcol: x,
            ycol: y,
            groupcol: [group for _ in range(len(x))]
        }
        new_data = new_data.append(pd.DataFrame(dict_), ignore_index=True)
    return new_data


def _load_data_dir(
    dir: str,
    x_dict_path: str,
    y_dict_path: str,
    group_dict_path: str = None,
    sort_descending: bool = False,
    filter = lambda x: True,
) -> Dict[str, list]:
    """
    Loads x and y data from results in dir.

    Args:
        dir (str): Directory containing results.
        x_dict_path (str): Dict path to x value.
        y_dict_path (str): Dict path to y value.
        group_dict_path (str): Path to group value (is used to distinguish
                the lines).
        sort_descending (bool, optional): If sort in descending order.
                Defaults to False.

    Returns:
        Tuple[list, list]: x, y data
    """
    loaded = [json_utils.load_dict(os.path.join(dir, f))
                for f in os.listdir(dir) if f.endswith('.json') and filter(f)]
    loaded = sorted(
        loaded,
        key=lambda x: du.path_get(x, x_dict_path),
        reverse=sort_descending,
    )
    x = list(du.path_get(d, x_dict_path) for d in loaded)
    y = list(du.path_get(d, y_dict_path) for d in loaded)
    data = {
        x_dict_path: x,
        y_dict_path: y,
    }
    if group_dict_path is not None:
        grouping = list(du.path_get(d, group_dict_path) for d in loaded)
        data[group_dict_path] = grouping
    return data


def load_data_dirs(
    dirs: List[str],
    x_dict_path: str,
    y_dict_path: str,
    xlabel: str,
    ylabel: str,
    sort_descending: bool = False,
    group_dict_path: str = None,
    group_label: str = None,
    filter = lambda x: True,
) -> pd.DataFrame:
    """
    Loads data from dirs.

    Args:
        dirs (List[str]): Directories containing results.
        x_dict_path (str): Dict path to x value.
        y_dict_path (str): Dict path to y value.
        xlabel (str): Label for the x axis
        ylabel (str): Label for the y axis
        sort_descending (bool, optional): If sort in descending order.
                Defaults to False.
        group_dict_path (str, optional): Path to group value (is used to
                distinguish the lines). Defaults to None.
        group_label (str, optional): Label for group values. Defaults to None.
    Returns:
        Tuple[list, list]: x, y data
    """
    cols = [x_dict_path, y_dict_path]
    sort_cols = [x_dict_path]

    if group_dict_path is not None:
        cols.append(group_dict_path)
        sort_cols.insert(0, group_dict_path)

    data = pd.DataFrame(columns=cols)

    for dir in dirs:
        d = _load_data_dir(dir, x_dict_path, y_dict_path,
                           group_dict_path, sort_descending, filter)
        df = pd.DataFrame(d)
        data = data.append(df, ignore_index=True)

    data = data.sort_values(sort_cols)

    data.rename(
        {
            x_dict_path: xlabel,
            y_dict_path: ylabel,
            group_dict_path: group_label,
        },
        axis=1,
        inplace=True
    )

    return data

def agregate_mean(
    dir: str,
    to_agregate: List[str],
    out_dir: str,
) -> None:
    agregations = defaultdict(lambda: {'data': {}, 'count': 0})
    for file in os.listdir(dir):

        if not file.endswith('.json'):
            continue

        file_path = os.path.join(dir, file)
        dict_ = json_utils.load_dict(file_path)

        file_id = ''.join(file.split('-')[:-1])

        agregations[file_id]['count'] += 1

        for to_agr_path in to_agregate:
            current = du.path_get(agregations[file_id]['data'], to_agr_path, None)

            if current is None:
                agregations[file_id]['data'] = dict_
                continue

            to_agr = du.path_get(dict_, to_agr_path)
            new = current + 1 / agregations[file_id]['count'] * (to_agr - current)

            du.path_set(agregations[file_id]['data'], to_agr_path, new)

    for file_name in agregations:
        file_path = os.path.join(out_dir, file_name)
        json_utils.save_dict(agregations[file_name]['data'], file_path+'.json')
