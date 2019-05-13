import numpy as np
from matplotlib import pyplot as plt
import json
import seaborn as sns
import os
import glob

sns.set_style('ticks')


class Plotter(object):
    markers = ('+', 'x', 'v', 'o', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    color_list = ['b', 'r', 'g', 'm', 'y', 'k', 'cyan', 'plum', 'darkgreen', 'darkorange', 'oldlace', 'chocolate',
                  'purple', 'lightskyblue', 'gray', 'seagreen', 'antiquewhite',
                  'snow', 'darkviolet', 'brown', 'skyblue', 'mediumaquamarine', 'midnightblue', 'darkturquoise',
                  'sienna', 'lightsteelblue', 'gold', 'teal', 'blueviolet', 'mistyrose', 'seashell', 'goldenrod',
                  'forestgreen', 'aquamarine', 'linen', 'deeppink', 'darkslategray', 'mediumseagreen', 'dimgray',
                  'mediumpurple', 'lightgray', 'khaki', 'dodgerblue', 'papayawhip', 'salmon', 'floralwhite',
                  'lightpink', 'gainsboro', 'coral', 'indigo', 'darksalmon', 'royalblue', 'navy', 'orangered',
                  'cadetblue', 'orchid', 'palegreen', 'magenta', 'honeydew', 'darkgray', 'palegoldenrod', 'springgreen',
                  'lawngreen', 'palevioletred', 'olive', 'red', 'lime', 'yellowgreen', 'aliceblue', 'orange',
                  'chartreuse', 'lavender', 'paleturquoise', 'blue', 'azure', 'yellow', 'aqua', 'mediumspringgreen',
                  'cornsilk', 'lightblue', 'steelblue', 'violet', 'sandybrown', 'wheat', 'greenyellow', 'darkred',
                  'mediumslateblue', 'lightseagreen', 'darkblue', 'moccasin', 'lightyellow', 'turquoise', 'tan',
                  'mediumvioletred', 'mediumturquoise', 'limegreen', 'slategray', 'lightslategray', 'mintcream',
                  'darkgreen', 'white', 'mediumorchid', 'firebrick', 'bisque', 'darkcyan', 'ghostwhite', 'powderblue',
                  'tomato', 'lavenderblush', 'darkorchid', 'cornflowerblue', 'plum', 'ivory', 'darkgoldenrod', 'green',
                  'burlywood', 'hotpink', 'cyan', 'silver', 'peru', 'thistle', 'indianred', 'olivedrab',
                  'lightgoldenrodyellow', 'maroon', 'black', 'crimson', 'darkolivegreen', 'lightgreen', 'darkseagreen',
                  'lightcyan', 'saddlebrown', 'deepskyblue', 'slateblue', 'whitesmoke', 'pink', 'darkmagenta',
                  'darkkhaki', 'mediumblue', 'beige', 'blanchedalmond', 'lightsalmon', 'lemonchiffon', 'navajowhite',
                  'darkslateblue', 'lightcoral', 'rosybrown', 'fuchsia', 'peachpuff']

    def plot_fig(self, fig_num, col_id, x, y, title, x_lable, y_label, label=' ', marker='*'):
        plt.figure(fig_num, figsize=(6, 5))
        plt.title(title)
        plt.xlabel(x_lable)
        plt.ylabel(y_label)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout()

        marker_every = max(int(len(x) / 10), 1)
        if len(np.array(y).shape) > 1:
            new_shape = np.array(y).shape

            res = np.reshape(np.reshape(np.array([y]), newshape=[-1]), newshape=[new_shape[1], new_shape[0]],
                             order='F').tolist()
            res = list(res)
            for i in range(len(res)):
                res_i = res[i]
                plt.subplot(len(res), 1, i + 1)
                plt.title(title + '_' + str(i))
                plt.plot(x, res_i, self.color_list[col_id], label=label + '_' + str(i), marker=marker,
                         markevery=marker_every, markersize=6, linewidth=1)
                col_id += 1
        else:
            plt.plot(x, y, self.color_list[col_id], label=label, marker=marker, markevery=marker_every, markersize=6,
                     linewidth=1)
        plt.legend()

    @staticmethod
    def plot_any_key_in_log(file_name, key, index, scatter_flag=False, save_flag=False,
                            save_path=None,
                            path_list=None,
                            res_dict=None,
                            res_name=None,
                            histogram_flag=False,
                            his_bins=1,
                            sub_graph_flag=False,
                            value_range=None,
                            average_over=1,
                            fig_id=4, label='', restrict_dict=None, fn=None):
        if not path_list:
            with open(res_dict[res_name], 'r') as f:
                path_list = json.load(f)
        plt.figure(fig_id)
        plt.title("%s_%s_%s" % (res_name, file_name, key))
        plt.xlabel('index')
        plt.ylabel(key)
        total_graph = len(path_list)
        for i in range(len(path_list)):
            print("Load {}".format(path_list[i]))
            test_reward = []
            real_env_sample_count_index = []
            with open(file=path_list[i] + '/loss/' + file_name, mode='r') as f:
                test_data = json.load(fp=f)
                if value_range:
                    test_data = test_data[value_range[0]: min(len(test_data), value_range[1])]
                for sample in test_data:
                    if fn:
                        if fn(sample) is not None:
                            test_reward.append(fn(sample))
                            real_env_sample_count_index.append(sample[index] // average_over)
                    else:
                        if restrict_dict is not None:
                            flag = True
                            for re_key, re_value in restrict_dict.items():
                                if sample[re_key] != re_value:
                                    flag = False
                            if flag is True:
                                test_reward.append(sample[key])
                                real_env_sample_count_index.append(sample[index] // average_over)
                        else:
                            if sample[key]:
                                test_reward.append(sample[key])
                                real_env_sample_count_index.append(sample[index] // average_over)

            x_keys = []
            y_values = []
            last_key = real_env_sample_count_index[0]
            last_set = []

            for j in range(len(real_env_sample_count_index)):
                if real_env_sample_count_index[j] == last_key:
                    last_set.append(test_reward[j])
                else:
                    x_keys.append(last_key)
                    y_values.append(last_set)
                    last_key = real_env_sample_count_index[j]
                    last_set = [test_reward[j]]
            y_values_mean = [np.mean(y_values[j]) for j in range(len(y_values))]
            if sub_graph_flag is True:
                plt.subplot(total_graph, 1, i + 1)
            if scatter_flag is True:
                plt.scatter(x_keys, y_values_mean, c=Plotter.color_list[i], label=key + label + str(i),
                            marker=Plotter.markers[i])
            elif histogram_flag is True:
                plt.hist(y_values_mean, bins=his_bins, label=key + label + str(i))

            else:
                plt.plot(x_keys, y_values_mean, c=Plotter.color_list[i], label=key + label + str(i),
                         marker=Plotter.markers[i])
            plt.legend()

        if save_flag is True:
            for path in path_list:
                plt.savefig(path + '/loss/' + '/%s_%s.png' % (file_name, key))
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_any_scatter_in_log(res_dict, res_name, file_name, key, index, op, scatter_flag=False, save_flag=False,
                                save_path=None,
                                fig_id=4, label='', restrict_dict=None):
        with open(res_dict[res_name], 'r') as f:
            path_list = json.load(f)
        plt.figure(fig_id)
        plt.title("%s_%s_%s" % (res_name, file_name, key))
        plt.xlabel('index')
        plt.ylabel(key)
        for i in range(len(path_list)):
            test_reward = []
            real_env_sample_count_index = []
            with open(file=path_list[i] + '/loss/' + file_name, mode='r') as f:
                test_data = json.load(fp=f)
                for sample in test_data:
                    if restrict_dict is not None:
                        flag = True
                        for re_key, re_value in restrict_dict.items():
                            if sample[re_key] != re_value:
                                flag = False
                        if flag is True:
                            test_reward.append(sample[key])
                            real_env_sample_count_index.append(sample[index])
                    else:
                        test_reward.append(sample[key])
                        real_env_sample_count_index.append(sample[index])
            test_reward, real_env_sample_count_index = op(test_reward, real_env_sample_count_index)
            x_keys = []
            y_values = []
            last_key = real_env_sample_count_index[0]
            last_set = []

            for j in range(len(real_env_sample_count_index)):
                if real_env_sample_count_index[j] == last_key:
                    last_set.append(test_reward[j])
                else:
                    x_keys.append(last_key)
                    y_values.append(last_set)
                    last_key = real_env_sample_count_index[j]
                    last_set = [test_reward[j]]
            x_keys.append(last_key)
            y_values.append(last_set)
            y_values_mean = [np.mean(y_values[j]) for j in range(len(y_values))]
            if scatter_flag is True:
                plt.scatter(x_keys, y_values_mean, c=Plotter.color_list[i], label=key + label + str(i),
                            marker=Plotter.markers[i])
            else:
                plt.plot(x_keys, y_values_mean, c=Plotter.color_list[i], label=key + label + str(i),
                         marker=Plotter.markers[i])

        plt.legend()
        if save_flag is True:
            for path in path_list:
                plt.savefig(path + '/loss/' + '/%s_%s.png' % (file_name, key))
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
