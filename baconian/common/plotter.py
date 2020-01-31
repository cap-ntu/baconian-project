import numpy as np
from matplotlib import pyplot as plt
import json
import seaborn as sns

sns.set_style('whitegrid')


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

    def plot_fig(self, fig_num, col_id, x, y, title, x_label, y_label, label=' ', marker='*'):
        plt.figure(fig_num, figsize=(6, 5))
        plt.title(title)
        plt.xlabel(x_label)
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
    def plot_any_key_in_log(data, key, index, exp_num=1,
                            sub_log_dir_name=None,
                            scatter_flag=False,
                            histogram_flag=False,
                            save_flag=False,
                            save_path=None,
                            save_format='png',
                            file_name=None,
                            separate_exp_flag=True,
                            mean_stddev_flag=False,
                            path_list=None,
                            ):
        """
        :param data: a pandas DataFrame containing (index and) key columns
        :param key: in y-axis, the variable to plot, assigned by user
        :param index: in x-axis, the argument assigned by user
        :param sub_log_dir_name: the sub-directory which the log file to plot is saved in
        :param exp_num: [optional] the number of experiments to visualize
        :param scatter_flag: [optional] draw scatter plot if true
        :param histogram_flag: [optional] draw histogram if true
        :param save_flag: [optional] save the figure to a file if true
        :param save_path: [optional] save path of figure, assigned by user, the directory of log_path by default
        :param save_format: [optional] format of figure to save, png by default
        :param file_name: [optional] the file name of the file to save, key_VERUS_index by default
        :param separate_exp_flag: [optional] plot the results of each experiment separately if true
        :param mean_stddev_flag: [optional] plot the mean value of multiple experiment results and standard deviation
        :param path_list: [optional] the list of save paths assigned by users, figure file will be saved to each path
        :return:
        """

        marker_every = max(int(data.shape[0] / 10), 1)

        # plt.figure()
        fig, ax = plt.subplots(1)

        if separate_exp_flag is True:
            for i in range(exp_num):
                if scatter_flag is True:
                    ax.scatter(data[index], data.iloc[:, i + 1], lw=1, label=key + '_' + str(i),
                               c=Plotter.color_list[i], alpha=0.8, )
                elif histogram_flag is True:
                    num_bins = 20
                    n, bins, patches = ax.hist(x=data[0], bins=num_bins)
                else:
                    ax.plot(data[index], data.iloc[:, i + 1], lw=1, label=key + '_' + str(i),
                            color=Plotter.color_list[i],
                            marker=Plotter.markers[i], markevery=marker_every, markersize=6, )
        if mean_stddev_flag is True:
            if histogram_flag is not True:
                ax.plot(data[index], data['MEAN'], lw=3, label='MEAN', color='silver')
                ax.fill_between(data[index], data['MEAN'] + data['STD_DEV'], data['MEAN'] - data['STD_DEV'],
                                facecolor='silver', alpha=0.5)

        plt.title(sub_log_dir_name)
        lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=3)
        if histogram_flag is not True:
            plt.xlabel(index)
            plt.ylabel(key)
        else:
            plt.xlabel(key)
            plt.ylabel('count')
        plt.legend()
        # Save the figure to a file to a path or paths in a list
        if save_flag is True:
            if file_name is None:
                file_name = '/%s_VERSUS_%s' % (key, index)

            if path_list is not None:
                for path in path_list:
                    plt.savefig(path + '/%s.%s' % (file_name, save_format), bbox_extra_artists=(lgd,),
                                bbox_inches='tight', format=save_format)
                    print("Save plot figure to {path} as {file_name}".format(path=path,
                                                                             file_name='%s.%s' % (
                                                                                 file_name, save_format)))
            if save_path is not None:
                plt.savefig(save_path + '/%s.%s' % (file_name, save_format), bbox_extra_artists=(lgd,),
                            bbox_inches='tight', format=save_format)
                print("Save plot figure to {path} as {file_name}".format(path=save_path,
                                                                         file_name='%s.%s' % (
                                                                             file_name, save_format)))
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
                plt.savefig(path + '/%s_%s.png' % (file_name, key))
        if save_path is not None:
            plt.savefig(save_path)
            print()
        plt.show()
